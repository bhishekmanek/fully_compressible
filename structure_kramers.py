"""
Dedalus script for computing equilibrated background for heated
atmospheres (using adiabaitc polytropes to set initial guess),
with specified number of density scale heights of stratification.

The equations are in theta formalism:
1. grad(h) = - grad(phi) +  h*grad(s)
2. lap(h) = -h*k_cons*grad(theta)*[(3-b)*grad(theta)-(1+a)*grad(Y)]-ε/k
3. (gamma-1)*Y + (sc/cp)*gamma*s = log(h)

Usage:
    structure_kramers.py [options]

Options:
    --n_h=<n_h>         Enthalpy scale heights [default: 0.5]
    --gamma=<gamma>     Gamma of ideal gas (cp/cv) [default: 5/3]
    --nz=<nz>           vertical z (chebyshev) resolution [default: 64]

    --non-Kramers       Use a fixed kappa (e.g., non-Kramers)

    --ref_point=<ref>   Reference point, bottom, bot or top [default: bot]

    --ncc_cutoff=<ncc>  Amplitude cutoff for NCCs [default: 1e-8]
    --aa=<aa>           Value of the free parameter a [default: 1.0]
    --bb=<bb>           Value of the free parameter b [default: -3.5]
    --verbose           Show structure plots at end of solve
    --bc_jump=<bc>      Jump in the enthalpy top boundary condition [default: 0]
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
for system in ['evaluator', 'matplotlib']:
    logging.getLogger(system).setLevel(logging.WARNING)
import dedalus.public as de

dealias = 2

def kramers_opacity_polytrope(nz, γ, n_h, aa, bb, bc_jump,
                              ref_point='bottom',
                              Kramers=True,
                              dealias=dealias, ncc_cutoff=1e-10, tolerance=1e-13,
                              comm=None):
    import numpy as np
    #cP = γ/(γ-1)
    # m_ad = 1/(γ-1)
    # s_c_over_c_P = scrS = 1 # s_c/c_P = 1

    # h(z=0) = 1

    grad_φ = (γ-1)/γ

    n = (3-bb)/(aa+1)
    if ref_point=='bottom' or ref_point=='bot':
        h_top = np.exp(-n_h)
        h_bot = 1
    elif ref_point=='top':
        h_top = 1
        h_bot = np.exp(n_h)
    else:
        raise ValueError(f'reference point "{ref_point}" not currently implemented')
    h_slope = -1/(1+n)
    Lz = 1/h_slope*(h_top - h_bot) # polytrope intuition
    if ref_point=='bottom' or ref_point=='bot':
        z_ref = 0
    elif ref_point=='top':
        z_ref = Lz
    θ_top = np.log(h_top)
    θ_bot = np.log(h_bot)

    coords = de.CartesianCoordinates('z')
    dist = de.Distributor(coords, comm=comm, dtype=np.float64)
    zb = de.ChebyshevT(coords.coords[-1], size=nz, bounds=(0, Lz), dealias=dealias)
    bases = zb
    z = dist.local_grid(zb)
    z_grid = dist.Field(name='z_grid', bases=bases)
    z_grid['g'] = z

    # Fields
    θ = dist.Field(name='θ', bases=bases)
    Υ = dist.Field(name='Υ', bases=bases)
    s = dist.Field(name='s', bases=bases)
    κ = dist.Field(name='κ', bases=bases)

    # Taus
    lift_basis2 = zb.derivative_basis(2)
    lift2 = lambda A, n: de.Lift(A, lift_basis2, n)
    lift_basis1 = zb.derivative_basis(1)
    lift1 = lambda A, n: de.Lift(A, lift_basis1, n)
    τ_h1 = dist.VectorField(coords, name='τ_h1')
    τ_s1 = dist.Field(name='τ_s1')
    τ_s2 = dist.Field(name='τ_s2')

    # Parameters and operators
    ez, = coords.unit_vector_fields(dist)

    structure = {'s':s, 'θ':θ, 'Υ':Υ, 'z':z_grid}
    for key in structure:
        structure[key].change_scales(dealias)
    # initial guess: polytrope
    θ['g'] = np.log(np.exp(θ_bot)+z_grid*h_slope).evaluate()['g'] # log enthalpy
    Υ['g'] = (n*θ).evaluate()['g'] # polytrope
    s['g'] = (1/γ*θ - (γ-1)/γ*Υ).evaluate()['g'] # EOS

    δS = bc_jump
    vars = [θ, Υ, s]
    taus = [τ_s1, τ_s2, τ_h1]
    problem = de.NLBVP(vars+taus, namespace=locals())
    # assumes s_c_over_c_P = 1
    problem.add_equation("grad(θ) - grad(s) + lift1(τ_h1,-1) = -grad_φ*ez*np.exp(-θ)")
    if Kramers:
        problem.add_equation("lap(θ) + lift2(τ_s1,-1) + lift2(τ_s2,-2) = -grad(θ)@((4-bb)*grad(θ)-(1+aa)*grad(Υ))")
    else:
        problem.add_equation("lap(θ) + lift2(τ_s1,-1) + lift2(τ_s2,-2) = -grad(θ)@*grad(θ)")
    problem.add_equation("θ - (γ-1)*Υ - γ*s  = 0")
    problem.add_equation("θ(z=0)  = θ_bot")
    problem.add_equation("θ(z=Lz) = θ_top + γ*δS")
    problem.add_equation("Υ(z=z_ref) = 0 ")

    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info('current perturbation norm = {:.3g}'.format(pert_norm))

    for key in structure:
        structure[key].change_scales(1)

    return structure

def plot_structure(structure, polytrope, aa, bb, dealias=dealias, label=None):
    n = (3-bb)/(aa+1)
    # atmosphere values from full solve
    θ = structure['θ']
    Υ = structure['Υ']
    s = structure['s']

    # atmosphere values from related polytrope
    θ_poly = polytrope['θ']
    Υ_poly = polytrope['Υ']
    s_poly = polytrope['s']

    # auxiliary variables for plotting
    h = np.exp(θ).evaluate()
    ρ = np.exp(Υ).evaluate()

    h_poly = np.exp(θ_poly).evaluate()
    ρ_poly = np.exp(Υ_poly).evaluate()

    lnκ = (3-bb)*θ - (1+aa)*Υ
    κ = np.exp(lnκ).evaluate()

    lnκ_poly = (3-bb)*θ_poly - (1+aa)*Υ_poly
    κ_poly = np.exp(lnκ_poly).evaluate()

    κ_0 = '{:.2f}'.format(κ_poly.evaluate()['g'][0])
    lnκ_0 = '{:.2f}'.format(lnκ_poly.evaluate()['g'][0])

    z_grid = structure['z']
    z_grid.change_scales(dealias)
    z = z_grid['g']

    for q in [θ, Υ, s, θ_poly, Υ_poly, s_poly]:
        q.change_scales(dealias)

    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    fig.suptitle(r'Background Stratification, $n = $'+f'{n}'+ ', $\kappa(z) = $'+f'{κ_0}'+', $\lambda(z) = \log(\kappa(z)) = $'+f'{lnκ_0}', fontsize=15)

    axs[0].plot(z, h_poly['g'], color='xkcd:dark grey', label=r'$h$')
    axs[0].legend(fontsize=12,loc='lower left')

    axs_0 = axs[0].twinx()
    axs_0.plot(z, θ_poly['g'], 'r--', label=r'$\theta = \log(h)$')

    axs[1].plot(z, ρ_poly['g'], color='xkcd:dark grey', label=r'$\rho$')
    axs[1].legend(fontsize=12,loc='lower left')

    axs_1 = axs[1].twinx()
    axs_1.plot(z, Υ_poly['g'], 'r--', label=r'$\Upsilon = \log(\rho)$')
    axs_1.tick_params(axis='y', labelcolor='r')
    axs_1.legend(fontsize=12,loc='upper right')

    axs[2].plot(z, s_poly['g'], color='xkcd:dark grey', label=r'$s$')
    axs[2].legend(fontsize=12,loc='upper left')

    for axi in axs:
        axi.set_xlabel('z',fontsize=15)
        axi.tick_params(axis='x', labelsize=10)

    fig.subplots_adjust(wspace=0.6)
    fig.tight_layout()
    fig.savefig('poly.pdf', dpi=300, bbox_inches='tight')

    fig, axs = plt.subplots(ncols=5, figsize=(13,4))
    axs[0].plot(z, h['g']-h_poly['g'], color='xkcd:dark grey', label='h')
    axs[1].plot(z, θ['g']-θ_poly['g'], label='theta')
    axs[2].plot(z, ρ['g']-ρ_poly['g'], label='rho')
    axs[3].plot(z, Υ['g']-Υ_poly['g'], label='Y')
    axs[4].plot(z, s['g']-s_poly['g'], label='s')
    for axi in axs:
        axi.legend()
    fig.tight_layout()
    fig.savefig('pert.pdf',bbox_inches='tight')

    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    axs[0].plot(z, κ_poly['g'], label='kappa_poly')
    axs[1].plot(z, κ['g'], label='kappa_NLBVP')
    axs[2].plot(z, κ['g']-κ_poly['g'], label='perturbation')
    for axi in axs:
        axi.legend()
    fig.tight_layout()
    fig.savefig('kappa.pdf',bbox_inches='tight')

    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    axs[0].plot(z, lnκ_poly['g'], label='lnκ_poly')
    axs[1].plot(z, lnκ['g'], label='lnκ_NLBVP')
    axs[2].plot(z, lnκ['g']-lnκ_poly['g'], label='perturbation')
    for axi in axs:
        axi.legend()
    fig.tight_layout()
    fig.savefig('lambda.pdf',bbox_inches='tight')

    fig, axs = plt.subplots(ncols=6, figsize=(13, 4))
    fig.subplots_adjust(hspace=0.9, wspace=0.4)
    axs[0].text(-0.45, 1.1, '(b) a={:.3g}'.format(aa) + ', b={:.3g}'.format(bb) + ', n={:.3g}'.format(n), fontsize=12)
    axs[0].set_title(r'$h$')
    axs[0].plot(z, h['g'], color='xkcd:dark grey', label='$h$')
    axs[0].plot(z, h_poly['g'], linestyle='dashed', color='red', label=r'$h_\text{poly}$')

    axs[1].set_title(r'$\theta=\log(h)$')
    axs[1].plot(z, θ['g'], label=r'$\log(h)$')

    axs[2].set_title(r'$\rho$')
    axs[2].plot(z, ρ['g'], label=r'$\rho$')
    axs[2].plot(z, ρ_poly['g'], linestyle='dashed', color='blue', label=r'$\rho_\text{poly}$')

    axs[3].set_title(r'$Y=\log(\rho)$')
    axs[3].plot(z, Υ['g'], label=r'$\log(\rho)$')

    axs[4].set_title(r'$s$')
    axs[4].plot(z, s['g'], color='xkcd:brick red', label=r'$s$')
    axs[4].plot(z, s_poly['g'], color='xkcd:dark grey', label=r'$s_\text{poly}$', linestyle='dashed', alpha=0.5)

    axs[5].set_title('kappa')
    axs[5].plot(z, κ['g'], label='kappa')
    for axi in axs:
        axi.set_xlabel('z')
        axi.legend()
    filename = f'kramers_solve_bc_jump{bc_jump}_a{aa}_b{bb:.3g}_n{n:.3g}'
    if label:
        filename += f'_{label}'
    fig.savefig(f'{filename}.pdf',bbox_inches='tight')

if __name__=='__main__':
    from docopt import docopt
    args = docopt(__doc__)
    from fractions import Fraction
    print(args)
    ncc_cutoff = float(args['--ncc_cutoff'])

    #Resolution
    nz = int(args['--nz'])

    #Free parameters (exponents) characterizing Kramers-like opacity (see. Barekat & Brandenburg 2014)
    aa = float(args['--aa'])
    bb = float(args['--bb'])
    bc_jump = float(args['--bc_jump'])
    γ  = float(Fraction(args['--gamma']))
    n_h = float(args['--n_h'])
    ref_point = args['--ref_point']

    if args['--non-Kramers']:
        Kramers = False
    else:
        Kramers = True

    structure = kramers_opacity_polytrope(nz, γ, n_h, aa, bb, bc_jump,
                                          ref_point=ref_point, Kramers=Kramers)

    polytrope = kramers_opacity_polytrope(nz, γ, n_h, aa, bb, 0, ref_point=ref_point)

    if args['--verbose']:
        plot_structure(structure, polytrope, aa, bb, label=ref_point)

    for key, q in structure.items():
        q.change_scales(1)
        print(q, q['g'])
