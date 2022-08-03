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
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0.5]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --verbose                            Show structure plots at end of solve
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

def kramers_opacity_polytrope(nz, γ, ε, n_h, verbose=False, dealias=2,
                              ncc_cutoff = 1e-10,tolerance = 1e-8, comm=None):

    import dedalus.public as de

    cP = γ/(γ-1)
    m_ad = 1/(γ-1)

    s_c_over_c_P = scrS = 1 # s_c/c_P = 1

    h_bot = 1.5
    h_slope = -1/(1+m_ad)
    grad_φ = (γ-1)/γ

    κ_00 = 1
    σ_sb = 1

    aa=1
    bb=-3.5
    n = (3-bb)/(aa+1)
    κ00=1
    σ_sb=1
    κ_const = 16*σ_sb/(3*κ00)

    Lz = -1/h_slope*(1-np.exp(-n_h))
    
    print("All input parameters")
    print("nz:{}".format(nz) + "\nγ: {}".format(γ) + "\nε: {}".format(ε))
    print("n_h:{}".format(n_h) + "\ncP: {}".format(cP) + "\nm_ad: {}".format(m_ad))
    print("h_bot:{}".format(h_bot) + "\nh_slope: {}".format(h_slope) + "\ngrad_φ:".format(grad_φ))
    print("κ0: {}".format(κ00) + "\nσ_sb: {}".format(σ_sb) + "\nκ_const: {}".format(κ_const))
    print("Lz: {}".format(Lz))   

    c = de.CartesianCoordinates('z')
    d = de.Distributor(c, comm=comm, dtype=np.float64)
    zb = de.ChebyshevT(c.coords[-1], size=nz, bounds=(0, Lz), dealias=dealias)
    b = zb
    z = zb.local_grid(1)
    zd = zb.local_grid(dealias)

    # Fields
    θ = d.Field(name='θ', bases=b)
    Y = d.Field(name='Y', bases=b)
    s = d.Field(name='s', bases=b)
    κ = d.Field(name='κ', bases=b)

    # Taus
    lift_basis = zb.clone_with(a=zb.a+2,b=zb.b+2)
    lift = lambda A, n: de.Lift(A, lift_basis, n)
    lift_basis1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
    lift1 = lambda A, n: de.Lift(A, lift_basis1, n)
    τ_h1 = d.VectorField(c, name='τ_h1')
    τ_h2 = d.VectorField(c, name='τ_h2')
    τ_s1 = d.Field(name='τ_s1')
    τ_s2 = d.Field(name='τ_s2')
    τ_y1 = d.Field(name='τ_y1')

    # Parameters and operators
    lap = lambda A: de.Laplacian(A, c)
    grad = lambda A: de.Gradient(A, c)
    integ = lambda A: de.Integrate(A, z)
    ez, = c.unit_vector_fields(d)

    # NLBVP goes here
    # initial guess
    h0 = d.Field(name='h0', bases=zb)
    θ0 = d.Field(name='θ0', bases=zb)
    Υ0 = d.Field(name='Υ0', bases=zb)
    s0 = d.Field(name='s0', bases=zb)
    κ0 = d.Field(name='κ0', bases=zb)

    structure = {'h':h0, 's':s0, 'θ':θ0, 'Υ':Υ0, 'κ':κ0}
    for key in structure:
        structure[key].change_scales(dealias)
    h0['g'] = h_bot #+ zd*h_slope #enthalpy
    θ0['g'] = np.log(h0).evaluate()['g'] # log enthalpy
    Υ0['g'] = (m_ad*θ0).evaluate()['g'] # log rho
    s0['g'] = ((-1/m_ad)*Υ0+θ0).evaluate()['g'] # entropy
    κ0['g'] = (κ_const*h0**(3-bb)/(np.exp(Υ0))**(1+aa)).evaluate()['g']

    problem = de.NLBVP([h0, s0, Υ0, τ_s1, τ_s2, τ_h1])
    problem.add_equation((grad(h0) + lift(τ_h1,-1),
                         -grad_φ*ez + h0*grad(s0)))
    problem.add_equation((lap(h0) + lift(τ_s1,-1) + lift(τ_s2,-2)
                         ,- h0*κ_const*grad(np.log(h0))@((3-bb)*grad(θ0)-(1+aa)*grad(Υ0)) -ε/κ0))
    problem.add_equation(((γ-1)*Υ0 + s_c_over_c_P*γ*s0, np.log(h0)))
    problem.add_equation((h0(z=0), h_bot))
    problem.add_equation((h0(z=Lz), 1.1))
    problem.add_equation((Υ0(z=0), 0))

    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info('current perturbation norm = {:.3g}'.format(pert_norm))

    # re-normalize density and entropy (Υ0(z=0)=0, s(z=0)=0)
    Υ0 = (Υ0-Υ0(z=0)).evaluate()
    Υ0.name='Υ0'
    structure['Υ'] = Υ0
    s0 = (s0-s0(z=0)).evaluate()
    s0.name = 's0'
    structure['s'] = s0


    if verbose:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        fig.subplots_adjust(hspace=0.9, wspace=0.4)
        plt.subplot(1,5,1)
        plt.text(-0.45, 1.505, '(b)', fontsize=15)
        plt.gca().set_title(r'$h$')
        plt.xlabel('z')
        plt.plot(zd,h0['g'], linestyle='dashed', color='xkcd:dark grey', label='h')
        plt.subplot(1,5,2)
        plt.gca().set_title(r'$\theta=\log(h)$')
        plt.xlabel('z')
        plt.plot(zd,np.log(h0['g']))


        plt.subplot(1,5,3)
        plt.gca().set_title(r'$\rho$')
        plt.xlabel('z')
        plt.plot(zd,np.exp(Υ0['g']), label=r'$\rho$')
        plt.subplot(1,5,4)
        plt.gca().set_title(r'$Y=\log(\rho)$')
        plt.xlabel('z')
        plt.plot(zd,Υ0['g'])


        plt.subplot(1,5,5)
        plt.gca().set_title(r'$s$')
        plt.xlabel('z')
        plt.plot(zd,s0['g'], color='xkcd:brick red', label=r'$s$')

        plt.savefig('kramers_solve_h_linear_ic2_nh{}_eps{:.3g}_gamma{:.3g}.pdf'.format(n_h,ε,γ),bbox_inches='tight')

    for key in structure:
        structure[key].change_scales(1)

    return structure


if __name__=='__main__':
    from docopt import docopt
    args = docopt(__doc__)
    from fractions import Fraction

    ncc_cutoff = float(args['--ncc_cutoff'])

    #Resolution
    nz = int(args['--nz'])

    γ  = float(Fraction(args['--gamma']))
    m_ad = 1/(γ-1)

    if args['--m']:
        m = float(args['--m'])
        strat_label = 'm{}'.format(args['--m'])
    else:
        m = m_ad - float(args['--epsilon'])
        strat_label = 'eps{}'.format(args['--epsilon'])
    ε = m_ad - m

    n_h = float(args['--n_h'])

    verbose = args['--verbose']

    structure = kramers_opacity_polytrope(nz, γ, ε, n_h, verbose=verbose)
    for key in structure:
        print(structure[key], structure[key]['g'])
