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
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]
    --aa=<aa>			 	 Value of the free parameter a [default: 1.0]
    --bb=<bb>				 Value of the free parameter b [default: -3.5]
    --verbose                            Show structure plots at end of solve
    --bc_jump=<bc_jump>			 Jump in the enthalpy top boundary condition [default: 0.0] 
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

def kramers_opacity_polytrope(nz, γ, n_h, aa, bb, bc_jump, verbose=False, dealias=2,
                              ncc_cutoff = 1e-10,tolerance = 1e-13, comm=None):

    import dedalus.public as de

    cP = γ/(γ-1)
    m_ad = 1/(γ-1)

    s_c_over_c_P = scrS = 1 # s_c/c_P = 1

    h_bot = 1
    #h_slope = -1/(1+m_ad)
    grad_φ = (γ-1)/γ

    κ_00 = 1
    σ_sb = 1

    n = (3-bb)/(aa+1)
    h_slope = -1/(1+n)
#    κ00=10000000.0
    κ00=1
    σ_sb=1
    κ_const = 16*σ_sb/(3*κ00)

    Lz = -1/h_slope*(1-np.exp(-n_h))
    
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
    integ = lambda A: de.Integrate(A, 'z')
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
    h0['g'] = h_bot + 1.0*zd*h_slope #enthalpy
    θ0['g'] = np.log(h0).evaluate()['g'] # log enthalpy
    Υ0['g'] = (n*θ0).evaluate()['g'] # log rho
    s0['g'] = 0.0#((-1/m_ad)*Υ0+θ0).evaluate()['g'] # entropy - we are starting with a entropy profile that is 0
    κ0['g'] = (κ_const*h0**(3-bb)/(np.exp(Υ0))**(1+aa)).evaluate()['g']

    problem = de.NLBVP([h0, s0, Υ0, τ_s1, τ_s2, τ_h1])
    problem.add_equation((grad(h0) + lift(τ_h1,-1),
                         -grad_φ*ez + h0*grad(s0)))
    problem.add_equation((lap(h0) + lift(τ_s1,-1) + lift(τ_s2,-2)
                         ,-grad(h0)@((3-bb)*grad(np.log(h0))-(1+aa)*grad(Υ0))))
    problem.add_equation(((γ-1)*Υ0 + s_c_over_c_P*γ*s0, np.log(h0)))
    problem.add_equation((h0(z=0), h_bot))
    problem.add_equation((h0(z=Lz), np.exp(-n_h)+bc_jump))
    problem.add_equation((integ(np.exp(Υ0)), 0.5*Lz*(-Lz*h_slope)))
    #problem.add_equation((Υ0(z=0), 0.0))

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

    enth = h_bot - zd/(1+n)
    dens = (enth)**n

    κ0['g'] = (κ_const*h0**(3-bb)/(np.exp(Υ0))**(1+aa)).evaluate()['g']

    #print(h0['g']-enth)
    #print(dens-np.exp(Υ0['g']))

    if verbose:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 5, figsize=(13, 4))
        fig.subplots_adjust(hspace=0.9, wspace=0.4)
        plt.subplot(1,6,1)
        plt.text(-0.45, 1.07, '(b) a={:.3g}'.format(aa) + ', b={:.3g}'.format(bb) + ', n={:.3g}'.format(n), fontsize=12)
        #plt.text(-0.45, 2.13, '(b) a={:.3g}'.format(aa) + ', b={:.3g}'.format(bb) + ', n={:.3g}'.format(n), fontsize=12)
        plt.gca().set_title(r'$h$')
        plt.xlabel('z')
        plt.plot(zd,h0['g'], color='xkcd:dark grey', label='h')
        plt.plot(zd,enth, linestyle='dashed', color='red', label='polytrope')
        plt.legend()
        plt.subplot(1,6,2)
        plt.gca().set_title(r'$\theta=\log(h)$')
        plt.xlabel('z')
        plt.plot(zd,np.log(h0['g']), label=r'$\log(h)$')
        plt.legend()

        plt.subplot(1,6,3)
        plt.gca().set_title(r'$\rho$')
        plt.xlabel('z')
        plt.plot(zd,np.exp(Υ0['g']), label=r'$\rho$')
        plt.plot(zd,dens, linestyle='dashed', color='blue', label='polytrope')
        plt.legend()
        plt.subplot(1,6,4)
        plt.gca().set_title(r'$Y=\log(\rho)$')
        plt.xlabel('z')
        plt.plot(zd,Υ0['g'], label=r'$\log(\rho)$')
        plt.legend()

        plt.subplot(1,6,5)
        plt.gca().set_title(r'$s$')
        plt.xlabel('z')
        plt.plot(zd,s0['g'], color='xkcd:brick red', label=r'$s$')
        plt.legend()

        plt.subplot(1,6,6)
        plt.gca().set_title('kappa')
        plt.xlabel('z')
        plt.plot(zd,κ0['g'], label='kappa')
        plt.legend()

        plt.savefig('kramers_solve_bc_jump{}_a{}_b{:.3g}_n{:.3g}.pdf'.format(bc_jump,aa,bb,n),bbox_inches='tight')

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

    #Free parameters (exponents) characterizing Kramers-like opacity (see. Barekat & Brandenburg 2014)
    aa = float(args['--aa'])
    bb = float(args['--bb'])

    bc_jump = float(args['--bc_jump'])

    γ  = float(Fraction(args['--gamma']))
    m_ad = 1/(γ-1)

    n_h = float(args['--n_h'])

    verbose = args['--verbose']

    structure = kramers_opacity_polytrope(nz, γ, n_h, aa, bb, bc_jump, verbose=verbose)
    for key in structure:
        print(structure[key], structure[key]['g'])
