"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_Kramers.py [options]

Options:
    --R=<R>                              Reynolds number [default: 1e2]
    --Ma2=<Ma2>                          Square of Mach number [default: 1]
    --mu=<mu>                            Dynamic viscosity [default: 0.01]
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]
    --aa=<aa>                            Free parameter a for Kramer's opacity [default: 1]
    --bb=<bb>                            Free parameter b for Kramer's opacity [default: -2]
    --bc_jump=<bc_jump>                  Enthalpy top boundary condition jump [default: -0.05]

    --no_slip	                         Use no-slip boundary conditions
    --safety=<safety>                    CFL safety factor
    --SBDF2                              Use SBDF2
    --max_dt=<max_dt>                    Largest timestep; also sets initial dt [default: 1]

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=aspect*nz

    --run_time=<run_time>                Run time, in houru
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --restart=<restart>                  Merged chechpoint file to restart from.

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --label=<label>                      Additional label for run output directory
"""

from mpi4py import MPI
import numpy as np
import sys
import os

# docopt reads the args
from docopt import docopt
args = docopt(__doc__)
from fractions import Fraction

#Bhishek: What is this used for?
# This is an amplitude cutoff for the non-constant coefficients. This has a large performance impact when the polynomials are not low-degrees.
ncc_cutoff = float(args['--ncc_cutoff'])

#Resolution
nz = int(args['--nz'])
nx = args['--nx']
if nx is not None:
    nx = int(nx)
else:
    nx = int(nz*float(args['--aspect']))

# Bhishek
# Get the buoyancy run time and the no. of iterations to run for
run_time_buoy = args['--run_time_buoy']
if run_time_buoy != None:
    run_time_buoy = float(run_time_buoy)

run_time_iter = args['--run_time_iter']
if run_time_iter != None:
    run_time_iter = int(float(run_time_iter))
else:
    run_time_iter = np.inf

run_time = args['--run_time']
if args['--run_time']:
    run_time = float(args['--run_time'])
else:
    run_time = np.inf

# Bhishek
# Define all the parameters. aa, bb, bc_jump are used for solving the NLBVP and getting the background stratification.
# mu is prescribed (NEEDS TO BE CHECKED)
R = float(args['--R'])
R_inv = scrR = 1/R
γ  = float(Fraction(args['--gamma']))
mu = float(args['--mu'])
aa = float(args['--aa'])
bb = float(args['--bb'])
bc_jump = float(args['--bc_jump'])
n_poly = (3-bb)/(aa+1) #Polytropic index from the Kramers free parameters

m_ad = 1/(γ-1)

cP = γ/(γ-1)
Ma2 = float(args['--Ma2'])
scrM = 1/Ma2
s_c_over_c_P = scrS = 1 # s_c/c_P = 1

no_slip = args['--no_slip']

#Bhishek
#Define the directory name in which to store the run data
data_dir = sys.argv[0].split('.py')[0]
if no_slip:
    data_dir += '_NS'
data_dir += "_nh{}_R{}_Ma2_{}_mu{}_bc_jump{}".format(args['--n_h'], args['--R'], args['--Ma2'], args['--mu'], args['--bc_jump'])
data_dir += "_a{}_npoly{}".format(args['--aspect'], n_poly)
data_dir += "_nz{:d}_nx{:d}".format(nz,nx)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

#Bhishek
#Deal with the log file. NOT REALLY SURE WHAT THIS MEANS.
import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'

from dedalus.tools.parallel import Sync
with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

import dedalus.public as de
from dedalus.extras import flow_tools
rank = MPI.COMM_WORLD.rank

logger.info("Ma2 = {:.3g}, R = {:.3g}, R_inv = {:.3g}, mu = {:.3g}, γ = {:.3g}".format(Ma2, R, R_inv, mu, γ))

logger.info(args)
logger.info("saving data in: {}".format(data_dir))

#Bhishek
# Check if you need h_bot over here. NOT REALLY
# this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)

#h_bot = 1 #New change

# generally, h_slope = -1/(1+m)
# start in an adibatic state, heat from there

#Bhishek
#HERE H_SLOPE NEEDS TO CHANGE. IT SHOULD BE -1/(1+N) WHERE N IS THE POLYTROPIC INDEX DEFINED BY AA AND BB FREE PARAMETERS.
h_slope = -1/(1+n_poly) # Changing m_ad here to n
grad_φ = (γ-1)/γ

#Bhishek
#CHECK THE DERIVATION FOR THIS. HOW IS LZ DEFINED IN TERMS OF ENTHALPY SCALE HEIGHT
n_h = float(args['--n_h'])
Lz = -1/h_slope*(1-np.exp(-n_h))
Lx = float(args['--aspect'])*Lz

#Bhishek
#What does dealias do?
dealias = 2
c = de.CartesianCoordinates('x', 'z')
d = de.Distributor(c, dtype=np.float64) #Distributor directs parallelization and distribution of fields defined in the coordinate system "c".
xb = de.RealFourier(c.coords[0], size=nx, bounds=(0, Lx), dealias=dealias) # Define xb on the real Fourier sine/cosine basis
zb = de.ChebyshevT(c.coords[1], size=nz, bounds=(0, Lz), dealias=dealias) # Define zb basis as a Chebyshev polynomial of the first kind.

b = (xb, zb) # This is the basis on which we will define all fields. 
x = xb.local_grid(1) #Bhishek - Not sure what this does. Nothing on documentation.
z = zb.local_grid(1)

#Bhishek
# Defining the fields on the bases b. log(h), log(rho), s, and u. 
# Fields
θ = d.Field(name='θ', bases=b)
Υ = d.Field(name='Υ', bases=b)
s = d.Field(name='s', bases=b)
u = d.VectorField(c, name='u', bases=b)

# Bhishek
# What is Lambda?
# Taus
zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)
τ_s1 = d.Field(name='τ_s1', bases=xb)
τ_s2 = d.Field(name='τ_s2', bases=xb)
τ_u1 = d.VectorField(c, name='τ_u1', bases=(xb,))
τ_u2 = d.VectorField(c, name='τ_u2', bases=(xb,))

# Parameters and operators
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, c)
grad = lambda A: de.Gradient(A, c)
cross = lambda A, B: de.CrossProduct(A, B)
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)
dt = lambda A: de.TimeDerivative(A)

integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

from dedalus.core.operators import Skew
skew = lambda A: Skew(A)

ex, ez = c.unit_vector_fields(d)

# stress-free bcs
e = grad(u) + trans(grad(u))
e.store_last = True

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
trace_e.store_last = True
Phi = 0.5*trace(e@e) - 1/3*(trace_e*trace_e)

############### Trying to bring structure in a parallel run #########################################################
from structure_kramers import kramers_opacity_polytrope
structure = kramers_opacity_polytrope(nz, γ, n_h, aa, bb, bc_jump, verbose=True, comm=MPI.COMM_SELF)

h0 = d.Field(name='h0', bases=zb)
θ0 = d.Field(name='θ0', bases=zb)
Υ0 = d.Field(name='Υ0', bases=zb)
s0 = d.Field(name='s0', bases=zb)
κ0 = d.Field(name='κ0', bases=zb)

if h0['g'].size > 0 :
   for i, z_i in enumerate(z[0,:]):
        h0['g'][:,i] = structure['h'](z=z_i).evaluate()['g'].real
        s0['g'][:,i] = structure['s'](z=z_i).evaluate()['g'].real
        θ0['g'][:,i] = structure['θ'](z=z_i).evaluate()['g'].real
        Υ0['g'][:,i] = structure['Υ'](z=z_i).evaluate()['g'].real
        κ0['g'][:,i] = structure['κ'](z=z_i).evaluate()['g'].real

# Calculting rho and other quantities. Mostly playing with this because of the log formulation.
ρ0 = np.exp(Υ0).evaluate()
ρ0.name = 'ρ0'
ρ0_inv = np.exp(-Υ0).evaluate()
ρ0_inv.name = '1/ρ0'
grad_h0 = grad(h0).evaluate()
grad_θ0 = grad(θ0).evaluate()
grad_Υ0 = grad(Υ0).evaluate()
grad_s0 = grad(s0).evaluate()

ln_κ0 = np.log(κ0).evaluate()
grad_ln_κ0 = grad(ln_κ0).evaluate()

h0_g = de.Grid(h0).evaluate()
h0_inv_g = de.Grid(1/h0).evaluate()
grad_h0_g = de.Grid(grad(h0)).evaluate()
ρ0_g = de.Grid(ρ0).evaluate()

ρ0_grad_h0_g = de.Grid(ρ0*grad(h0)).evaluate()
ρ0_h0_g = de.Grid(ρ0*h0).evaluate()

Υ_bot = Υ0(z=0).evaluate()['g']
Υ_top = Υ0(z=Lz).evaluate()['g']

θ_bot = θ0(z=0).evaluate()['g']
θ_top = θ0(z=Lz).evaluate()['g']

###### Debug ##################################
#print("Priting log density at bottom and top")
#print(Υ_bot,Υ_top)
#print("Priting log enthalpy at bottom and top")
#print(θ_bot,θ_top)
###### Debug ##################################

if rank ==0:
    logger.info("Δθ = {:.2g} ({:.2g} to {:.2g})".format(θ_bot[0][0]-θ_top[0][0],θ_bot[0][0],θ_top[0][0]))
    logger.info("ΔΥ = {:.2g} ({:.2g} to {:.2g})".format(Υ_bot[0][0]-Υ_top[0][0],Υ_bot[0][0],Υ_top[0][0]))

verbose = False
if verbose:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2)

# Bhishek - What is this doing exactly?
# Seems to be putting a threshold (defined by ncc_cutoff) on all the NCC expansions.
logger.info("NCC expansions:")
for ncc in [h0, ρ0, ρ0*grad(h0), ρ0*h0, ρ0*grad(θ0), h0*grad(Υ0)]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))
    if verbose:
        ncc = ncc.evaluate()
        ncc.change_scales(1)
        if ncc['g'].ndim == 3:
            i = (1, 0, slice(None))
        else:
            i = (0, slice(None))
        ax[0].plot(z[0,:], ncc['g'][i])
        ax[1].plot(z[0,:], np.abs(ncc['g'][i]), label=ncc.name)
if verbose:
    ax[1].set_yscale('log')
    ax[1].legend()
    fig.savefig('structure.pdf')

# Defining the Prandtl number here.
Pr = mu*cP/κ0
Pr_inv = 1/Pr

# Υ = ln(ρ), θ = ln(h)
problem = de.IVP([u, Υ, θ, s, τ_u1, τ_u2, τ_s1, τ_s2])
problem.add_equation((ρ0*(dt(u) + 1/Ma2*(h0*grad(θ) + grad_h0*θ)
                      - 1/Ma2*s_c_over_c_P*h0*grad(s)
                      - 1/Ma2*h0*grad_s0*θ)
                      - R_inv*viscous_terms
                      + lift(τ_u1,-1) + lift(τ_u2,-2),
                      - ρ0_g*u@grad(u)
                      - 1/Ma2*ρ0_grad_h0_g*(np.expm1(θ)-θ)
                      - 1/Ma2*ρ0_h0_g*np.expm1(θ)*grad(θ)
                      + 1/Ma2*scrS*ρ0_h0_g*np.expm1(θ)*grad(s)
                      + 1/Ma2*scrS*ρ0_h0_g*grad_s0*(np.expm1(θ)-θ)
                      ))
problem.add_equation((h0*(dt(Υ) + div(u) + u@grad_Υ0) + R*lift(τ_u2,-1)@ez,
                      -h0_g*u@grad(Υ) ))
problem.add_equation((θ - (γ-1)*Υ - s_c_over_c_P*γ*s, 0)) #EOS, s_c/cP = scrS
problem.add_equation((ρ0*s_c_over_c_P*dt(s)
                      - R_inv*Pr_inv*(lap(θ)+2*grad_θ0@grad(θ)+grad_ln_κ0@grad(θ))
#                      + ρ0_g*s_c_over_c_P*u@grad(s0)
                      + lift(τ_s1,-1) + lift(τ_s2,-2),
                      - ρ0_g*s_c_over_c_P*u@grad(s0)
                      - ρ0_g*s_c_over_c_P*u@grad(s)
                      + R_inv*Pr_inv*(grad(θ)@grad(θ))#+grad_ln_κ0@grad(θ0))
                      + R_inv*Ma2*h0_inv_g*Phi )) # + R_inv*Ma2*0.5*h0_inv_g*Phi

if no_slip:
    problem.add_equation((u(z=0), 0))
    problem.add_equation((u(z=Lz), 0))
else:
    problem.add_equation((ez@u(z=0), 0))
    problem.add_equation((ez@(ex@e(z=0)), 0))
    problem.add_equation((ez@u(z=Lz), 0))
    problem.add_equation((ez@(ex@e(z=Lz)), 0))
#problem.add_equation((θ(z=0), 0)) #Ideally it should be np.log(h_bot)
#problem.add_equation((θ(z=Lz), np.log(np.exp(-n_h)+bc_jump)))
#problem.add_equation((s(z=0), 0))
problem.add_equation((s(z=Lz), 0))
#Old BC's
problem.add_equation((ez@grad(θ)(z=0), 0))
#problem.add_equation((θ(z=Lz), 0))
#problem.add_equation((θ(z=0), 0))
logger.info("Problem built")

# initial conditions
amp = 1e-4

zb, zt = zb.bounds
noise = d.Field(name='noise', bases=b)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.25)

s['g'] = noise['g']*np.cos(np.pi/2*z/Lz)
# pressure balanced ICs
Υ['g'] = -scrS*γ/(γ-1)*s['g']
θ['g'] = scrS*γ*s['g'] + (γ-1)*Υ['g'] # this should evaluate to zero

if args['--SBDF2']:
    ts = de.SBDF2
    cfl_safety_factor = 0.2
else:
    ts = de.RK443
    cfl_safety_factor = 0.4
if args['--safety']:
    cfl_safety_factor = float(args['--safety'])

solver = problem.build_solver(ts)
solver.stop_iteration = run_time_iter
#solver.stop_wall_time = run_time

# Check whether to restart or append the simulation
#if not args['--restart']:
#    mode = 'overwrite'
#else:
#    write, dt = solver.load_state(args['--restart'])
#    mode = 'append'

if not args['--restart']:
    mode = 'overwrite'
    Δt = max_Δt = float(args['--max_dt'])
else:
    write, dt = solver.load_state(args['--restart'], -1)
    Δt = dt
    max_Δt = float(args['--max_dt'])
    mode = 'append'


#Δt = max_Δt = float(args['--max_dt'])
cfl = flow_tools.CFL(solver, Δt, safety=cfl_safety_factor, cadence=1, threshold=0.1,
                     max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

#s0 = d.Field(name='s0')
#s0['g'] = 0

ρ = ρ0*np.exp(Υ).evaluate()
h = h0*np.exp(θ).evaluate()
KE = 0.5*ρ*u@u
IE = 1/Ma2*ρ*h
PE = -1/Ma2*ρ*h*(s+s0)
Re = (ρ*R)*np.sqrt(u@u)
ω = -div(skew(u))
N2 = -((grad_φ*ez)@grad(s+s0))/cP
KE.store_last = True
PE.store_last = True
IE.store_last = True
Re.store_last = True
ω.store_last = True
N2.store_last = True

# Checkpoint save - wall_dt is in seconds - wall_dt=6900 means that it dumps a checkpoint at 115 minutes - ideal for a 2hr run
checkpoint = solver.evaluator.add_file_handler(data_dir+'/checkpoints', wall_dt = 28500, max_writes = 1, virtual_file=True, mode=mode)
checkpoint.add_tasks(solver.state)

average_dt = 30.0


slice_output = solver.evaluator.add_file_handler(data_dir+'/slices', sim_dt=average_dt, max_writes=10, mode=mode)
slice_output.add_task(s-x_avg(s), name='srem')
#snap.add_task((s-xy_avg(s))(x=Lx/2), name='srem yz midplane')


# Adding file handlers for writing data
# You can add an arbitrary number of file handlers to save different sets of tasks at different cadences and to different files. (Dedalus webpage)

# Instead of sim_dt, it is possible to use wall_dt and iter too. 
# horizontal averages
#print("Prandtl number is", mu*cP*pow(κ0['g'],-1))
averages = solver.evaluator.add_file_handler(data_dir+'/averages', sim_dt=average_dt, max_writes=10, mode=mode)
averages.add_task(x_avg(s+s0), name='stot(z)')
averages.add_task(x_avg(s-s0), name='sfluc(z)')
#averages.add_task(x_avg(s0), name='s_ic(z)')
averages.add_task(x_avg(s), name='s(z)')
averages.add_task(x_avg(h), name='h(z)')
averages.add_task(x_avg(h-h0), name='hfluc(z)')
averages.add_task(x_avg(h+h0), name='htot(z)')
#averages.add_task(x_avg(h0), name='h_ic(z)')
averages.add_task(x_avg(-R_inv/Pr*grad(h-h0)@ez), name='F_κ(z)')
#averages.add_task(x_avg(R_inv), name='R_inv(z)')
#averages.add_task(x_avg(Pr), name='Pr(z)')
averages.add_task(x_avg(grad(h-h0)@ez), name='gradh_h0(z)')
averages.add_task(x_avg(grad(h)@ez), name='gradh(z)')
#averages.add_task(x_avg(grad(h0)@ez), name='gradh0(z)')
averages.add_task(x_avg(θ), name='θ(z)')
#averages.add_task(x_avg(θ0), name='θ0(z)')
averages.add_task(x_avg(θ+θ0), name='θ_tot(z)')
averages.add_task(x_avg(θ-θ0), name='θ_fluc(z)')
averages.add_task(x_avg(Υ), name='Υ(z)')
#averages.add_task(x_avg(Υ0), name='Υ0(z)')
averages.add_task(x_avg(Υ+Υ0), name='Υ_tot(z)')
averages.add_task(x_avg(Υ-Υ0), name='Υ_fluc(z)')
averages.add_task(κ0, name='κ0(z)')
averages.add_task(x_avg(ρ), name='ρ(z)')
#averages.add_task(x_avg(ρ0), name='ρ0(z)')
averages.add_task(x_avg(ρ-ρ0), name='ρ_fluc(z)')
averages.add_task(x_avg(0.5*ρ*u@ez*u@u), name='F_KE(z)')
averages.add_task(x_avg(u@ez*ρ*h/Ma2), name='F_h(z)')
averages.add_task(x_avg(-u@ez*ρ*h*s/Ma2), name='F_PE(z)')
averages.add_task(x_avg(u@ez*ρ*grad_φ/Ma2), name='F_g(z)')
averages.add_task(x_avg(u@ez), name='uz(z)')
#averages.add_task(x_avg(ω@ω), name='enstrophy(z)')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=average_dt, max_writes=10, mode=mode)
traces.add_task(avg(0.5*ρ*u@u), name='KE')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(IE), name='IE')
traces.add_task(avg(Re), name='Re')
traces.add_task(avg(N2), name='BV_freq')
traces.add_task(avg(ω**2), name='avg_enstrophy')

Ma_ad2 = Ma2*u@u*cP/(γ*h)

report_cadence = 5
good_solution = True

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(IE, name='IE')
flow.add_property(np.sqrt(avg(Ma_ad2)), name='Ma_ad')
flow.add_property(τ_u1, name='τ_u1')
flow.add_property(τ_u2, name='τ_u2')
flow.add_property(τ_s1, name='τ_s1')
flow.add_property(τ_s2, name='τ_s2')

KE_avg = 0
while solver.proceed and good_solution:
    # advance
    solver.step(Δt)
    if solver.iteration % report_cadence == 0:
        KE_avg = flow.grid_average('KE')
        IE_avg = flow.grid_average('IE')
        Ma_ad_avg = flow.grid_average('Ma_ad')
        Re_avg = flow.grid_average('Re')
        Re_max = flow.max('Re')
        τu1_max = flow.max('τ_u1')
        τu2_max = flow.max('τ_u2')
        τs1_max = flow.max('τ_s1')
        τs2_max = flow.max('τ_s2')
        τ_max = np.max([τu1_max,τu2_max,τs1_max,τs2_max])
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:5.1e}'.format(solver.iteration, solver.sim_time, Δt)
        log_string += ', KE: {:.2g}, Ma: {:.2g}, IE: {:.2g}, Re: {:.2g} ({:.2g})'.format(KE_avg, Ma_ad_avg, IE_avg, Re_avg, Re_max)
        log_string += ', τ: {:.2g}'.format(τ_max)
        logger.info(log_string)
    Δt = cfl.compute_timestep()
    good_solution = np.isfinite(Δt)*np.isfinite(KE_avg)

if not good_solution:
    logger.info("simulation terminated with good_solution = {}".format(good_solution))
    logger.info("Δt = {}".format(Δt))
    logger.info("KE = {}".format(KE_avg))
    logger.info("τu = {}".format((τu1_max,τu2_max,τs1_max,τs2_max)))

solver.log_stats()
logger.debug("mode-stages/DOF = {}".format(solver.total_modes/(nx*nz)))
