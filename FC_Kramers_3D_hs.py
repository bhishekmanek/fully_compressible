"""
Dedalus script for 3D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_Kramers_3D_pert_hs.py [options]

Options:
    --Re=<Re>                            Reynolds number [default: 1000]
    --Pr=<Pr>                            Prandtl number [default: 1]
    --Ma2=<Ma2>                          Square of Mach number [default: 1]
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]
    --aa=<aa>                            Free parameter a for Kramer's opacity [default: 1]
    --bb=<bb>                            Free parameter b for Kramer's opacity [default: -3.5]
    --bc_jump=<bc_jump>                  Enthalpy top boundary condition jump [default: -0.05]

    --no_slip                            Use no-slip boundary conditions
    --safety=<safety>                    CFL safety factor
    --SBDF2                              Use SBDF2
    --max_dt=<max_dt>                    Largest timestep; also sets initial dt [default: 1]

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=aspect*nz
    --ny=<ny>                            Horizontal y; if not set, ny=aspect*nz 

    --run_time=<run_time>                Run time, in houru
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --data_dt=<data_dt>                  Time interval between two data dumps [default: 10]

    --restart=<restart>                  Merged chechpoint file to restart from.

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --label=<label>                      Additional label for run output directory

    --mesh=<mesh>                        Parallel decomposition mesh
"""

from mpi4py import MPI
import numpy as np
import sys
import os

# docopt reads the args
from docopt import docopt
args = docopt(__doc__)
from fractions import Fraction

ncc_cutoff = float(args['--ncc_cutoff'])

#Resolution
nz = int(args['--nz'])
nx = int(nz*float(args['--aspect']))
ny = int(nz*float(args['--aspect']))

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

# Define all the parameters. aa, bb, bc_jump are used for solving the NLBVP and getting the background stratification.
# mu is prescribed
γ  = float(Fraction(args['--gamma']))
Re = float(args['--Re'])
Pr = float(args['--Pr'])
R_inv = mu = 1./Re #dynamic shear viscosity
Pr_inv = 1./Pr
aa = float(args['--aa'])
bb = float(args['--bb'])
bc_jump = float(args['--bc_jump'])
n_poly = (3.0-bb)/(aa+1.0) #Polytropic index from the Kramers free parameters

m_ad = 1/(γ-1)

cP = γ/(γ-1)
Ma2 = float(args['--Ma2'])

no_slip = args['--no_slip']

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

import logging
logger = logging.getLogger(__name__)

# Mesh decomposition
mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

# Create data directory with appropriate naming convention
data_dir = sys.argv[0].split('.py')[0]
if no_slip:
    data_dir += '_NS'
data_dir += "_nh{}_R{}_mu{}".format(args['--n_h'], args['--Re'], args['--mu'])
data_dir += "_aa{}_bb{}_bc{}".format(args['--aa'], args['--bb'], args['--bc_jump'])
data_dir += "_a{}".format(args['--aspect'])
data_dir += "_nz{:d}_nx{:d}".format(nz,nx)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

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

logger.info("Ma2 = {:.3g}, Re = {:.3g}, R_inv = {:.3g}, Pr = {:.3g}, γ = {:.3g}".format(Ma2, Re, R_inv, Pr, γ))

logger.info(args)
logger.info("saving data in: {}".format(data_dir))

h_slope = -1/(1+n_poly) # Changing m_ad here to n
grad_φ = (γ-1)/γ

n_h = float(args['--n_h'])
Lz = -1/h_slope*(1-np.exp(-n_h))
Ly = float(args['--aspect'])*Lz
Lx = float(args['--aspect'])*Lz

vol = Lx*Ly*Lz

dealias = 3/2
c = de.CartesianCoordinates('x', 'y', 'z')
d = de.Distributor(c, mesh=mesh, dtype=np.float64) #Distributor directs parallelization and distribution of fields defined in the coordinate system "c".
xb = de.RealFourier(c.coords[0], size=nx, bounds=(0, Lx), dealias=dealias) # Define xb on the real Fourier sine/cosine basis
yb = de.RealFourier(c.coords[1], size=ny, bounds=(0, Ly), dealias=dealias)
zb = de.ChebyshevT(c.coords[2], size=nz, bounds=(0, Lz), dealias=dealias) # Define zb basis as a Chebyshev polynomial of the first kind.

b = (xb, yb, zb) # This is the basis on which we will define all fields. 
ba_p = (xb, yb)
x = xb.local_grid(1)
y = yb.local_grid(1)
z = zb.local_grid(1)

# Defining the fields on the bases b. log(h), log(rho), s, and u. 
# Fields
h1 = d.Field(name='h1', bases=b)
Υ1 = d.Field(name='Υ1', bases=b)
s1 = d.Field(name='s1', bases=b)
u = d.VectorField(c, name='u', bases=b)

# Taus
zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)
τ_s1 = d.Field(name='τ_s1', bases=ba_p)
τ_s2 = d.Field(name='τ_s2', bases=ba_p)
τ_u1 = d.VectorField(c, name='τ_u1', bases=ba_p)
τ_u2 = d.VectorField(c, name='τ_u2', bases=ba_p)

# Parameters and operators
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, c)
grad = lambda A: de.Gradient(A, c)
cross = lambda A, B: de.CrossProduct(A, B)
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)
curl = lambda A: de.Curl(A)
dt = lambda A: de.TimeDerivative(A)

integ = lambda A: de.Integrate(de.Integrate(de.Integrate(A, 'x'), 'y'), 'z')
avg = lambda A: integ(A)/(Lx*Ly*Lz)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)
xy_avg = lambda A: de.Integrate(de.Integrate(A, 'x'), 'y')/(Lx*Ly)

from dedalus.core.operators import Skew
skew = lambda A: Skew(A)

ex, ey, ez = c.unit_vector_fields(d)

# stress-free bcs
e = grad(u) + trans(grad(u))
ω = curl(u)

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
Phi = 0.5*trace(e@e) - 1/3*(trace_e*trace_e)

# Get the structure from an NLBVP solve "structure_kramers"
from structure_kramers import kramers_opacity_polytrope

structure = kramers_opacity_polytrope(nz, γ, n_h, aa, bb, bc_jump, comm=MPI.COMM_SELF)
polytrope = kramers_opacity_polytrope(nz, γ, n_h, aa, bb, 0,  comm=MPI.COMM_SELF)

θ0 = d.Field(name='θ0', bases=zb)
Υ0 = d.Field(name='Υ0', bases=zb)
s0 = d.Field(name='s0', bases=zb)
lnκ0 = d.Field(name='lnκ', bases=zb)

for q in structure:
    structure[q].require_coeff_space()

if s0['c'].size > 0:
    s0['c'][0,:] = structure['s']['c']
    θ0['c'][0,:] = structure['θ']['c']
    Υ0['c'][0,:] = structure['Υ']['c']
    lnκ0['c'][0,:] = structure['lnκ']['c']

# fixed in time diffusion coeff, shaped like initial thermal eq kappa
κ0 = np.exp(lnκ0).evaluate()
κ0.name='κ0'

# Calculting rho and other quantities
ρ0 = np.exp(Υ0).evaluate()
ρ0.name = 'ρ0'
ρ0_inv = np.exp(-Υ0).evaluate()
ρ0_inv.name = '1/ρ0'
h0 = np.exp(θ0).evaluate()
h0.name = 'h0'
grad_h0 = grad(h0).evaluate()
grad_θ0 = grad(θ0).evaluate()
grad_Υ0 = grad(Υ0).evaluate()
grad_s0 = grad(s0).evaluate()

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

if rank ==0:
    logger.info("Δθ = {:.2g} ({:.2g} to {:.2g})".format(θ_bot[0][0][0]-θ_top[0][0][0],θ_bot[0][0][0],θ_top[0][0][0]))
    logger.info("ΔΥ = {:.2g} ({:.2g} to {:.2g})".format(Υ_bot[0][0][0]-Υ_top[0][0][0],Υ_bot[0][0][0],Υ_top[0][0][0]))

# Calculate the NCCs in the equations
verbose = False
if verbose:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2)

logger.info("NCC expansions:")
for ncc in [h0, ρ0, ρ0*grad(h0), ρ0*grad(s0), ρ0*h0,
            ρ0*grad(θ0), h0*grad(Υ0), grad(Υ0),
            R_inv*Pr_inv*κ0, R_inv*Pr_inv*κ0*grad(lnκ0), R_inv*Pr_inv*κ0*grad(θ0)]:    
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
if verbose and rank==0:
    ax[1].set_xlabel('Tz')
    ax[1].set_yscale('log')
    ax[1].legend(loc='center right')
    fig.savefig('ncc_structure.pdf')

τ_u = lift(τ_u1,-1) + lift(τ_u2,-2)
τ_s = lift(τ_s1,-1) + lift(τ_s2,-2)
# Υ = ln(ρ), θ = ln(h)
vars = [u, Υ1, h1, s1]
taus = [τ_u1, τ_u2, τ_s1, τ_s2]

problem = de.IVP(vars+taus)
problem.add_equation((ρ0*(dt(u)
                      + 1/Ma2*grad(h1)
                      - 1/Ma2*h0*grad(s1)
                      - 1/Ma2*h1*grad(s0))
                      - R_inv*viscous_terms
                      + τ_u,
                      - ρ0_g*u@grad(u)
                      + 1/Ma2*ρ0_g*h1*grad(s1)
                      ))
problem.add_equation((h0*(dt(Υ1) + div(u) + u@grad_Υ0) + Re*lift(τ_u2,-1)@ez,
                      -h0_g*u@grad(Υ1) ))
problem.add_equation((h0*((γ-1)*Υ1 + γ*s1)-h1, h0_g*np.log(h1*h0_inv_g+1)-h1)) #EOS, s_c/cP = scrS
problem.add_equation((h0*ρ0*(dt(s1)
                      + u@grad(s0))
                      # small cheat, h0 + h1 -> h0, ρ0 + ρ1 -> ρ0 in denominator
                      - R_inv*Pr_inv*κ0*lap(h1)
                      - R_inv*Pr_inv*grad(κ0)@grad(h1)
                      + τ_s,
                      - ρ0_h0_g*u@grad(s1)
                      + R_inv*Ma2*Phi ))

if no_slip:
    problem.add_equation((u(z=0), 0))
    problem.add_equation((u(z=Lz), 0))
else:
    problem.add_equation((ez@u(z=0), 0))
    problem.add_equation((ez@(ex@e(z=0)), 0))
    problem.add_equation((ez@u(z=Lz), 0))
    problem.add_equation((ez@(ex@e(z=Lz)), 0))
problem.add_equation((ez@grad(h1)(z=0), 0))
problem.add_equation((s1(z=Lz), 0))

logger.info("Problem built")

# initial conditions
amp = 1e-4
zb, zt = zb.bounds
noise = d.Field(name='noise', bases=b)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.25)
noise['g'] *= np.cos(np.pi/2*z/Lz)

# isobaric entropy perturbations, to avoid launching acoustic modes
s1['g'] += noise['g']
Υ1['g'] += -γ/(γ-1)*noise['g']
h1['g'] += 0.0

if args['--SBDF2']:
    ts = de.SBDF2
    cfl_safety_factor = 0.1
else:
    ts = de.RK443
    cfl_safety_factor = 0.4
if args['--safety']:
    cfl_safety_factor = float(args['--safety'])

solver = problem.build_solver(ts)
solver.stop_iteration = run_time_iter
solver.stop_wall_time = run_time

# Check whether to restart or append the simulation
if not args['--restart']:
    mode = 'overwrite'
    Δt = max_Δt = float(args['--max_dt'])
else:
    write, dt = solver.load_state(args['--restart'], -1)
    Δt = dt
    max_Δt = float(args['--max_dt'])
    mode = 'append'

cfl = flow_tools.CFL(solver, Δt, safety=cfl_safety_factor, cadence=1, threshold=0.1,
                     max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

ρ = ρ0*np.exp(Υ1)
h = h0 + h1
s = s0 + s1
θ1 = np.log(h1/h0+1)
θ = θ1+θ0
Υ = Υ1+Υ0

ρ_fluc = ρ-xy_avg(ρ)
h_fluc = h-xy_avg(h)
s_fluc = s-xy_avg(s)

KE = 0.5*ρ*u@u
IE = 1/Ma2*ρ*h
PE = -1/Ma2*ρ*h*(s+s0)
Re = np.sqrt(u@u)*ρ0/mu # dissipation term chosen to only feel ρ0
N2 = (grad_φ*ez)@grad(s)
Ma_ad2 = Ma2*cP*u@u/(γ*h)

viscous_diffusion = u@e - 2/3*u@grad(u)

# Checkpoint save - wall_dt is in seconds
checkpoint = solver.evaluator.add_file_handler(data_dir+'/checkpoints', wall_dt = 28200, max_writes = 1)#, virtual_file=True, mode=mode)
checkpoint.add_tasks(solver.state)

vol_dt = 5.0
snap_dt = 2.0
trace_dt = 2.0
average_dt = 1.0

# Adding file handlers for writing data
# Instead of sim_dt, it is possible to use wall_dt and iter too. 
#
# 3D Volume data
vol_output = solver.evaluator.add_file_handler(data_dir+'/cube', sim_dt=vol_dt, max_writes=10, mode=mode)
vol_output.add_task(s_fluc, name='s_fluc')
vol_output.add_task(u@ez, name='uz')
vol_output.add_task(u@ey, name='uy')
vol_output.add_task(u@ex, name='ux')
vol_output.add_task(ω, name='vorticity')
vol_output.add_task(ω@ω, name='enstrophy')

# 2D Slice data
snap = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=snap_dt, max_writes=10, mode=mode)
snap.add_task(s_fluc(x=0), name='s_fluc yz side')
snap.add_task(s_fluc(y=0), name='s_fluc xz side')
snap.add_task(s_fluc(z=Lz/2), name='s_fluc xy midplane')
snap.add_task(s_fluc(y=Ly/2), name='s_fluc xz midplane')
snap.add_task(s_fluc(x=Lx/2), name='s_fluc yz midplane')
snap.add_task(s_fluc(z=Lz*3/4), name='s_fluc 0.75 z')
snap.add_task(s_fluc(z=0.99*Lz), name='s_fluc top')
snap.add_task((ω@ω)(x=0), name='enstrophy yz side')
snap.add_task((ω@ω)(y=0), name='enstrophy xz side')
snap.add_task((ω@ω)(z=Lz/2), name='enstrophy xy midplane')
snap.add_task((ω@ω)(y=Ly/2), name='enstrophy xz midplane')
snap.add_task((ω@ω)(x=Lx/2), name='enstrophy yz midpalne')
snap.add_task((ω@ω)(z=Lz*3/4), name='enstrophy 0.75 z')
snap.add_task((ω@ω)(z=0.99*Lz), name='enstrophy top')

# 1D average data
averages = solver.evaluator.add_file_handler(data_dir+'/averages', sim_dt=average_dt, max_writes=10, mode=mode)
averages.add_task(xy_avg(-R_inv*Pr_inv/Ma2/cP*κ0*grad(h-h0)@ez), name='F_κ_1(z)')
averages.add_task(xy_avg(-R_inv*Pr_inv/Ma2/cP*κ0*grad(h)@ez), name='F_κ(z)')
averages.add_task(xy_avg(0.5*ρ*u@ez*u@u), name='F_KE(z)')
averages.add_task(xy_avg(-R_inv*(viscous_diffusion@ez)),name='F_viscous(z)')
averages.add_task(xy_avg(u@ez*ρ*h/Ma2), name='F_h(z)')
averages.add_task(xy_avg(-u@ez*ρ*h*s/Ma2), name='F_PE(z)')
averages.add_task(xy_avg(u@ez*ρ*grad_φ/Ma2), name='F_g(z)')
averages.add_task(xy_avg(u@ez), name='uz(z)')
averages.add_task(xy_avg(N2), name='N2(z)')
#
averages.add_task(xy_avg(s1), name='s1(z)')
averages.add_task(xy_avg(h1), name='h1(z)')
averages.add_task(xy_avg(θ1), name='θ1(z)')
averages.add_task(xy_avg(Υ1), name='Υ1(z)')
averages.add_task(xy_avg(s_fluc), name='s_fluc(z)')
averages.add_task(xy_avg(h_fluc), name='h_fluc(z)')
averages.add_task(xy_avg(ρ_fluc), name='ρ_fluc(z)')
averages.add_task(xy_avg(s), name='s(z)')
averages.add_task(xy_avg(h), name='h(z)')
averages.add_task(xy_avg(θ), name='θ(z)')
averages.add_task(xy_avg(Υ), name='Υ(z)')
averages.add_task(np.sqrt(xy_avg(τ_u@τ_u)), name='τ_u')
averages.add_task(np.sqrt(xy_avg(τ_s**2)), name='τ_s')

# Scalars - as a function of time
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=trace_dt, max_writes=None, mode=mode)
traces.add_task(avg(KE), name='KE')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(IE), name='IE')
traces.add_task(avg(Re), name='Re')
traces.add_task(avg(N2), name='BV_freq')
traces.add_task(avg(ω@ω), name='enstrophy')
traces.add_task(np.sqrt(avg(Ma_ad2)), name='Ma_ad')
traces.add_task(xy_avg(np.sqrt(τ_u@τ_u)), name='τ_u')
traces.add_task(xy_avg(np.sqrt(τ_s*τ_s)), name='τ_s')

report_cadence = 100

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(IE, name='IE')
flow.add_property(np.sqrt(avg(Ma_ad2)), name='Ma_ad')
flow.add_property(np.sqrt(τ_s**2), name='|τ_s|')
flow.add_property(np.sqrt(τ_u@τ_u), name='|τ_u|')

KE_avg = 0
good_solution = True

while solver.proceed and good_solution:
    # advance
    solver.step(Δt)
    if solver.iteration % report_cadence == 0:
        KE_avg = flow.volume_integral('KE')/vol
        IE_avg = flow.volume_integral('IE')/vol
        Ma_ad_avg = flow.volume_integral('Ma_ad')/vol # Check this term
        Re_avg = flow.volume_integral('Re')/vol
        Re_max = flow.max('Re')
        τ_max = np.max([flow.max('|τ_s|'), flow.max('|τ_u|')])
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
    logger.info("τu = {}".format(τ_max))

solver.log_stats()
logger.debug("mode-stages/DOF = {}".format(solver.total_modes/(nx*nz)))
