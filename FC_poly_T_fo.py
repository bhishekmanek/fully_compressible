"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options]

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number (not used) [default: 1e4]
    --mu=<mu>                            Viscosity [default: 0.0015]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0.5]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]

    --safety=<safety>                    CFL safety factor
    --SBDF2                              Use SBDF2
    --max_dt=<max_dt>                    Largest timestep; also sets initial dt [default: 1]

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --label=<label>                      Additional label for run output directory
"""

import numpy as np
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
from dedalus.tools.parallel import Sync

from docopt import docopt
args = docopt(__doc__)
from fractions import Fraction

import sys
import os
import pathlib
import h5py

import logging
logger = logging.getLogger(__name__)

dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

ncc_cutoff = float(args['--ncc_cutoff'])

#Resolution
nz = int(args['--nz'])
nx = args['--nx']
if nx is not None:
    nx = int(nx)
else:
    nx = int(nz*float(args['--aspect']))

run_time_buoy = args['--run_time_buoy']
if run_time_buoy != None:
    run_time_buoy = float(run_time_buoy)

run_time_iter = args['--run_time_iter']
if run_time_iter != None:
    run_time_iter = int(float(run_time_iter))
else:
    run_time_iter = np.inf

Ra = Rayleigh = float(args['--Rayleigh']),
Pr = Prandtl = float(args['--Prandtl'])
γ  = float(Fraction(args['--gamma']))

m_ad = 1/(γ-1)
if args['--m']:
    m = float(args['--m'])
    strat_label = 'm{}'.format(args['--m'])
else:
    m = m_ad - float(args['--epsilon'])
    strat_label = 'eps{}'.format(args['--epsilon'])
ε = m_ad - m

cP = γ/(γ-1)

data_dir = sys.argv[0].split('.py')[0]
data_dir += "_nh{}_μ{}_Pr{}".format(args['--n_h'], args['--mu'], args['--Prandtl'])
data_dir += "_{}_a{}".format(strat_label, args['--aspect'])
data_dir += "_nz{:d}_nx{:d}".format(nz,nx)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'

with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

import dedalus.public as de
from dedalus.extras import flow_tools

logger.info(args)
logger.info("saving data in: {}".format(data_dir))


# this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
h_bot = 1
h_slope = -1/(1+m)
grad_φ = (γ-1)/γ

n_h = float(args['--n_h'])
Lz = -1/h_slope*(1-np.exp(-n_h))
Lx = float(args['--aspect'])*Lz

dealias = 2
c = de.CartesianCoordinates('x', 'z')
d = de.Distributor(c, dtype=np.float64)
xb = de.RealFourier(c.coords[0], size=nx, bounds=(0, Lx), dealias=dealias)
zb = de.ChebyshevT(c.coords[1], size=nz, bounds=(0, Lz), dealias=dealias)
b = (xb, zb)
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
T = d.Field(name='T', bases=b)
Υ = d.Field(name='Υ', bases=b)
s = d.Field(name='s', bases=b)
u = d.VectorField(c, name='u', bases=b)

# Taus
zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift_basis = zb.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: de.LiftTau(A, lift_basis, n)
τs1 = d.Field(name='τs1', bases=xb)
τs2 = d.Field(name='τs2', bases=xb)
τu1 = d.VectorField(c, name='τu1', bases=(xb,))
τu2 = d.VectorField(c, name='τu2', bases=(xb,))

# Parameters and operators
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, c)
grad = lambda A: de.Gradient(A, c)
#curl = lambda A: de.operators.Curl(A)
dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)
dt = lambda A: de.TimeDerivative(A)

integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
#x_avg = lambda A: de.Integrate(A, 'x')/(Lx)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

from dedalus.core.operators import Skew
skew = lambda A: Skew(A)


ex = d.VectorField(c, name='ex', bases=zb)
ez = d.VectorField(c, name='ez', bases=zb)
ex['g'][0] = 1
ez['g'][1] = 1
ez2 = d.VectorField(c, name='ez2', bases=zb2)
ez2['g'][1] = 1

h0 = d.Field(name='h0', bases=zb)

ln_cP = np.log(cP)
h0['g'] = h_bot+h_slope*z
T0 = h0.evaluate() #(h0/cP).evaluate()
T0.name = 'T0'
θ0 = np.log(h0).evaluate()
θ0.name = 'θ0'
Υ0 = (m*(θ0)).evaluate() # normalize to zero at bottom
Υ0.name = 'Υ0'
s0 = (1/γ*θ0 - (γ-1)/γ*Υ0).evaluate()
s0.name = 's0'
ρ0_inv = np.exp(-Υ0).evaluate()
ρ0_inv.name = 'ρ0_inv'

grad_u = grad(u) + ez*lift(τu1,-1) # First-order reduction
grad_T = grad(T) + ez*lift(τs1,-1) # First-order reduction

# stress-free bcs
e = grad_u + trans(grad_u)
e.store_last = True

viscous_terms = div(e) - 2/3*grad(trace(grad_u))
trace_e = trace(e)
trace_e.store_last = True
Phi = 0.5*trace(dot(e, e)) - 1/3*(trace_e*trace_e)

Ma2 = 1 #ε
Pr = 1

μ = float(args['--mu'])
κ = μ*cP/Pr # Mihalas & Mihalas eq (28.3)
s_bot = s0(z=0).evaluate()['g']
s_top = s0(z=Lz).evaluate()['g']

delta_s = s_bot-s_top
delta_s_2 = ε*np.log(1+Lz)
logger.info('delta_s = {:}, {:}'.format(delta_s, delta_s_2))
g = m+1
pre = g*(delta_s)*Lz**3
Ra_bot = pre*(1/(μ*κ*cP)*np.exp(2*Υ0)(z=0)).evaluate()['g']
Ra_mid = pre*(1/(μ*κ*cP)*np.exp(2*Υ0)(z=Lz/2)).evaluate()['g']
Ra_top = pre*(1/(μ*κ*cP)*np.exp(2*Υ0)(z=Lz)).evaluate()['g']

Υ_bot = Υ0(z=0).evaluate()['g']
Υ_top = Υ0(z=Lz).evaluate()['g']

θ_bot = θ0(z=0).evaluate()['g']
θ_top = θ0(z=Lz).evaluate()['g']

T_bot = T0(z=0).evaluate()['g']
T_top = T0(z=Lz).evaluate()['g']


if rank ==0:
    logger.info("Ra(z=0)   = {:.2g}".format(Ra_bot[0][0]))
    logger.info("Ra(z={:.1f}) = {:.2g}".format(Lz/2, Ra_mid[0][0]))
    logger.info("Ra(z={:.1f}) = {:.2g}".format(Lz, Ra_top[0][0]))
    logger.info("Δs = {:.2g} ({:.2g} to {:.2g})".format(s_bot[0][0]-s_top[0][0],s_bot[0][0],s_top[0][0]))
    logger.info("Δθ = {:.2g} ({:.2g} to {:.2g})".format(θ_bot[0][0]-θ_top[0][0],θ_bot[0][0],θ_top[0][0]))
    logger.info("ΔT = {:.2g} ({:.2g} to {:.2g})".format(T_bot[0][0]-T_top[0][0],T_bot[0][0],T_top[0][0]))
    logger.info("ΔΥ = {:.2g} ({:.2g} to {:.2g})".format(Υ_bot[0][0]-Υ_top[0][0],Υ_bot[0][0],Υ_top[0][0]))
scale = d.Field(name='scale', bases=zb2)
scale.require_scales(dealias)
scale['g'] = T0['g']

h0_grad_s0_g = de.Grid(h0*grad(s0)).evaluate()
h0_grad_s0_g.name = 'h0_grad_s0_g'
h0_g = de.Grid(h0).evaluate()
h0_g.name = 'h0_g'


for ncc in [grad(Υ0), grad(T0), T0, np.exp(-Υ0), ρ0_inv]:
    logger.info('scaled {:} has  {:} terms'.format(ncc,(np.abs((scale*ncc).evaluate()['c'])>ncc_cutoff).sum()))

# Υ = ln(ρ), θ = ln(h)
problem = de.IVP([Υ, u, T, τu1, τu2, τs1, τs2])
problem.add_equation((scale*(dt(Υ) + trace(grad_u) + dot(u, grad(Υ0))),
                      scale*(-dot(u, grad(Υ))) ))
problem.add_equation((scale*(dt(u) + grad(T) \
                      + T*grad(Υ0) + T0*grad(Υ)
                      - μ*ρ0_inv*viscous_terms) \
                      + lift(τu2,-1),
                      scale*(-dot(u,grad(u)) - T*grad(Υ)) )) # need nonlinear density effects on viscous terms
problem.add_equation((scale*(dt(T) + dot(u,grad(T0)) + T0*(γ-1)*trace(grad_u) - κ*ρ0_inv*div(grad_T)) \
                      + lift(τs2,-1),
                      scale*(-dot(u,grad(T)) - T*(γ-1)*div(u)) )) # need VH and nonlinear density effects on diffusion
problem.add_equation((T(z=0), 0))
problem.add_equation((u(z=0), 0))
problem.add_equation((T(z=Lz), 0))
problem.add_equation((u(z=Lz), 0))
logger.info("Problem built")

# initial conditions
amp = 1e-4*Ma2

zb, zt = zb.bounds
noise = d.Field(name='noise', bases=b)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.25)

# s['g'] = noise['g']*np.sin(np.pi*z/Lz)
# Υ['g'] = -γ/(γ-1)*s['g']
# T['g'] = np.exp(γ*s['g']) + np.exp((γ-1)*Υ['g'])
T0.require_scales(1)
T['g'] = noise['g']*np.sin(np.pi*z/Lz)*T0['g']

if args['--SBDF2']:
    ts = de.SBDF2
    cfl_safety_factor = 0.3
else:
    ts = de.RK443
    cfl_safety_factor = 0.4
if args['--safety']:
    cfl_safety_factor = float(args['--safety'])

solver = problem.build_solver(ts)
solver.stop_iteration = run_time_iter

Δt = max_Δt = float(args['--max_dt'])
cfl = flow_tools.CFL(solver, Δt, safety=cfl_safety_factor, cadence=1, threshold=0.1,
                     max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

ρ = np.exp(Υ0+Υ).evaluate()
h = cP*(T+T0)
KE = 0.5*ρ*dot(u,u)
IE = cP*Ma2*h*(s+s0)
Re = (ρ/μ)*np.sqrt(dot(u,u))
ω = -div(skew(u))
KE.store_last = True
IE.store_last = True
Re.store_last = True
ω.store_last = True

slice_output = solver.evaluator.add_file_handler(data_dir+'/slices',sim_dt=0.25,max_writes=20)
slice_output.add_task(s+s0, name='s+s0')
slice_output.add_task(s, name='s')
slice_output.add_task(T, name='T')
slice_output.add_task(ω, name='vorticity')
slice_output.add_task(ω**2, name='enstrophy')
slice_output.add_task(x_avg(-κ*dot(grad(h),ez)/cP), name='F_κ')
slice_output.add_task(x_avg(0.5*ρ*dot(u,ez)*dot(u,u)), name='F_KE')
slice_output.add_task(x_avg(dot(u,ez)*h), name='F_h')

traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=0.1, max_writes=np.inf)
traces.add_task(avg(0.5*ρ*dot(u,u)), name='KE')
traces.add_task(avg(IE), name='IE')
traces.add_task(avg(Re), name='Re')
traces.add_task(avg(ω**2), name='enstrophy')
traces.add_task(x_avg(np.sqrt(dot(τu1,τu1))), name='τu1')
traces.add_task(x_avg(np.sqrt(dot(τu2,τu2))), name='τu2')
traces.add_task(x_avg(np.sqrt(τs1**2)), name='τs1')
traces.add_task(x_avg(np.sqrt(τs2**2)), name='τs2')

report_cadence = 10
good_solution = True

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(IE, name='IE')
flow.add_property(τu1, name='τu1')
flow.add_property(τu2, name='τu2')
flow.add_property(τs1, name='τs1')
flow.add_property(τs2, name='τs2')

KE_avg = 0
while solver.proceed and good_solution:
    # advance
    solver.step(Δt)
    if solver.iteration % report_cadence == 0:
        KE_avg = flow.grid_average('KE')
        IE_avg = flow.grid_average('IE')
        Re_avg = flow.grid_average('Re')
        Re_max = flow.max('Re')
        τu1_max = flow.max('τu1')
        τu2_max = flow.max('τu2')
        τs1_max = flow.max('τs1')
        τs2_max = flow.max('τs2')
        τ_max = np.max([τu1_max,τu2_max,τs1_max,τs2_max])
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:5.1e}'.format(solver.iteration, solver.sim_time, Δt)
        log_string += ', KE: {:.2g}, IE: {:.2g}, Re: {:.2g} ({:.2g})'.format(KE_avg, IE_avg, Re_avg, Re_max)
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
