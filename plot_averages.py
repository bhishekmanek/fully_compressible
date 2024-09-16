"""
Plot radial profile outputs from joint analysis file.

Usage:
    plot_averages.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is last 10% of timespan.

    --output=<output>    Output directory; if blank a guess based on likely case name will be made
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py
import scipy.integrate as sci

import logging
logger = logging.getLogger(__name__.split('.')[-1])
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

from docopt import docopt
args = docopt(__doc__)

import dedalus.public as de
from dedalus.tools import post
file = args['<file>']
case = args['<file>'].split('/')[0]
print(case)
if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = case +'/'
    output_path = pathlib.Path(data_dir).absolute()

data = {}
z = None
times = None
logger.debug("opening file: {}".format(file))
f = h5py.File(file, 'r')
data_slices = (slice(None), 0, slice(None))
for task in f['tasks']:
    if '(z)' in task: # fluxes denoted with 'f(z)'
        logger.info("task: {}".format(task))
        if task in data:
            data[task] = np.append(data[task], f['tasks'][task][data_slices], axis=0)
        else:
            data[task] = np.array(f['tasks'][task][data_slices])
        if z is None:
            z = f['tasks'][task].dims[2][0][:]
if times is None:
    times = f['scales/sim_time'][:]
else:
    times = np.append(times, f['scales/sim_time'][:])
f.close()

print(times.shape)
if args['--times']:
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)

else:
    # choose last 10%
    t_max = np.max(times)
    t_min = 0.9*t_max

print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(times), max(times)))

i_t_min = np.argmin(np.abs(times-t_min))
i_t_max = np.argmin(np.abs(times-t_max))

times = times[i_t_min:i_t_max]
for task in data:
    data[task] = data[task][i_t_min:i_t_max,:]
print(times.shape)



def time_avg(f, axis=0):
    n_avg = f.shape[axis]
    return np.squeeze(np.sum(f, axis=axis))/n_avg

s_avg = time_avg(data['s(z)'])
fig, ax = plt.subplots(figsize=(4.5,4/1.5))
fig.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
for si in data['s(z)']:
    ax.plot(z, si, alpha=0.3)
ax.plot(z, s_avg, linewidth=2, color='black')
fig.savefig(f'{str(output_path)}/thermal_profile.pdf')
fig.savefig(f'{str(output_path)}/thermal_profile.png')

F_h = time_avg(data['F_h(z)'])
F_κ = time_avg(data['F_κ(z)'])
F_KE = time_avg(data['F_KE(z)'])
F_PE = time_avg(data['F_PE(z)'])
#F_μ_avg = time_avg(data['<Fμr>'])

norm = 1 #1/F_κ[-1]

F_h = F_h*norm
F_κ = F_κ*norm
F_KE = F_KE*norm
F_PE = F_PE*norm
#F_tot = F_h + F_KE + F_PE + F_κ
F_tot = F_h + F_KE + F_κ

fig_hr, ax_hr = plt.subplots(figsize=(4.5,4/1.5))
fig_hr.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
ax_hr.plot(z, F_tot, color='black', label=r'$F_\mathrm{tot}$', linewidth=3)
ax_hr.plot(z, F_h, label=r'$F_\mathrm{h}$')
ax_hr.plot(z, F_KE, label=r'$F_\mathrm{KE}$')
#ax_hr.plot(z, F_PE, label=r'$F_\mathrm{PE}$')
ax_hr.plot(z, F_κ, label=r'$F_\kappa$')
#ax_hr.axhline(y=0, linestyle='dashed', color='darkgrey', zorder=0)
#ax_hr.plot(r, F_μ, label=r'$F_\mu$')
ax_hr.legend()
ax_hr.set_ylabel(r'$L$')
ax_hr.set_xlabel(r'$z$')
fig_hr.savefig(f'{str(output_path)}/flux_balance.pdf')
fig_hr.savefig(f'{str(output_path)}/flux_balance.png', dpi=300)
