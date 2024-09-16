"""
Plot scalar outputs from scalar_output.h5 file.

Usage:
    plot_scalar.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is whole timespan.
    --output=<output>    Output directory; if blank, a guess based on <file> location will be made.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)
file = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

f = h5py.File(file, 'r')
data = {}
data_slice = (slice(None),0,0) # valid for 2-D, need general approach
t = f['scales/sim_time'][:]
for key in f['tasks']:
    data[key] = f['tasks/'+key][data_slice]
f.close()

if args['--times']:
    subrange = True
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)
    print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(t), max(t)))
else:
    subrange = False

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(t, data['Re'], label='Re')
ax_M = ax[0].twinx()
ax_M.plot(t, data['Ma_ad'], label='Ma', color='tab:orange')
ax[1].plot(t, data['Re'], label='Re')
ax[1].set_yscale('log')
for axi in ax:
    if subrange:
        axi.set_xlim(t_min,t_max)
    axi.set_xlabel('time')
    axi.set_ylabel('Re')
    axi.legend(loc='lower left')
ax_M.set_ylabel('Ma')
ax_M.legend(loc='lower right')
fig.tight_layout()
fig.savefig(f'{str(output_path)}/Re.pdf')
fig.savefig(f'{str(output_path)}/Re.png', dpi=300)

energy_keys = ['KE','IE','PE']

fig, ax = plt.subplots(nrows=2, sharex=True)
for key in energy_keys:
    ax[0].plot(t, data[key], label=key)
ax[1].plot(t, data['KE'], label='KE')
ax2 = ax[1].twinx()
ax2.plot(t, data['Re'], label='Re', linestyle='dotted')
ax2.legend(loc='upper left')

for axi in ax:
    if subrange:
        axi.set_xlim(t_min,t_max)
    axi.set_xlabel('time')
    axi.set_ylabel('energy density')
    axi.legend(loc='lower left')
fig.savefig(f'{str(output_path)}/energies.pdf')
fig.savefig(f'{str(output_path)}/energies.png', dpi=300)
for axi in ax:
    axi.set_yscale('log')
fig.savefig(f'{str(output_path)}/log_energies.pdf')
fig.savefig(f'{str(output_path)}/log_energies.png', dpi=300)

fig, ax = plt.subplots(nrows=2, sharex=True)
for key in energy_keys:
    ax[0].plot(t, data[key]-data[key][0], label=key+"'")
ax[1].plot(t, data['KE'], label='KE')

for axi in ax:
    if subrange:
        axi.set_xlim(t_min,t_max)
    axi.set_xlabel('time')
    axi.set_ylabel('energy density')
    axi.legend(loc='lower left')
fig.savefig(f'{str(output_path)}/energies_fluctuating.pdf')
fig.savefig(f'{str(output_path)}/energies_fluctuating.png', dpi=300)


fig, ax = plt.subplots(nrows=2, sharex=True)
for i in [0,1]:
    ax[i].plot(t, data['τ_c'], label=r'$\tau_{c}$')
    ax[i].plot(t, data['τ_u'], label=r'$\tau_{u}$')
    ax[i].plot(t, data['τ_s'], label=r'$\tau_{s}$')

for axi in ax:
    if subrange:
        axi.set_xlim(t_min,t_max)
    axi.set_xlabel('time')
    axi.set_ylabel(r'$L_\inf(\tau)$')
    axi.legend(loc='lower left')
ax[1].set_yscale('log')
fig.savefig(f'{str(output_path)}/tau_error.pdf')
fig.savefig(f'{str(output_path)}/tau_error.png', dpi=300)

benchmark_set = ['KE', 'IE', 'Re', 'Ma_ad']
i_ten = int(0.9*data[benchmark_set[0]].shape[0])
for benchmark in benchmark_set:
    print("{:s} = {:14.12g} +- {:4.2g} (averaged from {:g}-{:g})".format(benchmark, np.mean(data[benchmark][i_ten:]), np.std(data[benchmark][i_ten:]), t[i_ten], t[-1]))
print()
for benchmark in benchmark_set:
    print("{:s} = {:14.12g} (at t={:g})".format(benchmark, data[benchmark][-1], t[-1]))
print("total simulation time {:6.2g}".format(t[-1]-t[0]))
