# -*- coding: utf-8 -*-
"""
==============================
Elekta Phantom data with MUSIC
==============================
"""

# authors: Amit & Alex & Eric

from __future__ import print_function

from itertools import product

import numpy as np
import pandas as pd
from mne.parallel import parallel_func
from mne.inverse_sparse import gamma_map

from phantom_helpers import get_data, plot_errors, get_bench_params, get_fwd
from phantom_helpers import get_dataset

import time

# base_path, postfix = get_dataset('aston')
base_path, postfix = get_dataset('')

(maxfilter_options, dipole_amplitudes, dipole_indices,
 actual_pos, actual_ori, bads) = get_bench_params(base_path)

_, fwd = get_fwd(base_path)

# columns = ['dipole_index', 'dipole_amplitude', 'maxfilter', 'error']

columns = ['actual_pos_x', 'actual_pos_y', 'actual_pos_z',
           'estimated_pos_x', 'estimated_pos_y', 'estimated_pos_z',
           'actual_ori_x', 'actual_ori_y', 'actual_ori_z',
           'estimated_ori_x', 'estimated_ori_y', 'estimated_ori_z',
           'dipole_index', 'dipole_amplitude', 'maxfilter',
           'pos_error', 'ori_error', 'amp_error', 'run_time',
           'alpha', 'n_mxne_iter', 'depth', 'weight']

alpha, n_mxne_iter, depth, weight = '', '', '', ''


def run(da, di, mf):
    print(('Processing : %4d nAm (dip %d) : SSS=%s'
          % (da, di, mf)).ljust(42), end='')
    epochs, evoked, cov, sphere = get_data(
        base_path, di, da, mf, bads=bads)

    t_start = time.time()
    dip = gamma_map(
        evoked, fwd, cov, alpha=0.5, return_as_dipoles=True, loose=1)
    t_end = time.time() - t_start

    print(" n_sources=%s" % len(dip), end='')
    amp_max = [np.max(d.amplitude) for d in dip]
    idx_max = np.argmax(amp_max)

    pos = dip[idx_max].pos[0]
    ori = dip[idx_max].ori[0]
    amp = dip[idx_max].amplitude

    pos_error = 1e3 * np.linalg.norm(pos - actual_pos[di - 1])
    ori_error = np.arccos(np.abs(np.sum(ori * actual_ori[di - 1])))
    amp_error = np.mean(np.abs(da - amp / 1.e-9))

    print(" Location Error=%s mm" % np.round(pos_error, 1))
    return pd.DataFrame([(actual_pos[di - 1][0], actual_pos[di - 1][1],
                          actual_pos[di - 1][2], pos[0], pos[1], pos[2],
                          actual_ori[di - 1][0], actual_ori[di - 1][1],
                          actual_ori[di - 1][2], ori[0], ori[1], ori[2],
                          di, da, mf, pos_error, ori_error, amp_error, t_end,
                          alpha, n_mxne_iter, depth, weight)],
                        columns=columns)

parallel, prun, _ = parallel_func(run, n_jobs=4)
errors = parallel([prun(da, di, mf) for mf, da, di in
                   product(maxfilter_options, dipole_amplitudes,
                           dipole_indices)])
errors = pd.concat(errors, axis=0, ignore_index=True)

plot_errors(errors, 'gamma_map', postfix=postfix)
