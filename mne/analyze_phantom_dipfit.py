# -*- coding: utf-8 -*-
"""
===================================
Elekta phantom data with dipole fit
===================================
"""
# authors: Amit & Alex & Eric

from __future__ import print_function

from itertools import product

import numpy as np
import pandas as pd
import mne
from mne.parallel import parallel_func

from phantom_helpers import get_data, plot_errors, get_bench_params
from phantom_helpers import get_dataset

import time

# base_path, postfix = get_dataset('aston')
base_path, postfix = get_dataset('')

(maxfilter_options, dipole_amplitudes, dipole_indices,
 actual_pos, actual_ori, bads) = get_bench_params(base_path)

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
    # Do the dipole fit
    t_peak = 66e-3  # only fit the largest peak
    evoked.crop(t_peak, t_peak)

    t_start = time.time()
    dip = mne.fit_dipole(evoked, cov, sphere)[0]
    t_end = time.time() - t_start

    pos = dip.pos[0]
    ori = dip.ori[0]

    pos_error = 1e3 * np.linalg.norm(pos - actual_pos[di - 1])
    ori_error = np.arccos(np.abs(np.sum(ori * actual_ori[di - 1])))
    amp_error = np.mean(np.abs(da - dip.amplitude / 1.e-9))

    print(" Location Error=%s mm" % np.round(pos_error, 1))
    return pd.DataFrame([(actual_pos[di - 1][0], actual_pos[di - 1][1],
                          actual_pos[di - 1][2], pos[0], pos[1], pos[2],
                          actual_ori[di - 1][0], actual_ori[di - 1][1],
                          actual_ori[di - 1][2], ori[0], ori[1], ori[2],
                          di, da, mf, pos_error, ori_error, amp_error, t_end,
                          alpha, n_mxne_iter, depth, weight)],
                        columns=columns)


parallel, prun, _ = parallel_func(run, n_jobs=1)
errors = parallel([prun(da, di, mf) for mf, da, di in
                   product(maxfilter_options, dipole_amplitudes,
                           dipole_indices)])
errors = pd.concat(errors, axis=0, ignore_index=True)


plot_errors(errors, 'dipfit', postfix=postfix)
