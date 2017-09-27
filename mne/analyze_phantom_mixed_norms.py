# -*- coding: utf-8 -*-
"""
=============================
Elekta phantom data with LCMV
=============================
"""
# authors: Amit & Alex & Eric

from __future__ import print_function

from itertools import product

import numpy as np
import pandas as pd
import mne
from mne.parallel import parallel_func
from mne.inverse_sparse import mixed_norm
from mne.minimum_norm import make_inverse_operator, apply_inverse

from phantom_helpers import get_data, plot_errors, get_bench_params, get_fwd
from phantom_helpers import get_dataset

from mayavi import mlab
import time

# base_path, postfix = get_dataset('aston')
base_path, postfix = get_dataset('')

(maxfilter_options, dipole_amplitudes, dipole_indices, actual_pos,
 actual_ori, bads) = get_bench_params(base_path)

src, fwd = get_fwd(base_path)

columns = ['actual_pos_x', 'actual_pos_y', 'actual_pos_z',
           'estimated_pos_x', 'estimated_pos_y', 'estimated_pos_z',
           'actual_ori_x', 'actual_ori_y', 'actual_ori_z',
           'estimated_ori_x', 'estimated_ori_y', 'estimated_ori_z',
           'dipole_index', 'dipole_amplitude', 'maxfilter',
           'pos_error', 'ori_error', 'amp_error', 'run_time',
           'alpha', 'n_mxne_iter', 'depth', 'weight']

loose, depth = 1., 0.95
n_mxne_iter = 1
weight, weights_min = '', None
# dipole_indices = [[5, 6], [5, 7], [5, 8], [6, 7], [6, 8], [7, 8]]


def plot_pos_ori(pos, ori, color=(1., 0., 0.)):
    mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2], scale_factor=0.005,
                  color=color)
    mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
                  ori[:, 0], ori[:, 1], ori[:, 2],
                  scale_factor=0.000005,
                  color=[color] * len(ori))


def run(da, di, mf):
    print(('Processing : %4d nAm (dip %d) : SSS=%s'
          % (da, di, mf)).ljust(42), end='')
    if isinstance(di, int):
        epochs, evoked, cov, sphere = get_data(
            base_path, di, da, mf, bads=bads)
    else:
        epochs, _, _, sphere = get_data(
            base_path, di[0], da, mf, bads=bads)
        for i_di in di[1:]:
            epochs_b, _, _, _ = get_data(
                base_path, i_di, da, mf, bads=bads)
            epochs._data[:31, :, :] += epochs_b._data[:31, :, :]

    cov = mne.compute_covariance(epochs, tmax=-0.05)
    evoked = epochs.average()
    evoked.crop(-0.020, None)

    # Compute dSPM solution to be used as weights in MxNE
    if weight == 'dSPM':
        inverse_operator = make_inverse_operator(
            evoked.info, fwd, cov, loose=loose, depth=depth)
        stc_weight = apply_inverse(
            evoked, inverse_operator, lambda2=1. / 9., method='dSPM')
        weights_min = 4.
    else:
        stc_weight = None
        weights_min = None

    # Do MxNE
    alpha = 40.
    t_start = time.time()
    dip = mixed_norm(
        evoked, fwd, cov, alpha=alpha, n_mxne_iter=n_mxne_iter,
        depth=depth, weights=stc_weight, weights_min=weights_min,
        return_residual=False, return_as_dipoles=True)
    t_end = time.time() - t_start

    print(" n_sources=%s" % len(dip), end='')
    amp_max = [np.max(d.amplitude) for d in dip]
    idx_max = np.argmax(amp_max)
    pos = dip[idx_max].pos[0]
    ori = dip[idx_max].ori[0]

    pos_error = 1e3 * np.linalg.norm(pos - actual_pos[di - 1])
    ori_error = np.arccos(np.abs(np.sum(ori * actual_ori[di - 1])))
    amp_error = np.mean(np.abs(da / 2. - dip[idx_max].amplitude / 1.e-9))

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

iterative = '' if n_mxne_iter == 1 else 'ir'
depth = str(depth)[2:]
weight = '_' + weight
name = '%sMxNE_depth_%s%s' % (iterative, depth, weight)

plot_errors(errors, name, postfix=postfix)

# epochs, _, _, sphere = get_data(base_path, 5, 1000, True, bads=bads)
# _, _, _, _, _, actual_ori = get_bench_params(base_path)
# mne.viz.plot_alignment(epochs.info, bem=sphere, surfaces=[])
# actual_pos = np.concatenate(errors['actual_pos'].values, axis=0).reshape(-1, 3)
# plot_pos_ori(actual_pos, actual_ori[errors['dipole_index'] - 1], color=(1., 0., 0.))
# estimated_pos = np.concatenate(errors['estimated_pos'].values, axis=0).reshape(-1, 3)
# estimated_ori = np.concatenate(errors['estimated_ori'].values, axis=0).reshape(-1, 3)
# plot_pos_ori(estimated_pos, estimated_ori, color=(0., 1., 0.))
