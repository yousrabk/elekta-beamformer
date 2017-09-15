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
# import mne
from mne.parallel import parallel_func
from mne.inverse_sparse import mixed_norm

from phantom_helpers import get_data, plot_errors, get_bench_params, get_fwd
from phantom_helpers import get_dataset

# base_path, postfix = get_dataset('aston')
base_path, postfix = get_dataset('')

maxfilter_options, dipole_amplitudes, dipole_indices, actual_pos, bads =\
    get_bench_params(base_path)

src, fwd = get_fwd(base_path)

columns = ['dipole_index', 'dipole_amplitude', 'maxfilter', 'error']
loose, depth = 1., 0.9


def run(da, di, mf):
    print(('Processing : %4d nAm (dip %d) : SSS=%s'
          % (da, di, mf)).ljust(42), end='')
    epochs, evoked, cov, sphere = get_data(
        base_path, di, da, mf, bads=bads)
    # Do MxNE
    dip = mixed_norm(
        evoked, fwd, cov, alpha=60., n_mxne_iter=1,
        depth=depth, return_residual=False, return_as_dipoles=True)

    print(" n_sources=%s" % len(dip), end='')
    # If stc
    # idx_max = np.argmax(np.mean(stc.data, axis=1))
    # vertno_max = stc.vertices[idx_max]
    # pos = src[0]['rr'][vertno_max]

    # If dipoles
    amp_max = [np.max(d.amplitude) for d in dip]
    idx_max = np.argmax(amp_max)
    pos = dip[idx_max].pos[0]
    error = 1e3 * np.linalg.norm(pos - actual_pos[di - 1])

    print(" Error=%s mm" % np.round(error, 1))
    return pd.DataFrame([(di, da, mf, error)], columns=columns)

parallel, prun, _ = parallel_func(run, n_jobs=1)
errors = parallel([prun(da, di, mf) for mf, da, di in
                   product(maxfilter_options, dipole_amplitudes,
                           dipole_indices)])
errors = pd.concat(errors, axis=0, ignore_index=True)

plot_errors(errors, 'MxNE', postfix=postfix)
