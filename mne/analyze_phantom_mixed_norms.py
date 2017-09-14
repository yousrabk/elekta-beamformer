# -*- coding: utf-8 -*-
"""
=============================
Elekta phantom data with MxNE
=============================
"""

from __future__ import print_function

import numpy as np

from phantom_helpers import (get_data, get_fwd, plot_errors, actual_pos,
                             maxfilter_options, dipole_indices,
                             dipole_amplitudes)
from mne.inverse_sparse import mixed_norm

errors = np.empty(
    (len(maxfilter_options), len(dipole_amplitudes), len(dipole_indices)))

src, fwd = get_fwd()
loose, depth = None, 0.9

for ui, use_maxwell_filter in enumerate(maxfilter_options):
    for ai, dipole_amplitude in enumerate(dipole_amplitudes):
        print(('Processing : %4d nAm : SSS=%s'
               % (dipole_amplitude, use_maxwell_filter)).ljust(40), end='')
        for di, dipole_idx in enumerate(dipole_indices):
            epochs, evoked, cov, sphere = \
                get_data(dipole_idx, dipole_amplitude, use_maxwell_filter)
            stc = mixed_norm(evoked, fwd, cov, alpha=60, n_mxne_iter=10,
                             loose=loose, depth=depth, return_residual=False)
            print('number of sources found: %s lh - %s rh'
                  % (len(stc.vertices[0]), len(stc.vertices[1])))
            # Find the best idx
            idx_max = np.argmax(np.mean(stc.data, axis=1))
            if idx_max < len(stc.vertices[0]):
                vertno_max = stc.vertices[0][idx_max]
            else:
                vertno_max = stc.vertices[1][idx_max - len(stc.vertices[0])]
            pos = fwd['src'][0]['rr'][vertno_max]
            # pos = fwd['src'][0]['rr'][np.where(src[0]['vertno'] ==
            #                                    stc.vertices[0][0])[0][0]]
            errors[ui, ai, di] = 1e3 * np.linalg.norm(
                pos - actual_pos[dipole_idx - 1])
        print(np.round(errors[ui, ai], 1))

plot_errors(errors, 'MxNE')
