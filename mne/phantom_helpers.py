# -*- coding: utf-8 -*-
"""
================================
Helpers for phantom localization
================================
"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne


def plot_errors(errors, kind, postfix='', ylim=(0, 20), xkey='maxfilter',
                xlabel_mapping={"False": 'Raw', "True": 'SSS',
                                'mne': 'SSS$_{MNE}$'}, error_type='pos_error'):
    errors_init = errors.copy()
    errors = errors.copy()
    errors[xkey] = errors[xkey].astype(str)
    if xlabel_mapping is not None:
        errors[xkey] = errors[xkey].apply(lambda x: xlabel_mapping[x])
    dipole_amplitudes = errors['dipole_amplitude'].unique()
    dipole_indices = errors['dipole_index'].unique()
    n_amplidudes = dipole_amplitudes.size
    xticklabels = errors[xkey].unique()
    n_maxfilter = errors[xkey].unique().size

    xs = np.arange(n_maxfilter)
    fig, axs = plt.subplots(n_amplidudes + 1, 1, figsize=(4, 8))
    for ai, da in enumerate(dipole_amplitudes):
        ax = axs[ai]
        for di in dipole_indices:
            query = 'dipole_index==%s and dipole_amplitude==%s' % (di, da)
            this_errors = errors.query(query)
            this_errors = this_errors[[xkey, error_type]].set_index(xkey)
            this_errors = this_errors.loc[xticklabels].values
            ax.plot(xs, this_errors, label='%d' % di)
        if error_type == 'pos_error':
            ylabel = 'Position Error (mm)'
        elif error_type == 'ori_error':
            ylabel = 'Orientation Error (Rad)'
            ylim = (0, 2)
        elif error_type == 'amp_error':
            ylabel = 'Amplitude Error (nAm)'

        ax.set(title='%d nAm' % da, ylim=ylim, xticks=xs,
               ylabel=ylabel, xlim=[0, len(xticklabels) - 1])
        ax.set(xticklabels=[''] * len(xs))
        if ai == len(dipole_amplitudes) - 1:
            ax.set(xticklabels=xticklabels)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.25), ncol=2)
        ax.grid(True)
    fig.tight_layout()
    axs[-1].set_visible(False)
    basename = ('phantom_errors_%s%s.' % (kind, postfix))
    for ext in ('png', 'pdf'):
        fig_fname = basename + ext
        plt.savefig(op.join('figures', fig_fname))
    plt.show()
    postfix = ''
    basename = ('phantom_errors_%s%s.' % (kind, postfix))
    errors_init.to_csv(op.join('results', basename + 'csv'), index=False)


def get_fwd(base_path):
    # They all have approximately the same dev_head_t
    if "phantom_aston" in base_path:
        info = mne.io.read_info(base_path +
                                '/Amp1000_IASoff/Amp1000_Dip5_IASoff.fif')
    else:
        info = mne.io.read_info(base_path + '/1000nAm/dip05_1000nAm.fif')
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)
    src_fname = op.join(base_path, 'phantom-src.fif')
    if not op.isfile(src_fname):
        mne.setup_volume_source_space(
            subject=None, pos=3.5, mri=None,
            sphere=(0.0, 0.0, 0.0, 80.0), bem=None, mindist=5.0,
            exclude=2.0).save(src_fname)
    src = mne.read_source_spaces(src_fname)
    fwd_fname = op.join(base_path, 'phantom-fwd.fif')
    if not op.isfile(fwd_fname):
        mne.write_forward_solution(fwd_fname, mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=False,
            meg=True))
    fwd = mne.read_forward_solution(fwd_fname)
    return src, fwd


def get_data(base_path, dipole_idx, dipole_amplitude, use_maxwell_filter,
             bads=[], show=False):
    if "phantom_aston" in base_path:
        data_path = base_path + '/Amp%d_IASoff/' % dipole_amplitude
        fname = 'Amp%d_Dip%d_IASoff.fif' % (dipole_amplitude, dipole_idx)
        stim_channel = 'SYS201'
        assert use_maxwell_filter in ['mne', False]
    else:
        data_path = base_path + '/%dnAm/' % dipole_amplitude
        if use_maxwell_filter is True:
            fname = 'dip%02d_%dnAm_sss.fif' % (dipole_idx, dipole_amplitude)
        else:
            fname = 'dip%02d_%dnAm.fif' % (dipole_idx, dipole_amplitude)
        stim_channel = 'STI201'

    raw_fname = op.join(data_path, fname)
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose='error')
    raw.info['bads'] = bads

    if "phantom_aston" in base_path:
        raw.crop(20, None)
    events = mne.find_events(raw, stim_channel=stim_channel)
    if show:
        raw.plot(events=events)
    if show:
        raw.plot_psd(tmax=np.inf, fmax=60, average=False)

    raw.fix_mag_coil_types()
    if use_maxwell_filter == 'mne':
        # Use Maxwell filtering from MNE
        raw = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.))
        if show:
            raw.plot(events=events)

    #######################################################################
    # We know our phantom produces sinusoidal bursts below 25 Hz, so let's
    # filter.
    raw.filter(None, 40., h_trans_bandwidth='auto', filter_length='auto',
               phase='zero')
    if show:
        raw.plot(events=events)

    #######################################################################
    # Now we epoch our data, average it
    tmin, tmax = -0.15, 0.1
    event_id = events[0, 2]
    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax, baseline=(None, -0.05),
        preload=True)
    evoked = epochs.average()

    if show:
        evoked.plot(spatial_colors=True)
    if show:
        evoked.plot_joint()

    evoked.crop(0, None)
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.08)
    cov = mne.compute_covariance(epochs, tmax=-0.05)
    print(fname + " nave=%d" % evoked.nave, end='')
    return epochs, evoked, cov, sphere


def get_bench_params(base_path):
    if "aston" not in base_path:
        dipole_amplitudes = [20, 100, 200, 1000]
        dipole_indices = [5, 6, 7, 8]
        maxfilter_options = [False, True, 'mne']
        actual_pos, actual_ori = mne.dipole.get_phantom_dipoles('otaniemi')
        bads = ['MEG2233', 'MEG2422', 'MEG0111']
    else:
        dipole_amplitudes = [20, 200, 1000]
        dipole_indices = [5, 6, 7, 8, 9, 10, 11, 12]
        maxfilter_options = [False, 'mne']
        actual_pos = mne.dipole.get_phantom_dipoles('vectorview')[0]
        bads = ['MEG1323', 'MEG1133', 'MEG0613', 'MEG1032', 'MEG2313',
                'MEG1133', 'MEG0613', 'MEG0111', 'MEG2423']
    return (maxfilter_options, dipole_amplitudes, dipole_indices, actual_pos,
            actual_ori, bads)


def get_dataset(name):
    if name != 'aston':
        base_path = op.join(op.dirname(__file__), '..', '..', 'phantom')
        postfix = ''
    else:
        base_path = op.join(op.dirname(__file__), '..', '..', 'phantom_aston')
        postfix = '_aston'
    return base_path, postfix
