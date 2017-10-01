
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import find_events, fit_dipole
from mne.beamformer import rap_music
from mne.inverse_sparse import mixed_norm

from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

from mayavi import mlab

from phantom_helpers import compute_error
import pandas as pd
import seaborn as sns

actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = 100.  # nAm

data_path = bst_phantom_elekta.data_path()

raw_fname = op.join(data_path, 'kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif')
raw = read_raw_fif(raw_fname)

events = find_events(raw, 'STI201')

raw.info['bads'] = ['MEG2421']

raw.fix_mag_coil_types()
raw = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.))
raw.filter(None, 40., fir_design='firwin')

tmin, tmax = -0.1, 0.1
event_id = list(range(1, 33))
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, -0.01),
                    decim=3, preload=True)

# Compute the forward Operator
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)

src = mne.setup_volume_source_space(
    subject=None, pos=3.5, mri=None,
    sphere=(0.0, 0.0, 0.0, 80.0), bem=None, mindist=5.0,
    exclude=2.0)

fwd = mne.make_forward_solution(
    epochs.info, trans=None, src=src, bem=sphere, eeg=False,
    meg=True)

cov = mne.compute_covariance(epochs, tmax=0.)

df_errors = []

for event_id in [5, 15, 25]:
    evoked = epochs[str(event_id)].average()
    n_times = len(evoked.times)

    actual_params = dict(actual_pos=actual_pos[event_id],
                         actual_ori=actual_ori[event_id],
                         actual_amp=actual_amp)

    # RAP-MUSIC
    dip_music = rap_music(evoked, fwd, cov, n_dipoles=1,
                          return_residual=False)[0]
    error_music = compute_error(dip_music.pos[0], dip_music.ori[0],
                                dip_music.amplitude, **actual_params)

    # Mixed norm
    dip_mxne = mixed_norm(evoked, fwd, cov, alpha=70., n_mxne_iter=1,
                          depth=0.99, return_residual=False,
                          return_as_dipoles=True)
    amp_max = [np.max(d.amplitude) for d in dip_mxne]
    idx_max = np.argmax(amp_max)
    dip_mxne = dip_mxne[idx_max]

    idx_max = np.argmax(dip_mxne.amplitude)
    error_mxne = compute_error(dip_mxne.pos[idx_max], dip_mxne.ori[idx_max],
                               dip_mxne.amplitude, **actual_params)

    # Iterative mixed norm
    dip_irmxne = mixed_norm(evoked, fwd, cov, alpha=70., n_mxne_iter=10,
                            depth=0.99, return_residual=False,
                            return_as_dipoles=True)
    amp_max = [np.max(d.amplitude) for d in dip_irmxne]
    idx_max = np.argmax(amp_max)
    dip_irmxne = dip_irmxne[idx_max]

    idx_max = np.argmax(dip_irmxne.amplitude)
    error_irmxne = compute_error(dip_irmxne.pos[0], dip_irmxne.ori[idx_max],
                                 dip_irmxne.amplitude, **actual_params)

    # Dipole fit
    idx_peak = np.argmax(evoked.copy().pick_types(meg='grad').data.std(axis=0))
    t_peak = evoked.times[idx_peak]
    evoked.crop(t_peak, t_peak)

    dip_fit = fit_dipole(evoked, cov, sphere, n_jobs=1)[0]
    error_dipfit = compute_error(dip_fit.pos[0], dip_fit.ori[0],
                                 dip_fit.amplitude, **actual_params)

    df = pd.DataFrame.from_dict({'dipfit': error_dipfit,
                                 'music': error_music,
                                 'mxne': error_mxne,
                                 'irmxne': error_irmxne}).T
    df.index.name = 'method'
    df = df.reset_index()
    df['dip_index'] = event_id
    df_errors.append(df)

df = pd.concat(df_errors, axis=0).reset_index(drop=True)

columns_rename = {'loc_error': 'Loc. Error (mm)',
                  'ori_error': 'Ori. Error (rad)',
                  'amp_error': 'Amp. Error (nAm)',
                  'dip_index': 'Diple Index'}
df.rename(columns=columns_rename, inplace=True)
##########################################################################
# Now we can compare to the actual locations, taking the difference in mm:

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))

ax1, ax2, ax3 = axes
x = columns_rename['dip_index']
sns.barplot(x=x, y=columns_rename['loc_error'], hue='method', data=df, ax=ax1)
sns.barplot(x=x, y=columns_rename['ori_error'], hue='method', data=df, ax=ax2)
sns.barplot(x=x, y=columns_rename['amp_error'], hue='method', data=df, ax=ax3)

for ax in axes[1:]:
    ax.legend_.remove()

fig.tight_layout()
plt.show()


def plot_pos_ori(pos, ori, color=(0., 0., 0.)):
    mlab.points3d(pos[0], pos[1], pos[2], scale_factor=0.005,
                  color=color)
    mlab.quiver3d(pos[0], pos[1], pos[2],
                  ori[0], ori[1], ori[2],
                  scale_factor=0.03,
                  color=color)

# mne.viz.plot_alignment(epochs.info, bem=sphere, surfaces=[])
# plot_pos_ori(actual_pos[event_id],
#              actual_ori[event_id], color=(0., 0., 0.))

colors = [(0., 0., .8), (1., 0.5, 0.), (0., 1., 0.), (1., 0., 0.)]
for i_m, (m, col) in enumerate(zip(df.index, colors)):
    event_id = df['Diple Index'][i_m]
    plot_pos_ori(actual_pos[event_id],
                 actual_ori[event_id], color=(0., 0., 0.))

    dip_pos = df[['loc_x', 'loc_y', 'loc_z']].iloc[i_m:i_m + 1].values[0]
    dip_ori = df[['ori_x', 'ori_y', 'ori_z']].iloc[i_m:i_m + 1].values[0]

    plot_pos_ori(dip_pos, dip_ori, color=col)
