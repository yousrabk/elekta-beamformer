
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

event_id = 15
evoked = epochs[str(event_id)].average()
n_times = len(evoked.times)

actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = 100.  # nAm

# RAP-MUSIC
dip_music = rap_music(evoked, fwd, cov, n_dipoles=1,
                      return_residual=False)[0]
error_music_pos, error_music_ori, error_music_amp = \
    compute_error(actual_pos[event_id], dip_music.pos[0],
                  actual_ori[event_id], dip_music.ori[0],
                  actual_amp, dip_music.amplitude)

# Mixed norm
dip_mxne = mixed_norm(evoked, fwd, cov, alpha=10., n_mxne_iter=1,
                      depth=0.9, return_residual=False,
                      return_as_dipoles=True)
amp_max = [np.max(d.amplitude) for d in dip_mxne]
idx_max = np.argmax(amp_max)
dip_mxne = dip_mxne[idx_max]

error_mxne_pos, error_mxne_ori, error_mxne_amp = \
    compute_error(actual_pos[event_id], dip_mxne.pos[0],
                  actual_ori[event_id], dip_mxne.ori[0],
                  actual_amp, dip_mxne.amplitude)

# Iterative mixed norm
dip_irmxne = mixed_norm(evoked, fwd, cov, alpha=10., n_mxne_iter=10,
                        depth=0.9, return_residual=False,
                        return_as_dipoles=True)
amp_max = [np.max(d.amplitude) for d in dip_irmxne]
idx_max = np.argmax(amp_max)
dip_irmxne = dip_irmxne[idx_max]

error_irmxne_pos, error_irmxne_ori, error_irmxne_amp = \
    compute_error(actual_pos[event_id], dip_irmxne.pos[0],
                  actual_ori[event_id], dip_irmxne.ori[0],
                  actual_amp, dip_irmxne.amplitude)

# Dipole fit
idx_peak = np.argmax(evoked.copy().pick_types(meg='grad').data.std(axis=0))
t_peak = evoked.times[idx_peak]
evoked.crop(t_peak, t_peak)

dip_fit = fit_dipole(evoked, cov, sphere, n_jobs=1)[0]
error_dipfit_pos, error_dipfit_ori, error_dipfit_amp = \
    compute_error(actual_pos[event_id], dip_fit.pos[0],
                  actual_ori[event_id], dip_fit.ori[0],
                  actual_amp, dip_fit.amplitude)


columns = ['loc_x', 'loc_y', 'loc_z', 'ori_x', 'ori_y', 'ori_z',
           'loc_error', 'ori_error', 'amp_error']
index = ['dipfit', 'music', 'mxne', 'irmxne']

d = np.array([[dip_fit.pos[0][0], dip_fit.pos[0][1], dip_fit.pos[0][2],
               dip_fit.ori[0][0], dip_fit.ori[0][1], dip_fit.ori[0][2],
               error_dipfit_pos, error_dipfit_ori, error_dipfit_amp],
              [dip_music.pos[0][0], dip_music.pos[0][1], dip_music.pos[0][2],
               dip_music.ori[0][0], dip_music.ori[0][1], dip_music.ori[0][2],
               error_music_pos, error_music_ori, error_music_amp],
              [dip_mxne.pos[0][0], dip_mxne.pos[0][1], dip_mxne.pos[0][2],
               dip_mxne.ori[0][0], dip_mxne.ori[0][1], dip_mxne.ori[0][2],
               error_mxne_pos, error_mxne_ori, error_mxne_amp],
              [dip_irmxne.pos[0][0], dip_irmxne.pos[0][1],
               dip_irmxne.pos[0][2], dip_irmxne.ori[0][0],
               dip_irmxne.ori[0][1], dip_irmxne.ori[0][2],
               error_irmxne_pos, error_irmxne_ori, error_irmxne_amp]])

data = pd.DataFrame(data=d, index=index, columns=columns)

##########################################################################
# Now we can compare to the actual locations, taking the difference in mm:

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))

sns.barplot(x=index, y='loc_error', data=data, ax=ax1)
sns.barplot(x=index, y='ori_error', data=data, ax=ax2)
sns.barplot(x=index, y='amp_error', data=data, ax=ax3)
plt.show()


def plot_pos_ori(pos, ori, color=(0., 0., 0.)):
    mlab.points3d(pos[0], pos[1], pos[2], scale_factor=0.005,
                  color=color)
    mlab.quiver3d(pos[0], pos[1], pos[2],
                  ori[0], ori[1], ori[2],
                  scale_factor=0.03,
                  color=color)

# mne.viz.plot_alignment(epochs.info, bem=sphere, surfaces=[])
plot_pos_ori(actual_pos[event_id],
             actual_ori[event_id], color=(0., 0., 0.))

colors = [(0., 0., .8), (1., 0.5, 0.), (0., 1., 0.), (1., 0., 0.)]
for i_m, (m, col) in enumerate(zip(index, colors)):
    dip_pos = data[['loc_x', 'loc_y', 'loc_z']][i_m:i_m + 1].values[0]
    dip_ori = data[['ori_x', 'ori_y', 'ori_z']][i_m:i_m + 1].values[0]

    plot_pos_ori(dip_pos, dip_ori, color=col)
