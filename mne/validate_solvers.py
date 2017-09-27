
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

cov = mne.compute_covariance(epochs, tmax=0)

event_id = 10
evoked = epochs[str(event_id)].average()
n_times = len(evoked.times)

# RAP-MUSIC
dip_music = rap_music(evoked, fwd, cov, n_dipoles=1,
                      return_residual=False)[0]

# Mixed norm
dip_mxne = mixed_norm(evoked, fwd, cov, alpha=10., n_mxne_iter=1,
                      depth=0.9, return_residual=False,
                      return_as_dipoles=True)
amp_max = [np.max(d.amplitude) for d in dip_mxne]
idx_max = np.argmax(amp_max)
dip_mxne = dip_mxne[idx_max]

# Iterative mixed norm
dip_irmxne = mixed_norm(evoked, fwd, cov, alpha=10., n_mxne_iter=10,
                        depth=0.9, return_residual=False,
                        return_as_dipoles=True)
amp_max = [np.max(d.amplitude) for d in dip_irmxne]
idx_max = np.argmax(amp_max)
dip_irmxne = dip_irmxne[idx_max]

# Dipole fit
idx_peak = np.argmax(evoked.copy().pick_types(meg='grad').data.std(axis=0))
t_peak = evoked.times[idx_peak]
evoked.crop(t_peak, t_peak)

dip_fit = fit_dipole(evoked, cov, sphere, n_jobs=1)[0]

##########################################################################
# Now we can compare to the actual locations, taking the difference in mm:
names = ['dipfit', 'music', 'mxne', 'irmxne']
actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = [100.] * len(names)  # nAm
dip_fit.amplitude = np.concatenate(np.array([dip_fit.amplitude] * n_times))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))

dipoles_pos = [dip_fit.pos[0], dip_music.pos[0],
               dip_mxne.pos[0], dip_irmxne.pos[0]]
dipoles_ori = [dip_fit.ori[0], dip_music.ori[0],
               dip_mxne.ori[0], dip_irmxne.ori[0]]
dipoles_amp = [dip_fit.amplitude, dip_music.amplitude,
               dip_mxne.amplitude, dip_irmxne.amplitude]

diffs = 1000 * np.sqrt(np.sum((dipoles_pos - actual_pos[event_id]) ** 2,
                              axis=-1))
ax1.bar(range(len(names)), diffs)
ax1.set_xlabel('Dipole index')
ax1.set_ylabel('Loc. error (mm)')

angles = np.arccos(np.abs(np.sum(dipoles_ori * actual_ori[event_id], axis=1)))
ax2.bar(range(len(names)), angles)
ax2.set_xlabel('Dipole index')
ax2.set_ylabel('Angle error (rad)')

amps = np.abs(actual_amp - np.array(dipoles_amp).max(axis=1) / 1e-9)
ax3.bar(range(len(names)), amps)
ax3.set_xlabel('Dipole index')
ax3.set_ylabel('Amplitude error (nAm)')

fig.tight_layout()
plt.show()


def plot_pos_ori(pos, ori, color=(1., 0., 0.)):
    mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2], scale_factor=0.005,
                  color=color)
    mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
                  ori[:, 0], ori[:, 1], ori[:, 2],
                  scale_factor=0.03,
                  color=color)  # [color] * len(ori))

mne.viz.plot_alignment(epochs.info, bem=sphere, surfaces=[])
plot_pos_ori(actual_pos[event_id:event_id + 1],
             actual_ori[event_id:event_id + 1], color=(1., 0., 0.))
dipoles_pos = np.concatenate(dipoles_pos, axis=0).reshape(-1, 3)
dipoles_ori = np.concatenate(dipoles_ori, axis=0).reshape(-1, 3)
plot_pos_ori(dipoles_pos, dipoles_ori, color=(0., 1., 0.))
