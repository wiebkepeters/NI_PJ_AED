# %%
import wave
import numpy as np
# import scipy
import librosa
from IPython import embed
import pickle
import os
from sklearn import preprocessing


# def load_audio(filename, mono=True, fs=44100):
#     """Load audio file into numpy array
#     Supports 24-bit wav-format

#     Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python

#     Parameters
#     ----------
#     filename:  str
#         Path to audio file
#     mono : bool
#         In case of multi-channel audio, channels are averaged into single channel.
#         (Default value=True)
#     fs : int > 0 [scalar]
#         Target sample rate, if input audio does not fulfil this, audio is resampled.
#         (Default value=44100)
#     Returns
#     -------
#     audio_data : numpy.ndarray [shape=(signal_length, channel)]
#         Audio
#     sample_rate : integer
#         Sample rate
#     """
def create_folder(_fold_path):
    if not os.path.exists(_fold_path):
        os.makedirs(_fold_path)

def load_desc_file(_desc_file):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), __class_labels[words[-1]]])
    return _desc_dict


def extract_mbe(_y, _sr, _nfft, _nb_mel):
    # window = scipy.signal.get_window('hamming', Nx=1764) #, fftbins=False) <- symmetric window
    spec, _ = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=882, window="hamming", power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, htk=True)

    return np.log(np.dot(mel_basis, spec))

def load_mbe(file):
    """Load log mel energies into numpy array. Log mel energies are contained in the ... dataset.
    Returns
    """
    with open(file, "rb") as f:
        mbe = pickle.load(f, encoding="latin1")['feat'][0]

    return mbe

# ###################################################################
#              Main script starts here
# ###################################################################

is_mono = True
#%%
__class_labels = {
    'alarms_and_sirens': 0,
    'baby_crying': 1,
    'bird_singing': 2,
    'bus': 3,
    'cat_meowing': 4,
    'crowd_applause': 5,
    'crowd_cheering': 6,
    'dog_barking': 7,
    'footsteps': 8,
    'glass_smash': 9,
    'gun_shot': 10,
    'horsewalk': 11,
    'mixer': 12,
    'motorcycle': 13,
    'rain': 14,
    'thunder': 15
}

#%%
# location of data.
folds_list = [1, 2, 3, 4]
evaluation_setup_folder = '/home/lukas/Documents/Studium/SoSe21/NIP/data/TUT-SED-synthetic-2016.features/TUT-SED-synthetic-2016'
audio_folder = '/home/lukas/Documents/Studium/SoSe21/NIP/data/TUT-SED-synthetic-2016.features/TUT-SED-synthetic-2016/features'

# Output
feat_folder = './data'
create_folder(feat_folder)

# User set parameters
nfft = 1024
win_len = 1764
hop_len = win_len // 2
nb_mel_bands = 40
sr = 44100

# -----------------------------------------------------------------------
# Feature extraction and label generation
# -----------------------------------------------------------------------
# Load labels
train_file = os.path.join(evaluation_setup_folder, 'label.txt')#street_fold{}_train.txt'.format(1))
# evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(1))
desc_dict = load_desc_file(train_file)
# desc_dict.update(load_desc_file(evaluate_file)) # contains labels for all the audio in the dataset

# Extract features for all audio files, and save it along with labels
data = []
labels = []

for audio_filename in os.listdir(audio_folder):
    if audio_filename[-3:] != 'kle':
        continue
    audio_file = os.path.join(audio_folder, audio_filename)

    mbe = load_mbe(audio_file)
    audio_file = audio_filename[:-7] + 'wav'

    label = np.zeros((mbe.shape[0], len(__class_labels)))
    tmp_data = np.array(desc_dict[audio_file])
    frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = 1


tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_file[:-4]))
np.savez(tmp_feat_file, mbe, label)

# -----------------------------------------------------------------------
# Feature Normalization
# -----------------------------------------------------------------------

# for fold in folds_list:
#     train_file = os.path.join(evaluation_setup_folder, 'street_fold{}_train.txt'.format(fold))
#     evaluate_file = os.path.join(evaluation_setup_folder, 'street_fold{}_evaluate.txt'.format(fold))
#     train_dict = load_desc_file(train_file)
#     test_dict = load_desc_file(evaluate_file)

#     X_train, Y_train, X_test, Y_test = None, None, None, None
#     for key in train_dict.keys():
#         tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
#         dmp = np.load(tmp_feat_file)
#         tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
#         if X_train is None:
#             X_train, Y_train = tmp_mbe, tmp_label
#         else:
#             X_train, Y_train = np.concatenate((X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)

#     for key in test_dict.keys():
#         tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(key, 'mon' if is_mono else 'bin'))
#         dmp = np.load(tmp_feat_file)
#         tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
#         if X_test is None:
#             X_test, Y_test = tmp_mbe, tmp_label
#         else:
#             X_test, Y_test = np.concatenate((X_test, tmp_mbe), 0), np.concatenate((Y_test, tmp_label), 0)

#     # Normalize the training data, and scale the testing data using the training data weights
#     scaler = preprocessing.StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     normalized_feat_file = os.path.join(feat_folder, 'mbe_{}_fold{}.npz'.format('mon' if is_mono else 'bin', fold))
#     np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test)
#     print('normalized_feat_file : {}'.format(normalized_feat_file))

# %%
