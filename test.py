#%%
import torch
import torchaudio

import torchaudio.functional as F
import torchaudio.transforms as T
# %%
torchaudio.set_audio_backend("sox_io")
# %%

#mel feature extraction

# hamming_length = 1764
# hamming_asymm = True

path = "./TUT-SED-synthetic-2016-mix-1.wav"
n_fft = 1024
n_mels = 40
normalized = False
hop_length = 882

sample_rate = 44100
power = 1.0

#%%
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    f_min=0,
    f_max=sample_rate//2,
    n_mels=n_mels,
    power=power,
    window_fn=torch.hamming_window,
    normalized=normalized
)
#%%
waveform, sample_rate = torchaudio.load(path)

# %%

# %%
