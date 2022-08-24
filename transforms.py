import numpy as np
import librosa
import os
import config as c
from python_speech_features import fbank

res  = []
enroll_x=[]
enroll_y=[]   

def normalize_frames(m,Scale=True):
  print(m.shape)
  if Scale:
      return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
  else:
      return (m - np.mean(m, axis=0))

def get_feature(path, model, flag):
    if flag:
        for fname in path:
            struct  =  fname.split('/')
            # print(struct[-2],struct[-1])
            speaker  = struct[-2]
            n_digits = struct[-1]
            enroll_y.append(speaker)
            audio, sr = librosa.load(fname, sr=c.SAMPLE_RATE, mono=True)
            filter_banks, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.NFILT, winlen=0.025)
            filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
            feature = normalize_frames(filter_banks, Scale=False)
            enroll_x.append(feature)
        return enroll_x, enroll_y
    else:
        audio, sr = librosa.load(path, sr=c.SAMPLE_RATE, mono=True)
        filter_banks, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.NFILT, winlen=0.025)
        filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
        feature = normalize_frames(filter_banks, Scale=False)
        feature = np.expand_dims(feature, -1)#
        feature = np.expand_dims(feature,0)
        print('feature-shape:',feature.shape)
        tmp_input = model.predict(feature)#
        return tmp_input
      
def dict_class(name):
	num_wav = {
	'Nelson_Mandela': 1500,
	'Magaret_Tarcher': 1500,
	'Benjamin_Netanyau': 1500,
	'Jens_Stoltenberg': 1500,
	'Julia_Gillard': 1501
	}
	return num_wav[name]

def get_wav_paths(audio_path, speaker):
    speaker_path = audio_path +'/'+ speaker
    all_paths = [item for item in os.listdir(speaker_path)]
    return all_paths

def load_wav(audio_path, speaker):
    wave_path = get_wav_paths(audio_path, speaker)
    for idx in (wave_path):
        res.append(audio_path+'/'+speaker+'/'+idx)
    return res
