import os
import shutil
import pandas as pd
import librosa
import numpy as np
import config as c
from pathlib import Path
from python_speech_features import fbank
from transforms import load_wav, normalize_frames


def data_info(tot_wav):
    df_data = pd.DataFrame()
    df_data['filename'] = tot_wav
    df_data['speaker'] = df_data['filename'].apply(lambda x: x.split('/')[-2])
    num_speakers = len(set(df_data['speaker']))
    print(f'files: {len(df_data)}, speakers:{num_speakers}')
    return df_data, num_speakers  


if __name__ == "__main__":
    
    audio_folder = "audio"
    noise_folder = "noise"
    audio_path = os.path.join(c.ORIGINAL_DATA_DIR, audio_folder)
    noise_path = os.path.join(c.ORIGINAL_DATA_DIR, noise_folder)

    # arrange audio and noise
    for folder in os.listdir(c.ORIGINAL_DATA_DIR):
        if os.path.isdir(os.path.join(c.ORIGINAL_DATA_DIR, folder)):
            if folder in [audio_folder, noise_folder]:
                continue
            elif folder in ["other", "_background_noise_"]:
                shutil.move(
                    os.path.join(c.ORIGINAL_DATA_DIR, folder),
                    os.path.join(noise_path, folder),
                )
            else:
                shutil.move(
                    os.path.join(c.ORIGINAL_DATA_DIR, folder),
                    os.path.join(audio_path, folder),
                )
    #Get the list of all noise files
    noise_paths = []
    for subdir in os.listdir(noise_path):
        subdir_path = Path(noise_path) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]
    print(f"noise_paths:{noise_paths},audio_path:{audio_path}")

    res  = []
    wav_label = {}

    for i in c.SPK_LIST:
        res = load_wav(audio_path,i)    
    df_data, num_speakers = data_info(res)

    embedding_X=[]
    class_Y=[]

    #embedding .npy    
    for fname in df_data['filename']:
        print(fname)
        struct  =  fname.split('/')
        # print(struct[-2],struct[-1])
        speaker  = struct[-2]
        n_digits = struct[-1]

        class_Y.append(speaker)
        print(speaker)

        audio, sr = librosa.load(fname, sr=c.SAMPLE_RATE, mono=True)
        print(audio.shape)
        filter_banks, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.NFILT, winlen=0.025)
        filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
        feature = normalize_frames(filter_banks, Scale=False)
        embedding_X.append(feature)
  
    if not os.path.exists(c.PREPROCESS_DIR):
        os.makedirs(c.PREPROCESS_DIR)
    np.save(c.PREPROCESS_DIR+'embedding_X_save', embedding_X) # x_save.npy
    np.save(c.PREPROCESS_DIR+'class_Y_save', class_Y) # x_save.npy
