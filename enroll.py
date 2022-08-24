import os
import librosa
import torch
import numpy as np
import config as c
from preprocess import data_info, load_wav
from tensorflow.keras.models import load_model
from python_speech_features import fbank
from transforms import normalize_frames, get_feature, dict_class, get_wav_paths, load_wav

if __name__ == "__main__":
  model= load_model(c.MODEL_PATH+'embd_model.h5')

  audio_folder = "audio"
  tot_wav = []
  audio_path = os.path.join(c.ORIGINAL_DATA_DIR, audio_folder)
  res  = []
  wav_label = {}

  for i in c.ENROLL_SPK_LIST: 
    res = load_wav(audio_path,i)
  df_data, num_speakers = data_info(res)
 
  embeddings={} 
  embedding_dir = c.EMBEDDING_DIR

  enroll_speaker_list = c.ENROLL_SPK_LIST 
  cnt = 0
  
  for i in range(len(df_data['filename'][:4500])):  #register 3 person
    filename = df_data['filename'][i]
    spk = df_data['speaker'][i]
    activation = get_feature(filename, model, False)
    if spk in embeddings:
        embeddings[spk] += [activation]
    else:
        embeddings[spk] = [activation]
    cnt+=1
    print(cnt)
    print("Aggregates the activation (spk : %s)" % (spk))
    
  if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)

  embedding_path = os.path.join(embedding_dir,'embedding.pth')
  torch.save(embeddings, embedding_path)
    
  for spk_index in enroll_speaker_list:
    embedding_path = os.path.join(embedding_dir, spk_index+'.pth')
    torch.save(embeddings[spk_index], embedding_path)
    print("Save the embeddings for %s" % (spk_index))
