import os
import torch
import librosa 
import config as c
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from tensorflow.keras.models import load_model
from transforms import get_feature


def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def compare_identification(model, embeddings, test_filename, spk_list):
  # print(embeddings.keys())
  cnt = 0
  test_embedding = get_feature(test_filename, model, False)
  max_score = -1e9
  best_spk = None
  spk_list =  c.ENROLL_SPK_LIST
  for spk in spk_list:
    for idx in range(len(embeddings[spk])):
      df_test_embedding = pd.DataFrame(test_embedding).astype(float)
      embeddings[spk][idx] = pd.DataFrame(embeddings[spk][idx]).astype(float)

      df_test_embedding  =np.asarray(df_test_embedding).astype(np.float64)
      df_test_embedding = np.squeeze(np.asarray(df_test_embedding))
      
      embeddings[spk][idx]  =np.asarray(embeddings[spk][idx]).astype(np.float64)
      embeddings[spk][idx] = np.squeeze(np.asarray(embeddings[spk][idx]))      
      
      score = cos_sim(df_test_embedding, embeddings[spk][idx])
      cnt+=1
      print(cnt)
      print(f'score:{score}')
      if score > max_score:
          max_score = score
          best_spk = spk
  print(f'max_score:{max_score}')
  print(f"result:{best_spk}")
  true_spk = c.TEST_SPK
  print("=== Result ===")
  print(f"True speaker : {true_spk}\nPredicted speaker : {best_spk}")
  return best_spk


if __name__ == "__main__":

  test_dir = c.TEST_SPK_WAV_PATH + c.TEST_WAV_NAME #0.wav'
  model= load_model(c.MODEL_PATH+'embd_model.h5')
  embeddings = torch.load(c.EMBEDDING_DIR+'embedding.pth')
  
  best_spk = compare_identification(model, embeddings, test_dir,  c.ENROLL_SPK_LIST)