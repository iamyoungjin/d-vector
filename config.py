# Wave path
ORIGINAL_DATA_DIR = "/users/iamyoungjin/desktop/tech/d-vector/data/16000_pcm_speeches"

# Train and Enroll List
SPK_LIST = ['Nelson_Mandela','Magaret_Tarcher','Benjamin_Netanyau','Jens_Stoltenberg','Julia_Gillard']
ENROLL_SPK_LIST = ['Nelson_Mandela','Magaret_Tarcher','Benjamin_Netanyau']#,'Jens_Stoltenberg','Julia_Gillard']

# Test wav and label
TEST_SPK_WAV_PATH = '/users/iamyoungjin/desktop/tech/d-vector/data/test_data/'
TEST_SPK = 'Benjamin_Netanyau' 
TEST_WAV_NAME = '0.wav'

# Model path
MODEL_PATH= '/users/iamyoungjin/desktop/tech/d-vector/data/model/'

#Feature path
EMBEDDING_DIR = '/users/iamyoungjin/desktop/tech/d-vector/data/embeddings/enroll_embeddings/' 
PREPROCESS_DIR = '/users/iamyoungjin/desktop/tech/d-vector/data/preprocess_data/'

# Parameter 
SAMPLE_RATE = 16000
FILTER_BANK = 40
NFILT = 40
NUM_CLASSES = 5
INPUT_SHAPE = (99, 40, 1)
NUM_WIN_SIZE = 10 
