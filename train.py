import os
import numpy as np
import pandas as pd
import config as c
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import CNNModel
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger #ReduceLROnPlateau: 모델의 개선이 없을 경우, lr을 조절해 모델의 개선을 유도하는 콜백함수.

from history import train_history
    
if __name__ == "__main__":

    embedding_X = np.load(c.PREPROCESS_DIR+'embedding_X_save.npy')
    class_Y = np.load(c.PREPROCESS_DIR+'class_Y_save.npy')

    embedding_X = np.array(embedding_X)
    embedding_X = np.expand_dims(embedding_X, -1)

    class_Y = pd.DataFrame(class_Y)
    class_Y = class_Y.replace({'Nelson_Mandela':0,'Magaret_Tarcher':1,'Benjamin_Netanyau':2,'Jens_Stoltenberg':3,'Julia_Gillard':4})
    class_Y = np.array(class_Y).squeeze(1)
    class_Y = to_categorical(class_Y, num_classes=c.NUM_CLASSES)


    X_train, X_test, y_train, y_test = train_test_split(embedding_X, class_Y, test_size=0.2, random_state=1)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) #(6000, 99, 40, 1) (1501, 99, 40, 1) (6000, 5) (1501, 5)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) 

    #Model Train
    spkModel, spk = CNNModel.build(X_train[0].shape)
    sgd = SGD( lr = 0.02)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001)
    csv_logger = CSVLogger('training.log')
    spk.compile(optimizer=sgd, loss = 'categorical_crossentropy', metrics=['accuracy'])
    history = spk.fit(X_train, y_train, batch_size = 128, epochs = 20, verbose = 1, validation_data = (X_val, y_val), callbacks = [early_stopping, reduce_lr, csv_logger])

    if not os.path.exists(c.MODEL_PATH):
        os.makedirs(c.MODEL_PATH)
    spk.save(c.MODEL_PATH+'model.h5')
    spkModel.save(c.MODEL_PATH+'embd_model.h5')    
    train_history(history)



