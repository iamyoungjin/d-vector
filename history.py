import matplotlib.pyplot as plt

def train_history(history):
    plt.plot(history.history['accuracy'],label='Train_acc')
    plt.plot(history.history['val_accuracy'],label='val_acc')
    plt.xlabel('Epochs')
    plt.xlabel('Accuracy')
    plt.legend()
    plt.show