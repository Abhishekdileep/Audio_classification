

# Build the network arch 
#compile network 
# train network 

import json
from sklearn import metrics
import sklearn.model_selection as model_sel
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 
from tensorflow.keras import layers

DATASET_PATH = "data.json"

# load dataset 
def load_data(dataset_path):
    with open(dataset_path , "r") as fp:
        data = json.load(fp)
    
    input = np.array(data['mfcc'])
    targets = np.array(data['label'])

    return input, targets

#split the data into train and test set
def train_test_split( x , y , sample_size = 0.3):
    return model_sel.train_test_split( x , y , test_size = sample_size)

def model_gen() :
    return keras.Sequential(
    [
        layers.Flatten(input_shape = (input.shape[1] , input.shape[2])),
        layers.Dense(512, activation="relu", name="layer1" , kernel_regularizer = keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu", name="layer2" , kernel_regularizer = keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", name="layer3" , kernel_regularizer = keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax", name="layer4")
    ]
    )


def plot_history(history):

    fig , axis = plt.subplots(2)


    axis[0].plot(history.history['accuracy'] , label = "train accuracy")
    axis[0].plot(history.history['val_accuracy'] , label = "test accuracy")

    axis[0].set_ylabel("accuray")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")


    axis[1].plot(history.history['loss'] , label = "train loss")
    axis[1].plot(history.history['val_loss'] , label = "test loss")

    axis[1].set_ylabel("loss")
    axis[1].set_ylabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("loss eval")

    plt.show() 

if __name__ == "__main__":
    input , target = load_data(dataset_path=DATASET_PATH)

    x_train , x_test , y_train , y_test = train_test_split(input , target )

    model = model_gen()

    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    
    model.compile(
        optimizer = optimizer , 
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
     )

    model.summary()

    history = model.fit(x_train , y_train , validation_data = (x_test , y_test ) ,epochs = 50 , batch_size = 32)
    plot_history(history=history)
    
   
    
    #compile network 
    optimizer = keras 
   