import json 
import numpy as np
from sklearn import metrics 
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras 
from preprocess import save_mfcc

DATA_PATH = 'data.json'

def load_data(data_path):


    with open(data_path , "r" ) as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"])
    y = np.array(data['label'])
    return X,y 

def prepare_dataset(test_size , validation_size):

    X, y = load_data(DATA_PATH)

    X_train , X_test , Y_train , Y_test = train_test_split(X , y , test_size=test_size)

    X_train , X_validation , Y_train , Y_validation = train_test_split( X_train , Y_train , test_size=validation_size)

    # Only for CNN , we don't need it for RNN 
    # X_train = X_train[... , np.newaxis]
    # X_test = X_test[... , np.newaxis]
    # X_validation = X_validation[... , np.newaxis]

    return X_train , X_validation , X_test , Y_train , Y_validation , Y_test


def build_model(input_shape):

    model = keras.Sequential()

    #layer conv2d - 1
    # model.add(keras.layers.Conv2D(32 , (3 , 3 ) ,activation = 'relu' , input_shape = input_shape ) )
    
    # model.add(keras.layers.MaxPool2D((3,3) , strides=(2,2) , padding='same') )

    # model.add(keras.layers.BatchNormalization())

    # #layer conv2d -2 
    # model.add(keras.layers.Conv2D(32 , (2 , 2) ,activation = 'relu' , input_shape = input_shape ) )
    
    # model.add(keras.layers.MaxPool2D((3,3) , strides=(2,2) , padding='same') )

    # model.add(keras.layers.BatchNormalization())

    # #layer conv2d -3 
    # model.add(keras.layers.Conv2D(32 , (2 , 2) ,activation = 'relu' , input_shape = input_shape ) )
    
    # model.add(keras.layers.MaxPool2D((3,3) , strides=(2,2) , padding='same') )

    # model.add(keras.layers.BatchNormalization())

    # #flatten the output and feed it into dense layers 
    # model.add(keras.layers.Flatten())

    # model.add(keras.layers.Dense(64 , activation='relu' ))

    # model.add(keras.layers.Dropout(0.3)) 


    # LSTM Layers 
    model.add(keras.layers.LSTM(64 , input_shape=input_shape , return_sequences=True))
    model.add(keras.layers.LSTM(64))

    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer 

    model.add(keras.layers.Dense(10 , activation='softmax'))

    return model


def predict(X , y , model ):
    
    X = X [... , np.newaxis]
    Prediction = model.predict(X)

    #extract index with max

    predict_val = np.argmax(Prediction , axis=1)
    print("Expected index {} , predicted {} ".format(y , predict_val))



if __name__ == "__main__":
    
    # Train test split 
    
    X_train , X_validation , X_test , Y_train , Y_validation , Y_test = prepare_dataset(0.25 , 0.2)

    #Build the CNN Model 
    Input_shape = (X_train.shape[1] , X_train.shape[2]  )

    model = build_model(Input_shape)

    # Compile the model 
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(  optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy']
        )

    #Train CNN model 
    model.fit(X_train , Y_train , validation_data=(X_validation , Y_validation) , batch_size = 32 , epochs=30)


    #evaluate the CNN on the test set 

    test_error , test_acc = model.evaluate(X_test , Y_test ,verbose=1)

    print("Accuracy on test set is : {}".format(test_acc))

    # predict
    save_mfcc(dataset_path= "Data1" , json_path="data1.json")
    X,y = load_data('data1.json')
    predict(X , y , model)
