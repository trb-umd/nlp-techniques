
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import GlobalMaxPooling1D, Dropout, Activation
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import datetime


def apply_cnn(data, embedding_dims, tokenizer):

    # set up model parameters
    batch_size = 32
    filters = 250  # number of filters in convolutional network
    kernel_size = 3  # a window size of 3 tokens
    hidden_dims = 250  # number of neurons at the normal feedforward NN
    epochs = 50
    maxlen = 4500  # the size of our mission statements

    def create_model():
        model = Sequential()
        model.add(layers.Embedding(
            vocab_size, embedding_dims, input_length=maxlen))
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

    def create_modelLSTM():
        model = Sequential()
        model.add(layers.Embedding(
            vocab_size, embedding_dims, input_length=maxlen))
        model.add(Conv1D(128, 5, padding='same', activation='relu'))
        model.add(MaxPooling1D())
        model.add(LSTM(100))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

    def create_modelv2():  # Adding a fully connected layer and dropout to see if it helps overfitting
        model = Sequential()
        model.add(layers.Embedding(
            vocab_size, embedding_dims, input_length=maxlen))
        model.add(Conv1D(filters, kernel_size, padding='valid',
                         activation='relu', strides=10))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('relu'))
        model.compile(optimizer='adam', loss='mean_squared_error',
                      metrics=['mean_squared_error'])

        return model

######################################################################
# Main function - setting up data and metric collection
#               - running model through K-Folds to reduce overfitting
#
    CVmodels = pd.DataFrame(
        columns=["r2_score", "mae", "mse", "rmse", "data_type", "iter"])
    i = 0
    iter = 0
    x = data['Mission Statement']
    y = data['xOTIOverall']

    vocab_size = len(tokenizer.word_index) + 1
    ##
    # TIM: Can we make this a switch so he doesn't have to change code to pick a different model?
    ##
    model = create_modelLSTM()

    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(x):

        x_train = x.iloc[train_index]
        X_train = tokenizer.texts_to_sequences(x_train)
        # Pad with 0 to get equal length
        X_train_pad = pad_sequences(X_train, maxlen=maxlen, value=0.0)

        x_test = x.iloc[test_index]
        X_test = tokenizer.texts_to_sequences(x_test)
        # Pad with 0's to get equal length
        X_test_pad = pad_sequences(X_test, maxlen=maxlen, value=0.0)

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        history = model.fit(X_train_pad, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(X_test_pad, y_test))

        # Get model metrics with TRAINING data

        predictions = model.predict(X_train_pad)

        # Save off ground truth and prediction for TRAIN into a file for each iteration

        ts = datetime.datetime.now().timestamp()

        fn = "TrainIter" + str(iter) + "-" + str(ts) + ".csv"
        f = open(fn, "w")
        for j in range(len(y_train)):
            line = str(y_train.iloc[j]) + ", " + str(predictions[j][0]) + "\n"
            f.writelines(line)
        f.close()

        # calculate r^2 error and capture other metrics

        error = r2_score(y_train, predictions)
        mae = metrics.mean_absolute_error(y_train, predictions)
        mse = metrics.mean_squared_error(y_train, predictions)
        rmse = np.sqrt(mse)

        # Save model values

        CVmodels.loc[i] = [error, mae, mse, rmse, "train", iter]

        # increment index in dataframe
        i += 1

        # Now run the predictions on the test data

        predictions = model.predict(X_test_pad)

        # Save off ground truth and prediction for TEST into a file for each iteration
        fn = "TestIter" + str(iter) + "-" + str(ts) + ".csv"
        f = open(fn, "w")
        for k in range(len(y_test)):
            line = str(y_test.iloc[k]) + ", " + str(predictions[k][0]) + "\n"
            f.writelines(line)
        f.close()

        # calculate r^2 error and capture other metrics

        error = r2_score(y_test, predictions)
        mae = metrics.mean_absolute_error(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Save model values

        CVmodels.loc[i] = [error, mae, mse, rmse, "test", iter]

        # increment index in dataframe
        i += 1
        # incredment the k/fold iteration
        iter += 1
    #
    # Write model metrics to a file for evaluation
    #
    fn = "NNModelData" + "-" + str(ts) + ".csv"
    f = open(fn, "w")
    f.writelines(CVmodels.to_string())
    print("Neural Network metrics")
    print(CVmodels.to_string())
    f.close()
