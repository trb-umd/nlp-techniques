
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalMaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import gensim

# hyper parameters


def apply_cnn(data):

    # hyper parameters
    embedding_dims = 50  # for GlOVE
    batch_size = 32
    # embedding_dims = 300  # Length of the token vectors
    filters = 250  # number of filters in your Convnet
    kernel_size = 3  # a window size of 3 tokens
    hidden_dims = 250  # number of neurons at the normal feedforward NN
    epochs = 5
    maxlen = 4500

    def create_embedding_matrix(filepath, word_index, embedding_dim):
        vocab_size = len(word_index) + 1
        # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        f = open(filepath, encoding="utf8")
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                print("TRUE")
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                    :embedding_dim]
        return embedding_matrix

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

    CVmodels = pd.DataFrame(
        columns=["r2_score", "mae", "mse", "rmse", "model"])
    i = 0
    x = data['Mission Statement']
    y = data['xOTIOverall']

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(x)
    # Adding 1 because of  reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = create_embedding_matrix(
        'glove.6B.50d.txt', tokenizer.word_index, embedding_dims)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    model = create_model()

    for train_index, test_index in kf.split(x):

        x_train = x.iloc[train_index]
        X_train = tokenizer.texts_to_sequences(x_train)
        # Pad with 0 to get equal length
        X_train_pad = pad_sequences(X_train, maxlen=maxlen, value=0.0)

        x_test = x.iloc[test_index]
        X_test = tokenizer.texts_to_sequences(x_test)
        # Pad with 0 to get equal length
        X_test_pad = pad_sequences(X_test, maxlen=maxlen, value=0.0)

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # model = model.fit(x_trainF, y_train)
        history = model.fit(X_train_pad, y_train, batch_size=batch_size,
                            epochs=epochs, validation_data=(X_test_pad, y_test))

        predictions = model.predict(X_test_pad)

        # calculate r^2 error and capture other metrics

        error = r2_score(y_test, predictions)
        mae = metrics.mean_absolute_error(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Save model values

        CVmodels.loc[i] = [error, mae, mse, rmse, model]

        # increment index in dataframe
        i = i + 1

    # get the minimum model with least score
    maxVal = CVmodels.mse.min()

    print("\nCNN:\n")
    print("MinVal=", maxVal)
    print('Mean Absolute Error:',
          CVmodels.loc[CVmodels.mse == maxVal].mae.values[0])
    print('Mean Squared Error:', maxVal)
    print('Root Mean Squared Error:',
          CVmodels.loc[CVmodels.mse == maxVal].rmse.values[0])
