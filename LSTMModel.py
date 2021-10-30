import ModelBody

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import layers


def apply_lstm(data, tokenizer):

    # hyper parameters
    embedding_dims = 50  # for GlOVE
    batch_size = 32
    epochs = 50
    max_len = 4500
    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dims, input_length=max_len))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(LSTM(100))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    directory = "LSTM"

    ModelBody.model_body(data, model, tokenizer, max_len, batch_size, epochs, directory)
