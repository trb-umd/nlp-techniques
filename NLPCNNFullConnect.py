import ModelBody

from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPooling1D, Dropout


def apply_cnn_fully_connected(data, embedding_dims, tokenizer):

    batch_size = 32
    filters = 250  # number of filters in convolutional network
    kernel_size = 3  # a window size of 3 tokens
    hidden_dims = 250  # number of neurons at the normal feedforward NN
    epochs = 50
    max_len = 4500  # the size of our mission statements
    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dims, input_length=max_len))
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

    directory = "cnn-full-connect"

    ModelBody.model_body(data, model, tokenizer, max_len, batch_size, epochs, directory)
