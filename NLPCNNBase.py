import EmbeddingMatrix
import ModelBody

from keras.models import Sequential
from keras import layers

def apply_cnn(data, embedding_dims, tokenizer):

    # model parameters
    batch_size = 32
    epochs = 50
    max_len = 4500  # the size of our mission statements
    vocab_size = len(tokenizer.word_index) + 1
    EmbeddingMatrix.create_embedding_matrix("glove.6B.50d.txt", tokenizer.word_index, embedding_dims)

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dims, input_length=max_len))
    model.add(layers.Conv1D(128, 5, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1, activation="relu"))
    model.compile(optimizer="adam", loss="mean_squared_error",
                  metrics=["mean_squared_error"])

    directory = "cnn-base"

    ModelBody.model_body(data, model, tokenizer, max_len, batch_size, epochs, directory)
