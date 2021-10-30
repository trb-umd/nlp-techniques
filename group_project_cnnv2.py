# project module imports

#from SentimentAnalysis import analyze_sentiment
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
import numpy as np
import DataPrep
import LDAModel
import LinearRegressionModel
import RandomForestRegressorModel
import RandomForestClassifierModel
import SVCModel
import SVMModel
import WordGroup
import nlp_cnn_test
from sklearn.feature_extraction.text import TfidfVectorizer


embedding_dims = 50  # for selected GloVe file

##################################################################
# Call module to import data and perform pre-processing including
# NLP prep
##
data, data_nonvec, wc_data, tm_data, stop_words = DataPrep.data_prep()

# create grouping of words by mindset
fixed_data, middle_data, growth_data, fx_words, md_words, gr_words, fx_id2word, md_id2word, gr_id2word, \
    fx_corpus, md_corpus, gr_corpus = WordGroup.group_words(wc_data)
#
# Create the embedding matrix to represent the words as values using GloVe
#


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    f = open(filepath, encoding="utf8")
    for line in f:
        word, *vector = line.split()
        if word in word_index:
            idx = word_index[word]
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                :embedding_dim]
    return embedding_matrix

# separate into X and Y


x_all = data['Mission Statement']
y_all = data['xOTIOverall']

# TFIDF for std regression

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(data["Mission Statement"])

# Prepare the tokenizer to be used by our NN models
#
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_all)
embedding_matrix = create_embedding_matrix(
    'glove.6B.50d.txt', tokenizer.word_index, embedding_dims)


# Sentiment Analysis with BERT
#sentiment = analyze_sentiment(data_nonvec)
# CNN
nlp_test = nlp_cnn_test.apply_cnn(data, embedding_dims, tokenizer)

# Linear regression
#linear_regression = LinearRegressionModel.apply_regression(data, tfidf_vect)

# SVM
#svm_model = SVMModel.apply_SVM(data, tfidf_vect)

# Random Forest Regressor
# random_regressor = RandomForestRegressorModel.apply_RandomForestRegressor(
#    data, tfidf_vect)

# Random Forest Classifier
# random_classifier = RandomForestClassifierModel.apply_RandomForestClassifier(
#  data, tfidf_vect)

# SVC
#svc_model = SVCModel.apply_SVC(data, tfidf_vect)

# LDA for fixed
#LDAModel.apply_lda(fixed_data, fx_words, fx_id2word, fx_corpus, stop_words)

# LDA for middle
#LDAModel.apply_lda(middle_data, md_words, md_id2word, md_corpus, stop_words)

# LSA for growth
#LDAModel.apply_lda(growth_data, gr_words, gr_id2word, gr_corpus, stop_words)
