# project module imports

import DataPrep
import LinearRegressionModel
import LSTMModel
import RandomForestRegressorModel
import RandomForestClassifierModel
import SVCModel
import SVMModel
import WordGroup
import SentimentAnalysis
import BertModel
import NLPCNNBase
import EmbeddingMatrix
import NLPCNNFullConnect

# python module imports

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer

# get data and stopwords

data, wc_data, tm_data, stop_words = DataPrep.data_prep()

embedding_dims = 50  # for selected GloVe file

x_all = data['Mission Statement']
y_all = data['xOTIOverall']

# TFIDF

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(data["Mission Statement"])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_all)
embedding_matrix = EmbeddingMatrix.create_embedding_matrix(
    'glove.6B.50d.txt', tokenizer.word_index, embedding_dims)

# create grouping of words by mindset
fixed_data, middle_data, growth_data, fx_words, md_words, gr_words, fx_id2word, md_id2word, gr_id2word, \
           fx_corpus, md_corpus, gr_corpus = WordGroup.group_words(wc_data)

# separate into X and Y

x_all = data['Mission Statement']
x_all = tfidf_vect.transform(x_all)
y_all = data['xOTIOverall']

###############################

# standard regression models

# Linear regression
#LinearRegressionModel.apply_regression(data, tfidf_vect)

#Random Forest Classifier
#RandomForestClassifierModel.apply_RandomForestClassifier(data, tfidf_vect)

#Random Forest Regressor
#RandomForestRegressorModel.apply_RandomForestRegressor(data, tfidf_vect)

#SVC
#SVCModel.apply_SVC(data, tfidf_vect)

# SVM
#SVMModel.apply_SVM(data, tfidf_vect)

######################################

# Sentiment analysis model

#SentimentAnalysis.analyze_sentiment(data)

#######################################

# Neural networks

# BERT
BertModel.apply_bert(data)

# base CNN
#NLPCNNBase.apply_cnn(data, embedding_dims, tokenizer)

# fully connected CNN
#NLPCNNFullConnect.apply_cnn_fully_connected(data, embedding_dims, tokenizer)

# LSTM
#LSTMModel.apply_lstm(data, tokenizer)


