import ModelBodyClassifier

from sklearn import svm
from sklearn.pipeline import make_pipeline

def apply_SVC(data, tfidf_vect):

    model = make_pipeline(svm.SVC(kernel='linear'))

    directory = "SVC"

    ModelBodyClassifier.model_body(model, data, tfidf_vect, directory)
