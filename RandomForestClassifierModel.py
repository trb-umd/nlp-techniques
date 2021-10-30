import ModelBodyClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


def apply_RandomForestClassifier(data, tfidf_vect):

    model = make_pipeline(RandomForestClassifier(n_jobs=2, random_state=0))

    directory = "random-forest-classifier"

    ModelBodyClassifier.model_body(model, data, tfidf_vect, directory)


