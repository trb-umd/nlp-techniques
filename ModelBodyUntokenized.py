from mlinsights.plotting import pipeline2dot
import numpy as np
import os
import pandas as pd
import pydot
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def model_body(model, data, tfidf_vect, directory):

    if not os.path.isdir("./metrics"):
        os.mkdir("./metrics")

    if not os.path.isdir(f"./metrics/{directory}"):
        os.mkdir(f"./metrics/{directory}")

    CVmodels = pd.DataFrame(columns=["r2_score", "CoD_val", "mae", "mse", "rmse", "model"])

    x = data['Mission Statement']
    y = data['xOTIOverall']

    # make graph of pipeline

    x_input = pd.DataFrame(x)

    dot = pipeline2dot(model, x_input)

    dot_file = f"./metrics/{directory}/graph.dot"
    with open(dot_file, "w", encoding="utf-8") as f:
        f.write(dot)

    (graph,) = pydot.graph_from_dot_file(f"./metrics/{directory}/graph.dot")
    graph.write_png(f"./metrics/{directory}/model-graph.png")

    # loop over all possible models
    # shuffle reorders the data for less bias

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    i = 0

    for train_index, test_index in kf.split(x):
        x_train = x.iloc[train_index]
        x_trainF = tfidf_vect.transform(x_train)
        x_test = x.iloc[test_index]
        x_testF = tfidf_vect.transform(x_test)

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = model.fit(x_trainF, y_train)

        # get predictions based on x_testf

        predictions = model.predict(x_testF)

        # calculate r^2 error and capture other metrics

        error = r2_score(y_test, predictions)
        CoD_Val = model.score(x_testF, y_test)
        mae = metrics.mean_absolute_error(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Save model values

        CVmodels.loc[i] = [error, CoD_Val, mae, mse, rmse, model]

        # increment index in dataframe
        i = i + 1

    return CVmodels

