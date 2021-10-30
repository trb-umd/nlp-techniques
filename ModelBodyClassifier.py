from mlinsights.plotting import pipeline2dot
import os
import pydot
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold


def model_body(model, data, tfidf_vect, directory):
    if not os.path.isdir("./metrics"):
        os.mkdir("./metrics")

    if not os.path.isdir(f"./metrics/{directory}"):
        os.mkdir(f"./metrics/{directory}")

    CVmodels = pd.DataFrame(columns=["accuracy", "model"])

    x = data['Mission Statement']
    y = data['Mindset']

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

    minaccuracy = 1

    for train_index, test_index in kf.split(x):
        x_train = x.iloc[train_index]
        x_trainF = tfidf_vect.transform(x_train)
        x_test = x.iloc[test_index]
        x_testF = tfidf_vect.transform(x_test)

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = model.fit(x_trainF, y_train)

        # get predictions based on x_testf
        rand_pred = model.predict(x_testF)

        # calculate r^2 error and capture other metrics
        accuracy = metrics.accuracy_score(y_test, rand_pred)

        # save the metrics and classification report for the worst case in cross validation

        if accuracy < minaccuracy:
            rf_result = pd.crosstab(y_test, rand_pred, rownames=['Actual Result'], colnames=['Predicted Result'])
            report = classification_report(y_test, rand_pred)
            minaccuracy = accuracy

        # Save model values
        CVmodels.loc[i] = [accuracy, model]

        # increment index in dataframe
        i = i + 1

    lowVal = CVmodels.accuracy.min()
    highVal = CVmodels.accuracy.max()
    meanVal = CVmodels.accuracy.mean()
    sdev = CVmodels.accuracy.std()

    print(f"cross validation metrics, worst case accuracy = {lowVal}, average = {meanVal}, best case = {highVal}, "
          f"all +-{2 * sdev / 10 ** (1 / 2)} @ 95%")
    print("\n", rf_result)
    print("\n", report)

    file = open(f"./metrics/{directory}/summary.txt", "w")

    file.write(f"cross validation metrics, worst case accuracy = {lowVal}, average = {meanVal}, best case = {highVal}, "
               f"all +-{2 * sdev / 10 ** (1 / 2)} @ 95%")
    file.write("\n")
    file.write(str(rf_result))
    file.write("\n")
    file.write(report)

    file.close()
