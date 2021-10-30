import ModelBodyUntokenized

import datetime
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

def apply_SVM(data, tfidf_vect):

    model = make_pipeline(SVR(kernel='rbf'))

    directory = "SVM"

    CVmodels = ModelBodyUntokenized.model_body(model, data, tfidf_vect, directory)

    maxVal = CVmodels.mse.min()

    print("MaxVal=", maxVal)
    print('Coefficient of Determination:', CVmodels.loc[CVmodels.mse == maxVal].CoD_val.values[0])
    print('Mean Absolute Error:', CVmodels.loc[CVmodels.mse == maxVal].mae.values[0])
    print('Mean Squared Error:', maxVal)
    print('Root Mean Squared Error:', CVmodels.loc[CVmodels.mse == maxVal].rmse.values[0])

    ts = datetime.datetime.now().timestamp()

    # Write model metrics to a file for evaluation
    fn = f"./metrics/{directory}/ModelData" + "-" + str(ts) + ".csv"
    f = open(fn, "w")
    f.writelines(CVmodels.to_string())
    f.close()

