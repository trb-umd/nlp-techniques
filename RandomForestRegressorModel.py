import ModelBodyUntokenized

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


def apply_RandomForestRegressor(data, tfidf_vect):

    model = make_pipeline(RandomForestRegressor())

    directory = "random-forest-regressor"

    CVmodels = ModelBodyUntokenized.model_body(model, data, tfidf_vect, directory)

    # get the minimum model with least score
    max_val = CVmodels.mse.min()

    print("MaxVal=", max_val)
    print('Coefficient of Determination:', CVmodels.loc[CVmodels.mse == max_val].CoD_val.values[0])
    print('Mean Absolute Error:', CVmodels.loc[CVmodels.mse == max_val].mae.values[0])
    print('Mean Squared Error:', max_val)
    print('Root Mean Squared Error:', CVmodels.loc[CVmodels.mse == max_val].rmse.values[0])

    file = open(f"./metrics/{directory}/summary.txt", "w")

    file.write(f"MaxVal: {max_val}")
    file.write("\n")
    file.write(f"Coefficient of Determination: {CVmodels.loc[CVmodels.mse == max_val].CoD_val.values[0]}")
    file.write("\n")
    file.write(f"Mean Absolute Error: {CVmodels.loc[CVmodels.mse == max_val].mae.values[0]}")
    file.write("\n")
    file.write(f"Mean Squared Error: {max_val}")
    file.write("\n")
    file.write(f"Root Mean Squared Error: {CVmodels.loc[CVmodels.mse == max_val].rmse.values[0]}")

    file.close()

