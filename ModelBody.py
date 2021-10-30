import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def model_body(data, model, tokenizer, max_len, batch_size, epochs, directory):

    if not os.path.isdir("./metrics"):
        os.mkdir("./metrics")

    if not os.path.isdir(f"./metrics/{directory}"):
        os.mkdir(f"./metrics/{directory}")

    if not os.path.isdir(f"./metrics/{directory}/outputs"):
        os.mkdir(f"./metrics/{directory}/outputs")

    model.summary()

    with open(f"./metrics/{directory}/summary.txt", "w") as model_file:

        model.summary(print_fn=lambda x: model_file.write(x + '\n'))

    model_file.close()

    plot_model(model, to_file=f"./metrics/{directory}/model_plot.png", show_shapes=True, show_layer_names=True)

    callback = EarlyStopping(monitor="val_loss",
                             min_delta=0,
                             patience=5,
                             verbose=0,
                             mode="auto",
                             baseline=None,
                             restore_best_weights=False)

    CVmodels = pd.DataFrame(
        columns=["r2_score", "mae", "mse", "rmse", "data_type", "iter"])

    i = 0

    iter = 0

    x = data['Mission Statement']
    y = data['xOTIOverall']

    x_input = pd.DataFrame(x)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(x):

        x_train = x.iloc[train_index]
        X_train = tokenizer.texts_to_sequences(x_train)
        X_train_pad = pad_sequences(X_train, maxlen=max_len, value=0.0)

        x_test = x.iloc[test_index]
        X_test = tokenizer.texts_to_sequences(x_test)
        X_test_pad = pad_sequences(X_test, maxlen=max_len, value=0.0)

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test_pad, y_test),
                  callbacks=[callback])

        # Get model metrics with TRAINING data
        predictions = model.predict(X_train_pad)

        # Save off ground truth and prediction for TRAIN into a file for each iteration
        ts = datetime.datetime.now().timestamp()

        fn = f"./metrics/{directory}/outputs/TrainIter" + \
            str(iter) + "-" + str(ts) + ".csv"

        f = open(fn, "w")

        for j in range(len(y_train)):

            line = str(y_train.iloc[j]) + ", " + str(predictions[j][0]) + "\n"
            f.writelines(line)

        f.close()

        # calculate r^2 error and capture other metrics
        error = r2_score(y_train, predictions)
        mae = metrics.mean_absolute_error(y_train, predictions)
        mse = metrics.mean_squared_error(y_train, predictions)
        rmse = np.sqrt(mse)

        # Save model values
        CVmodels.loc[i] = [error, mae, mse, rmse, "train", iter]

        # increment index in dataframe
        i += 1

        # Now run the predictions on the test data
        predictions = model.predict(X_test_pad)

        # Save off ground truth and prediction for TEST into a file for each iteration
        fn = f"./metrics/{directory}/outputs/TestIter" + \
            str(iter) + "-" + str(ts) + ".csv"

        f = open(fn, "w")

        for k in range(len(y_test)):

            line = str(y_test.iloc[k]) + ", " + str(predictions[k][0]) + "\n"
            f.writelines(line)

        f.close()

        # calculate r^2 error and capture other metrics
        error = r2_score(y_test, predictions)
        mae = metrics.mean_absolute_error(y_test, predictions)
        mse = metrics.mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Save model values
        CVmodels.loc[i] = [error, mae, mse, rmse, "test", iter]

        # increment index in dataframe
        i += 1

        # increment the k/fold iteration
        iter += 1

        max_val = CVmodels.mse.min()

        print("\nR2 Score=", error)
        print('Mean Absolute Error:',
              CVmodels.loc[CVmodels.mse == max_val].mae.values[0])
        print('Mean Squared Error:', max_val)
        print('Root Mean Squared Error:',
              CVmodels.loc[CVmodels.mse == max_val].rmse.values[0])
        print('Iteration:',
              CVmodels.loc[CVmodels.mse == max_val].iter.values[0])
        print('Type:', CVmodels.loc[CVmodels.mse ==
                                    max_val].data_type.values[0])
        print("\n")

    # Write model metrics to a file for evaluation
    fn = f"./metrics/{directory}/ModelData" + "-" + str(ts) + ".csv"
    f = open(fn, "w")
    f.writelines(CVmodels.to_string())
    f.close()
