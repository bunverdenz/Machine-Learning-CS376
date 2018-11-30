from datetime import datetime
import numpy as np
import pandas as pd
import sys
import time

# Scikit-learn
import sklearn
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import from local files
from preprocess import DataFrameSelector

# Path for saving data
PATH_DATA           = "../data/"
PATH_DATA_CLEAN     = "../data_clean/"
PATH_DATA_PREPARE   = "../data_prepared/"
PATH_MODEL_SCIKIT   = "../model/scikit/saved_model/"
SCIKIT_MODEL_PATH   = "../model/scikit/saved_scikit_model/"

# What library to use
MODEL               = ["scikit"]  # Choose between "scikit" and "tensorflow"

# Scikit [mlp, gbr]
MODEL_SCIKIT        = "mlp"

# TODO: Please change this as need
# Flags
CLEAN_DATA          = False  # For cleaning data
PREPROCESS          = True  # For pre-processing data
TRAIN               = True  # For training the model (Separate train and test data)
TRAIN_FINAL         = True  # Train with the whole data
PREDICT             = True  # For file submission

# TODO: Please use your TEAM ID here
PATH_SUBMIT_FILE    = "../predict/"
TEAM                = "19"

'''
    t = time.process_time()
    # do some stuff
    elapsed_time = time.process_time() - t
    print("Elapsed time: ", elapsed_time)
'''


def main(argv):
    print("\n#################### Running ####################")
    print("Library")
    print("\tNumpy: {}".format(np.__version__))
    print("\tPandas: {}".format(pd.__version__))
    print("\tScikit-learn: {}".format(sklearn.__version__))

    ####################################################################################################################
    if CLEAN_DATA:
        # Demographic details
        if True:
            data_df = pd.read_csv("{}data_train.csv".format(PATH_DATA), index_col=False,
                                  skipinitialspace=True, parse_dates=[0,18])
            data_df.index = data_df['apt_id']
            data_df.columns = ["contract_date", "latitude", "longtitude", "altitude",
                               "1st_class_reg", "2nd_class_reg", "road_id", "apt_id",
                               "floor", "angle", "area", "limit_car_are", "total_car_area",
                               "ext_car_enter", "avg_fee", "num_house", "avg_age_ppl",
                               "builder_id", "done_date", "built_year", "num_school",
                               "num_bus", "num_subway", "label"]

            # Count days since apartment was constructed --------------------------------------------------------------
            data_df["day_diff"] = (data_df["done_date"] - data_df["contract_date"]).dt.days
            data_df.drop(["contract_date"], axis=1, inplace=True)
            data_df.drop(["done_date"], axis=1, inplace=True)
            # Calculate age --------------------------------------------------------------------------------------------
            data_df["age"] = 2018 - data_df["built_year"]
            data_df.drop(["built_year"], axis=1, inplace=True)

            # Change label ---------------------------------------------------------------------------------------------
            lb = preprocessing.LabelBinarizer()
            # 1st_class_reg, try to categorize id of 1st_class into six id (see test_data)
            temp = lb.fit_transform(data_df["1st_class_reg"].values.reshape(-1, 1))
            data_df.drop(["1st_class_reg"], axis=1, inplace=True)
            data_df = pd.concat([data_df, pd.DataFrame(temp, columns=["one", "two", "three", "four","five", "six"])], axis=1)

            # TODO: group lat,long for area that are close to each other -> calculate block by block area

            # TODO: avg of altitude then categorize altitude
            #     sa_df.loc[sa_df["tp"] == "DR"] = 0
            #     sa_df.loc[sa_df["tp"] == "CR"] = 1
            #     temp2 = sa_df.groupby(["ip_id"])["tp"].mean().reset_index()
            #     temp2.to_csv("{}avg_altitude.csv".format(PATH_DATA_CLEAN),index=False)
            # Save
            data_df.to_csv("{}data_df.csv".format(PATH_DATA_CLEAN), index=False)
            # TODO: check result
            # data_df.loc[["apt_id", "label"]].to_csv("{}y_train.csv".format(PATH_DATA), index=False)
            data_df.loc["label"].to_csv("{}y_train.csv".format(PATH_DATA), index=False)

        #   you can merge many raw data here && fill Null with 0
            data_all_df = pd.concat([data_df], axis=1, sort=False).fillna(0)
            data_all_df.to_csv("{}data_all.csv".format(PATH_DATA_CLEAN))
        #     # TODO: Don't touch here
        #     Make training data ---------------------------------------------------------------------------------------
            print("Make data for training ...")
            # This is the whole data that we have the label
            y_train_df = pd.read_csv("{}y_train.csv".format(PATH_DATA), skipinitialspace=True).set_index("apt_id")
            data_train_df = pd.concat([data_all_df.loc[data_all_df.index.isin(y_train_df.index)], y_train_df], axis=1, sort=False).reindex(y_train_df.index)
            data_train_df.to_csv("{}xy.csv".format(PATH_DATA_CLEAN), index=False)
            # Just in case
            x = data_train_df.drop("label", axis=1)
            x.to_csv("{}x.csv".format(PATH_DATA_CLEAN), index=False)
            y = data_train_df.loc[:, ["label"]]
            y.to_csv("{}y.csv".format(PATH_DATA_CLEAN), index=False)
            # Make testing data ----------------------------------------------------------------------------------------
            print("Make data for testing ...")
            id_test_df = pd.read_csv("{}y_test_index.csv".format(PATH_DATA), skipinitialspace=True)
            x_test_df = data_all_df.loc[data_all_df.index.isin(id_test_df["apt_id"])].reindex(id_test_df["apt_id"])
            x_test_df.to_csv("{}x_test.csv".format(PATH_DATA_CLEAN), index=False)


    ####################################################################################################################
    if PREPROCESS:
        # Scaling
        print("Scaling ...")
        # Read data
        x_df = pd.read_csv("{}x.csv".format(PATH_DATA_CLEAN))
        y_df = pd.read_csv("{}y.csv".format(PATH_DATA_CLEAN))
        x_test_df = pd.read_csv("{}x_test.csv".format(PATH_DATA_CLEAN))

        # TODO: categorize more later
        num_attribs = [i for i in x_df.columns.values if (i not in ["one", "two", "three", "four", "five", "six"])]

        cat_attribs = ["one", "two", "three", "four", "five", "six"]

        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('std_scaler', StandardScaler()),
        ])

        # Train data
        # Method1
        x_num_scaled = num_pipeline.fit_transform(x_df)
        x_scaled_df = pd.concat([x_df[cat_attribs], pd.DataFrame(x_num_scaled, columns=num_attribs)], axis=1, sort=False)
        xy_df = pd.concat([x_scaled_df, y_df], axis=1, sort=False)
        xy_df.to_csv("{}xy.csv".format(PATH_DATA_PREPARE), index=False)

        # Test data
        # Don't forget to transform
        x_test_num_scaled = num_pipeline.transform(x_test_df)
        x_test_scaled_df = pd.concat([x_test_df[cat_attribs], pd.DataFrame(x_test_num_scaled, columns=num_attribs)], axis=1, sort=False)
        x_test_scaled_df.to_csv("{}x_test_scaled.csv".format(PATH_DATA_PREPARE), index=False)

    else:
        print("Load the processed data ...")
        # This includes the header
        xy_df = pd.read_csv("{}xy.csv".format(PATH_DATA_PREPARE))

    # From now on, use numpy type data with the following names
    x = xy_df.drop("label", axis=1).values
    y = xy_df.loc[:, ["label"]].values
    n_features = x.shape[1]
    print("Number of features: {}".format(n_features))
    
    ####################################################################################################################
    if "scikit" in MODEL:
        print("Use: Scikit-learn")
        if TRAIN:
            print("\n#################### Training ####################")
            print("Model: {}".format(MODEL_SCIKIT))
            # Training models ------------------------------------------------------------------------------------------
            # TODO: Try more models
            if MODEL_SCIKIT == "mlp":
                hidden_layer = (200, 200, 200, 200)
                n_epochs = 100
                clf = MLPClassifier(hidden_layer_sizes=hidden_layer,
                                    activation='relu',
                                    solver='adam',
                                    alpha=1e-5,
                                    batch_size=50,
                                    learning_rate='invscaling',
                                    learning_rate_init=0.001,
                                    max_iter=n_epochs,
                                    shuffle=True,
                                    random_state=1,
                                    verbose=True,
                                    momentum=0.9,
                                    early_stopping=False)

            elif MODEL_SCIKIT == "gbr":
                params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                          'learning_rate': 0.01, 'loss': 'ls', 'verbose': True}
                clf = ensemble.GradientBoostingRegressor(**params)

            # ----------------------------------------------------------------------------------------------------------
            if TRAIN_FINAL:
                x_train = x
                y_train = y.flatten()

                n_inputs = x_train.shape[0]

                print(("\n-------------------- Final Train --------------------"
                       "\n\t# inputs = {}"                       
                       "\n\t# features = {}").format(n_inputs,
                                                     n_features, flush=True))
                clf.fit(x_train, y_train)
                joblib.dump(clf, "{}clf_final_{}.pkl".format(PATH_MODEL_SCIKIT, MODEL_SCIKIT))
                plot_conf_matrix(y_train, clf.predict_proba(x_train), "Train set")

            else:
                sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
                for [train_index, eval_index] in sss.split(x, y):
                    x_train = x[train_index]
                    y_train = y[train_index].flatten()
                    x_eval = x[eval_index]
                    y_eval = y[eval_index].flatten()
                    n_inputs = x_train.shape[0]

                    print(("\n------------------------------ Set ------------------------------"
                           "\n\t# inputs = {}"
                           "\n\t  Train label\t{{0: {}, 1: {}}}"
                           "\n\t  Eval label\t{{0: {}, 1: {}}}"
                           "\n\t# features = {}").format(n_inputs,
                                                         (y_train == 0).sum(), (y_train == 1).sum(),
                                                         (y_eval == 0).sum(), (y_eval == 1).sum(),
                                                         n_features, flush=True))

                    clf.fit(x_train, y_train)
                    print("\n-------------------- Evaluate ---------------------")
                    joblib.dump(clf, "{}clf_test_{}.pkl".format(PATH_MODEL_SCIKIT, MODEL_SCIKIT))
                    plot_conf_matrix(y_train, clf.predict_proba(x_train), str_data="Train set")
                    plot_conf_matrix(y_eval, clf.predict_proba(x_eval), str_data="Test set")

        if PREDICT:
            print("\n#################### Submission !!! ####################")
            print("Model: clf_final_{}.pkl".format(MODEL_SCIKIT))
            clf = joblib.load("{}clf_final_{}.pkl".format(PATH_MODEL_SCIKIT, MODEL_SCIKIT))
            x_test = pd.read_csv("{}x_test_scaled.csv".format(PATH_DATA_PREPARE)).values
            y_predict_prob = clf.predict_proba(x_test)
            # TODO: What's the file name???
            pd.DataFrame(y_predict_prob[:, 0]).to_csv(PATH_SUBMIT_FILE + "CS376-unique-TEAM-{}.csv".format(TEAM), index=False, header=False)

    print("\nEnd Process")


def plot_conf_matrix(y_true, y_predict, str_data):
    print("-------------------- {} --------------------".format(str_data))
    print("Actual class\t", y_true)
    y_predict_prob = y_predict[:, 1]
    y_predict = np.argmax(y_predict, axis=1)
    print("Predicted class\t", y_predict)
    print("Predicted probability\t", y_predict_prob)

    conf_mx = confusion_matrix(y_true, y_predict)
    print("Confusion matrix", conf_mx, sep='\n')

    print("\tAccuracy score\t{:.6f}".format(accuracy_score(y_true, y_predict)))
    print("\tPrecision score\t{:.6f}".format(precision_score(y_true, y_predict)))
    print("\tROC AUC\t\t\t{:.6f}".format(roc_auc_score(y_true, y_predict_prob)))


if __name__ == "__main__":
    main(sys.argv[1:])
