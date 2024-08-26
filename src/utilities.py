"""
Utilities for Machine Learning Model Training and Evaluation

This module provides utility functions and
imports necessary for preprocessing data,
training machine learning models,
and evaluating their performance.
It includes functions for splitting datasets,
generating reports, and handling various machine learning tasks.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

# preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from transformers import SavitzkyGolay


def sample_split_report(y_hd_train, y_th_train, y_hd_test, y_th_test, file: str):
    """
    Generate and save a report on the distribution of
    training and testing samples.

    This function calculates the value counts for
    the training and testing datasets
    for both head and thorax samples. It then creates a
    DataFrame to display these counts
    and saves the DataFrame to an Excel file.

    Parameters
    ----------
    y_hd_train : array-like
        Target vector for head samples in the training set.
    y_th_train : array-like
        Target vector for thorax samples in the training set.
    y_hd_test : array-like
        Target vector for head samples in the testing set.
    y_th_test : array-like
        Target vector for thorax samples in the testing set.
    file : str
        Filename to save the Excel report.

    Returns
    -------
    None
    """

    shape_data = {
        "train_head": y_hd_train.value_counts(),
        "train_thorax": y_th_train.value_counts(),
        "test_head": y_hd_test.value_counts(),
        "test_thorax": y_th_test.value_counts(),
    }

    shape_data_df = pd.DataFrame(shape_data)

    print(shape_data_df)
    shape_data_df.to_excel(
        "../results/tables/wholespectra_results/train_test_shape_" + file + ".xlsx"
    )


def baseline_accuracy(X, y):
    """Function produce baseline accuracies of different machine learning models with standard scaler as preprocessig.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,).
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    cv_results: dict of numpy (masked) ndarrays
                    A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
    """
    seed = 123

    # Set our crossvalidation method
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=7)

    # Set our pipeline: scaling and the model
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    # pipe = Pipeline([('scaler', SavitzkyGolay()), ('clf', LogisticRegression(max_iter=1000))])

    # Set the models we want to test
    param_grid = [
        {"clf": [LogisticRegression(max_iter=1000)]},
        {"clf": [RandomForestClassifier(random_state=seed)]},
        {"clf": [SVC()]},
        {"clf": [DecisionTreeClassifier()]},
    ]

    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.cv_results_


def baseline_accuracy_derivative(X, y):
    """Function produce baseline accuracies of different machine learning models with standard scaler as preprocessig.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,).
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    cv_results: dict of numpy (masked) ndarrays
                    A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
    """
    seed = 123

    # Set our crossvalidation method
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=7)

    # Set our pipeline: scaling and the model
    # pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    pipe = Pipeline(
        [("scaler", SavitzkyGolay()), ("clf", LogisticRegression(max_iter=1000))]
    )

    # Set the models we want to test
    param_grid = [
        {"clf": [LogisticRegression(max_iter=1000)]},
        {"clf": [RandomForestClassifier(random_state=seed)]},
        {"clf": [SVC()]},
        {"clf": [DecisionTreeClassifier()]},
    ]

    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.cv_results_


def dataplotmelt(grid_result):
    """
    Creates a dataframe with results from a gridsearch

    Parameters:
    -----------
    grid_results: dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns,
        that can be imported into a pandas DataFrame.

    Returns:
    -----------
    data_plot: dataframe
    """

    df_grid = pd.DataFrame.from_dict(grid_result)

    data_plot = pd.melt(
        df_grid,
        id_vars=["param_clf"],
        value_vars=[
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
            "split3_test_score",
            "split4_test_score",
            "split5_test_score",
            "split6_test_score",
            "split7_test_score",
            "split8_test_score",
            "split9_test_score",
        ],
    )
    data_plot["param_clf"] = data_plot["param_clf"].astype(str)
    data_plot["param_clf"] = data_plot["param_clf"].str.replace(
        "LogisticRegression(max_iter=1000)", repl="LR", regex=False
    )
    data_plot["param_clf"] = data_plot["param_clf"].str.replace(
        "RandomForestClassifier(random_state=123)", "RF", regex=False
    )
    data_plot["param_clf"] = data_plot["param_clf"].str.replace(
        "SVC()", "SVC", regex=False
    )
    data_plot["param_clf"] = data_plot["param_clf"].str.replace(
        "DecisionTreeClassifier()", "CART", regex=False
    )

    return data_plot


def model_optimization(X, y):
    """
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples, n_output) \
        or (n_samples,).
        Target relative to X for classification or regression;
    
    Returns:
    --------
    best_model: estimator
                Estimator that was chosen by the search, \ i.e. estimator which 
                gave highest score (or smallest loss if specified) on the left 
                out data.    
    """
    scaler = StandardScaler()
    # scaler = SavitzkyGolay()
    model = LogisticRegression(penalty="l2", max_iter=10000)
    # model = RandomForestClassifier(random_state=123)

    pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])

    # param_grid = {
    # 'model__bootstrap': [True],
    # 'model__max_depth': [80, 90, 100, 110],
    # 'model__max_features': [2, 3],
    # 'model__min_samples_leaf': [3, 4, 5],
    # 'model__min_samples_split': [8, 10, 12],
    # 'model__n_estimators': [100, 200, 300, 1000]

    # }

    param_grid = {
        "model__C": [100, 10, 1.0, 0.1, 0.01],
        "model__solver": ["newton-cg", "lbfgs", "liblinear"],
        "model__penalty": ["l2"],
    }

    cv_grid = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

    # define search
    search = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=cv_grid, refit=True)

    # execute search
    result = search.fit(X, y)

    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_

    print(f"Best model parameters{result.best_params_}")

    return best_model


def model_optimization_rf(X, y):
    """
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples, n_output) \
        or (n_samples,).
        Target relative to X for classification or regression;
    
    Returns:
    --------
    best_model: estimator
                Estimator that was chosen by the search, \ i.e. estimator which 
                gave highest score (or smallest loss if specified) on the left 
                out data.    
    """
    # scaler = StandardScaler()
    scaler = SavitzkyGolay()
    # model = LogisticRegression(penalty='l2', max_iter=10000)
    model = RandomForestClassifier(random_state=123)

    pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])

    param_grid = {
        "model__bootstrap": [True],
        "model__max_depth": [80, 90, 100, 110],
        "model__max_features": [2, 3],
        "model__min_samples_leaf": [3, 4, 5],
        "model__min_samples_split": [8, 10, 12],
        "model__n_estimators": [100, 200, 300, 1000],
    }

    # param_grid = {'model__C': [100, 10, 1.0, 0.1, 0.01], 'model__solver': ["newton-cg", 'lbfgs', 'liblinear'], 'model__penalty':['l2']}

    cv_grid = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

    # define search
    search = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=cv_grid, refit=True)

    # execute search
    result = search.fit(X, y)

    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_

    print(f"Best model parameters{result.best_params_}")

    return best_model


# Test model


def train_model(X_hd_train, y_hd_train, best_model, name: str):
    best_model.fit(X_hd_train, y_hd_train)
    filename = "../results/models/trained_model_" + name + ".sav"
    joblib.dump(best_model, filename)


def test_model(X_hd_test, y_hd_test, loaded_model):
    """
    Test the final optimised model on test set

    Parameters:
    ----------
    X_hd_train:
    X_hd_test:
    y_hd_train:
    best_model:


    Returns:
    --------
    y_hd_pred:
    y_hd_prob:
    pipe:

    """
    print(loaded_model)
    y_hd_pred = loaded_model.predict(X_hd_test)
    # y_hd_prob = loaded_model.predict_proba(X_hd_test)
    acc = accuracy_score(y_hd_test, y_hd_pred)
    print(f"Accuracy on test set using the head: {acc}")

    return y_hd_pred


# nested cross validation


def nested_crossvalidation(X, y):
    """ "Perform nested cross validation on a classifier
    Parameters
    ----------
    X: dataframe
    y:

    Return
    ------
     cm_nested:
     y_pred_nested:
     y_test_nested:
    """

    scaler = StandardScaler()
    model = LogisticRegression(penalty="l2", max_iter=10000)

    pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])

    # configure nested cross-validation layers
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # create confusion matrix list to save each of external cv layer
    cm_nested = []
    y_pred_nested = []
    y_test_nested = []
    outer_results = list()

    for train_ix, test_ix in cv_outer.split(X, y):
        # split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # define search space
        param_grid = {
            "model__C": [100, 10, 1.0, 0.1, 0.01],
            "model__solver": ["newton-cg", "lbfgs", "liblinear"],
        }
        # define search
        search = GridSearchCV(
            pipe, param_grid, scoring="roc_auc", cv=cv_inner, refit=True
        )
        # execute search
        result = search.fit(X_train, y_train)

        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # create pipeline with the best model
        best_pipe = Pipeline(steps=[("scaler", scaler), ("best_model", best_model)])
        best_pipe.fit(X_train, y_train)

        # evaluate model on the hold out dataset

        y_pred = best_pipe.predict(X_test)
        y_desicion = best_pipe.decision_function(X_test)

        # evaluate the model
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # store the result
        outer_results.append(acc)
        cm_nested.append(cm)
        y_pred_nested.append(y_desicion)
        y_test_nested.append(y_test)

        print(
            f"Acc={(acc):.3f}, auc-roc={(result.best_score_):.3f}, cfg={result.best_params_}"
        )

    # print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
    print(f"Mean accuracy: {np.mean(outer_results):.2f} ({np.std(outer_results):.2f})")
    return cm_nested, y_pred_nested, y_test_nested, outer_results


def montecarlo_crossvalidation(X, y, best_model, binary=True):
    """
    Performs a Montecarlo crossvalidation for binary classifaction.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features

    y: array-like of shape (n_samples,)
        Target relative to X for classification or regression; None for unsupervised learning.

    best_model: Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data.

    binary: If the prediction problem is binary. Default=True

    Return
    ------
     accuracies_mc:
     sensitivty_total:
     specificity_total:

    """
    accuracies_mc = []
    SPLITS = 500
    sensitivty_total = []
    specificity_total = []

    cv = ShuffleSplit(n_splits=SPLITS, test_size=0.2, random_state=7)
    scaler = StandardScaler()
    # model = LogisticRegression(penalty='l2', max_iter=10000)

    pipe = Pipeline(steps=[("scaler", scaler), ("model", best_model)])

    for train_ix, test_ix in tqdm(
        cv.split(X, y), total=cv.get_n_splits(), desc="shuffle split"
    ):
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        accuracy_montecarlo = accuracy_score(y_test, y_pred)
        if binary == True:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivty = tp / (tp + fn)
            sensitivty_total.append(sensitivty)
            specificity = tn / (tn + fp)
            specificity_total.append(specificity)
            accuracies_mc.append(accuracy_montecarlo)
        else:
            accuracies_mc.append(accuracy_montecarlo)

    print(
        f"Model perfomance using monte carlo cross-validation\nMean accuracy = {np.mean(accuracies_mc):.2f} ± {np.std(accuracies_mc):.2f}"
    )

    return accuracies_mc, sensitivty_total, specificity_total


def variable_importance_df(wavenumbers, pipeline_best):
    """Function returns a dataframe with the 10 highest coefficients (positive and negative)
    Paramaters:
    -----------
    wvenumbers: array
    coefficients: pipeline

    Return
    --------
    final_sort: dataframe
    """

    # variable_importance = pd.DataFrame({"Wavenumbers": wavenumbers,
    #'Feature importance': pipeline_best['model'].feature_importances_})

    variable_importance = pd.DataFrame(
        {
            "Wavenumbers": wavenumbers,
            "Feature importance": pipeline_best["model"].coef_[0],
        }
    )

    variable_importance_sort_positive = variable_importance.sort_values(
        by=["Feature importance"], ascending=False
    ).head(10)
    variable_importance_sort_negative = variable_importance.sort_values(
        by=["Feature importance"], ascending=True
    ).head(10)

    final_sortlist = [
        variable_importance_sort_positive,
        variable_importance_sort_negative,
    ]
    final_sort = pd.concat(final_sortlist)
    # variable_importance_sort_positive['Wavenumbers'] = variable_importance_sort_positive['Wavenumbers'].astype('category')
    # return variable_importance_sort_positive
    return final_sort


def variable_importance_df_derv(wavenumbers, pipeline_best):
    """Function returns a dataframe with the 10 highest coefficients (positive and negative)
    Paramaters:
    -----------
    wvenumbers: array
    coefficients: pipeline

    Return
    --------
    final_sort: dataframe
    """

    variable_importance = pd.DataFrame(
        {
            "Wavenumbers": wavenumbers,
            "Feature importance": pipeline_best["model"].feature_importances_,
        }
    )

    # variable_importance = pd.DataFrame({"Wavenumbers": wavenumbers,
    #'Feature importance': pipeline_best['model'].coef_[0]})

    variable_importance_sort_positive = variable_importance.sort_values(
        by=["Feature importance"], ascending=False
    ).head(10)
    variable_importance_sort_negative = variable_importance.sort_values(
        by=["Feature importance"], ascending=True
    ).head(10)

    final_sortlist = [
        variable_importance_sort_positive,
        variable_importance_sort_negative,
    ]
    final_sort = pd.concat(final_sortlist)
    # variable_importance_sort_positive['Wavenumbers'] = variable_importance_sort_positive['Wavenumbers'].astype('category')
    # return variable_importance_sort_positive
    return final_sort


def gridsearch_bias(X_thorax_part1, y_thorax):
    clf1 = LogisticRegression(max_iter=1000, random_state=123)
    clf2 = SVC(random_state=123)
    clf3 = RandomForestClassifier(random_state=123)

    param1 = {}
    param1["clf__kernel"] = ["rbf", "linear"]
    param1["clf"] = [clf2]

    param2 = {}
    param2["clf"] = [clf1]

    param3 = {}
    param3["clf"] = [clf3]

    pipe = Pipeline([("clf", clf1)])

    # pipe = Pipeline([('scaler', SavitzkyGolay()), ('clf', clf1)])

    # pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf1)])

    params = [param1, param2, param3]
    cv = StratifiedShuffleSplit(n_splits=10, random_state=7, test_size=0.2)

    grid = GridSearchCV(pipe, params, cv=cv, scoring="accuracy")

    results_part1 = grid.fit(X_thorax_part1, y_thorax)

    dx = pd.DataFrame.from_dict(results_part1.cv_results_)
    mean_accuracies = dx[["param_clf", "param_clf__kernel", "mean_test_score"]]
    mean_accuracies_copy = mean_accuracies.copy()

    mean_accuracies_copy["param_clf"] = mean_accuracies_copy["param_clf"].astype(str)
    mean_accuracies_copy["param_clf"] = mean_accuracies_copy["param_clf"].str.replace(
        "LogisticRegression(max_iter=1000, random_state=123)", repl="LR", regex=False
    )
    mean_accuracies_copy["param_clf"] = mean_accuracies_copy["param_clf"].str.replace(
        "SVC(kernel='linear', random_state=123)", "SVM", regex=False
    )
    mean_accuracies_copy["param_clf"] = mean_accuracies_copy["param_clf"].str.replace(
        "RandomForestClassifier(random_state=123)", "RF", regex=False
    )

    return mean_accuracies_copy


import matplotlib.path as mpath
import matplotlib.patches as mpatches


def add_label_band(ax, top, bottom, label, *, spine_pos=-0.05, tip_pos=-0.02):
    """
    Helper function to add bracket around y-tick labels.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the bracket to

    top, bottom : floats
        The positions in *data* space to bracket on the y-axis

    label : str
        The label to add to the bracket

    spine_pos, tip_pos : float, optional
        The position in *axes fraction* of the spine and tips of the bracket.
        These will typically be negative

    Returns
    -------
    bracket : matplotlib.patches.PathPatch
        The "bracket" Aritst.  Modify this Artist to change the color etc of
        the bracket from the defaults.

    txt : matplotlib.text.Text
        The label Artist.  Modify this to change the color etc of the label
        from the defaults.

    """
    # grab the yaxis blended transform
    transform = ax.get_yaxis_transform()

    # add the bracket
    bracket = mpatches.PathPatch(
        mpath.Path(
            [
                [tip_pos, top],
                [spine_pos, top],
                [spine_pos, bottom],
                [tip_pos, bottom],
            ]
        ),
        transform=transform,
        clip_on=False,
        facecolor="none",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_artist(bracket)

    # add the label
    txt = ax.text(
        spine_pos,
        (top + bottom) / 2,
        label,
        ha="right",
        va="center",
        rotation="vertical",
        clip_on=False,
        transform=transform,
    )

    return bracket, txt
