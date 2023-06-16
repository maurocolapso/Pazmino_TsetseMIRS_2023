import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import seaborn as sn
from tqdm import tqdm

# preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit

# models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def baseline_accuracy(X, y):
    '''Function produce baseline accuracies of different machine learning models with standard scaler as preprocessig.
    
    Parameters:
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,)The target variable to try to predict in the case of
        supervised learning.
    
    Returns
    -------
    cv_results: dict of numpy (masked) ndarrays
    A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
    '''
    seed=123

    # Set our crossvalidation method
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=7)

    # Set our pipeline: scaling and the model
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])

# Set the models we want to test
    param_grid = [{"clf": [LogisticRegression(max_iter=1000)]},
                {"clf": [RandomForestClassifier(random_state=seed)]},
                {"clf": [SVC()]},
                {"clf": [DecisionTreeClassifier()]}]
    
    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1,n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.cv_results_

def dataplotmelt(grid_result):
    """ Crate a dataframe with results from a gridsearch
    Parameters:
    
    grid_results: dict of numpy (masked) ndarrays
    A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.

    data_plot: dataframe
    """

    df_grid = pd.DataFrame.from_dict(grid_result)

    data_plot = pd.melt(df_grid, id_vars = ["param_clf"], value_vars=["split0_test_score",	"split1_test_score","split2_test_score"	,"split3_test_score",	"split4_test_score","split5_test_score","split6_test_score"	,"split7_test_score", "split8_test_score",	"split9_test_score"])
    data_plot['param_clf'] = data_plot['param_clf'].astype(str)
    data_plot['param_clf'] = data_plot['param_clf'].str.replace('LogisticRegression(max_iter=1000)',repl='LR', regex=False)
    data_plot['param_clf'] = data_plot['param_clf'].str.replace('RandomForestClassifier(random_state=123)', 'RF',regex=False)
    data_plot['param_clf'] = data_plot['param_clf'].str.replace('SVC()', 'SVC',regex=False)
    data_plot['param_clf'] = data_plot['param_clf'].str.replace('DecisionTreeClassifier()', 'CART', regex=False)
    
    return data_plot


def model_optimization(X,y):
    
    scaler = StandardScaler()
    model = LogisticRegression(penalty='l2', max_iter=10000)

    pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])


    param_grid = {'model__C': [100, 10, 1.0, 0.1, 0.01], 'model__solver': ["newton-cg", 'lbfgs', 'liblinear'], 'model__penalty':['l2']}
    
    cv_grid = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

    # define search
    search = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=cv_grid, refit=True)

    # execute search
    result = search.fit(X, y)
        
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_

    print(f'Best model parameters{result.best_params_}')

    return best_model


# Test model
def test_model(X_hd_train, X_hd_test, y_hd_train, best_model):
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', best_model)])
    pipe.fit(X_hd_train, y_hd_train)
    y_hd_pred = pipe.predict(X_hd_test)
    y_hd_prob = pipe.predict_proba(X_hd_test)

    return y_hd_pred, y_hd_prob, pipe


# nested cross validation

def nested_crossvalidation(X, y):
    """"Perform nested cross validation on a classifier
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
    model = LogisticRegression(penalty='l2', max_iter=10000)

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
        param_grid = {'model__C': [100, 10, 1.0, 0.1, 0.01], 'model__solver': ["newton-cg", 'lbfgs', 'liblinear']}
        # define search
        search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv_inner, refit=True)
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

        print(f"Acc={(acc):.3f}, auc-roc={(result.best_score_):.3f}, cfg={result.best_params_}")
    
    #print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
    print(f'Mean accuracy: {np.mean(outer_results):.2f} ({np.std(outer_results):.2f})')    
    return cm_nested, y_pred_nested, y_test_nested, outer_results


def nested_ROC_plot(y_test_nested, y_pred_nested, ax=None):
    """
    Plot ROC curve of each outer layer from nested crossvalidation
    with the standard deviation.

    Parameters
    ----------
    y_test_nested: Estimated targets as returned by a classifier.
         Ground truth (correct) target values.
         y_test splits used in nested cv.

    y_pred_nested: Estimated targets as returned by a classifier.
        Desicion function of the classifier.

    ax: matplotlib axes, default=None
        Axes object to plot on. If None, a new figure and axes is created.
    
    Return
    ------
     viz: RocCurveDisplay
        Object that stores computed values
    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    sn.set_style("ticks")
    #sn.set_context("talk", font_scale=1.5)
    
    # create axis
    ax=ax

    for y_test, y_pred in zip(y_test_nested, y_pred_nested):

    # ROC curve from predictions
        viz = RocCurveDisplay.from_predictions(
                y_test,
                y_pred,
                alpha=0.4,
                lw=1,
                #label="_",
                ax=ax)


        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Adding dumb classifier and standard deviation 
    
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)


    ax.plot(
        mean_fpr,
        mean_tpr,
        color="darkblue",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2.5,
        alpha=1)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2)
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        ylabel="True positive rate",
        xlabel="False positive rate")


    #ax.legend(fontsize=15)
    ax.get_legend().remove()
    print(f"Mean AUC = {mean_auc:.3f} ({std_auc:.3f})")
    return viz


def montecarlo_crossvalidation(X,y, best_model, binary=True):

    accuracies_mc = []
    SPLITS = 100
    sensitivty_total = []
    specificity_total = []

    cv = ShuffleSplit(n_splits=SPLITS, test_size=0.2, random_state=7)
    scaler = StandardScaler()
    #model = LogisticRegression(penalty='l2', max_iter=10000)

    pipe = Pipeline(steps=[("scaler", scaler), ("model", best_model)])

    for train_ix, test_ix in tqdm(cv.split(X, y),total=cv.get_n_splits(), desc="shuffle split"):
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        accuracy_montecarlo = accuracy_score(y_test, y_pred)
        if binary == True:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivty = tp/(tp+fn)
            sensitivty_total.append(sensitivty)
            specificity = tn/(tn+fp)
            specificity_total.append(specificity)
            accuracies_mc.append(accuracy_montecarlo)
        else:
            accuracies_mc.append(accuracy_montecarlo)
    
    print(f'Model perfomance using monte carlo cross-validation\nMean accuracy = {np.mean(accuracies_mc):.2f} ± {np.std(accuracies_mc):.2f}')
                  
    
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

    variable_importance = pd.DataFrame({"Wavenumbers": wavenumbers,
    'Coefficients': pipeline_best['model'].coef_[0].T})


    variable_importance_sort_positive = variable_importance.sort_values(by=["Coefficients"], ascending=False).head(10)
    variable_importance_sort_negative = variable_importance.sort_values(by=["Coefficients"], ascending=True).head(10)

    final_sortlist = [variable_importance_sort_positive,variable_importance_sort_negative]
    final_sort = pd.concat(final_sortlist)
    return final_sort
