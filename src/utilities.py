import pandas as pd

# preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# Test model
def test_model(X_hd_train, X_hd_test, y_hd_train):
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    pipe.fit(X_hd_train, y_hd_train)
    y_hd_pred = pipe.predict(X_hd_test)

    return y_hd_pred, pipe