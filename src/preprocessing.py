import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sn


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from scipy.signal import savgol_filter

from transformers import SavitzkyGolay
from transformers import StandardNormalVariate
from transformers import MultipleScatterCorrection
from transformers import RobustNormalVariate


def LR_baseline(X, y, win: float, order: float):
    """
    Perform logistic regression with a Savitzky-Golay filter preprocessing step.

    This function creates a pipeline that scales the data, applies a Savitzky-Golay filter,
    and then performs logistic regression. It uses stratified shuffle split cross-validation
    to evaluate the model's performance.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    win : float
        Window length for the Savitzky-Golay filter.
    order : float
        Derivative order for the Savitzky-Golay filter.

    Returns
    -------
    list
        A list containing the cross-validation accuracy scores.
    """
    
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("Savgol", SavitzkyGolay(filter_win=win, deriv_order=order)),
            ("LR", LogisticRegression(max_iter=10000)),
        ]
    )
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=7)

    results = []
    cv_results = cross_val_score(pipe, X, y.ravel(), cv=sss, scoring="accuracy")
    results.append(cv_results)
    print(cv_results.mean(), cv_results.std())
    return results


def baseline_gridsearch(
    X,
    y,
    win: float,
    order: float,
):
    """Create a baseline report of the accuray of common ML algorithms
    with different scatter corrections.

    Parameters
    ----------
    X : Features
    y : labels
    win : smoothing window
    order: Derivative order

    Returns
    -------
    gridesearch : dictionary

    """
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=7)
    pipe = Pipeline([("clf", LogisticRegression(max_iter=10000))])

    param_grid = [
        {"clf": [LogisticRegression(max_iter=10000)]},
        {"clf": [RandomForestClassifier()]},
        {"clf": [SGDClassifier()]},
        {"clf": [ExtraTreeClassifier()]},
        {"clf": [GaussianNB()]},
        {"clf": [SVC()]},
        {"clf": [DecisionTreeClassifier()]},
    ]

    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1, n_jobs=-1)

    X_sg = savgol_filter(X, window_length=win, deriv=order, polyorder=3)

    grid_search.fit(X_sg, y.ravel())

    return grid_search


def baseline_gridsearch_SNV(X, y, win: float, order: float):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=7)
    pipe = Pipeline(
        [
            ("scaler", StandardNormalVariate()),
            ("Savgol", SavitzkyGolay(filter_win=win, deriv_order=order)),
            ("clf", LogisticRegression(max_iter=10000)),
        ]
    )

    param_grid = [
        {"clf": [LogisticRegression(max_iter=10000)]},
        {"clf": [RandomForestClassifier()]},
        {"clf": [SGDClassifier()]},
        {"clf": [ExtraTreeClassifier()]},
        {"clf": [GaussianNB()]},
        {"clf": [SVC()]},
        {"clf": [DecisionTreeClassifier()]},
    ]
    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1, n_jobs=-1)
    grid_search.fit(X, y.ravel())

    return grid_search


def baseline_gridsearch_RNV(X, y, win: float, order: float):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=7)
    pipe = Pipeline(
        [
            ("scaler", RobustNormalVariate()),
            ("Savgol", SavitzkyGolay(filter_win=win, deriv_order=order)),
            ("clf", LogisticRegression(max_iter=10000)),
        ]
    )

    param_grid = [
        {"clf": [LogisticRegression(max_iter=10000)]},
        {"clf": [RandomForestClassifier()]},
        {"clf": [SGDClassifier()]},
        {"clf": [LinearDiscriminantAnalysis()]},
        {"clf": [ExtraTreeClassifier()]},
        {"clf": [GaussianNB()]},
        {"clf": [SVC()]},
        {"clf": [DecisionTreeClassifier()]},
    ]
    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1, n_jobs=-1)
    grid_search.fit(X, y.ravel())

    return grid_search


def baseline_gridsearch_MSC(X, y, win: float, order: float):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=7)
    pipe = Pipeline(
        [
            ("scaler", MultipleScatterCorrection()),
            ("Savgol", SavitzkyGolay(filter_win=win, deriv_order=order)),
            ("clf", LogisticRegression(max_iter=10000)),
        ]
    )

    param_grid = [
        {"clf": [LogisticRegression(max_iter=10000)]},
        {"clf": [RandomForestClassifier()]},
        {"clf": [SGDClassifier()]},
        {"clf": [LinearDiscriminantAnalysis()]},
        {"clf": [ExtraTreeClassifier()]},
        {"clf": [GaussianNB()]},
        {"clf": [SVC()]},
        {"clf": [DecisionTreeClassifier()]},
    ]
    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1, n_jobs=-1)
    grid_search.fit(X, y.ravel())

    return grid_search


def prepo_results_windows(
    total_zero_order, total_first_order, total_second_order, name: str
):
    raw_results = [total_zero_order, total_first_order, total_second_order]
    orders = [0, 1, 2]
    windows = [9, 11, 21]
    final_raw = []
    for t, o in zip(raw_results, orders):
        for i, n in zip(windows, t):
            dx = pd.DataFrame.from_dict(n.cv_results_)
            dx["Derivative"] = o
            dx["Window"] = i
            dx["Preprocessing"] = name
            final_raw.append(dx)
    df_final_first = pd.concat(final_raw)
    return df_final_first


def final_df_preprocessing(X):
    col = [
        "param_clf",
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
        "Derivative",
        "Window",
        "Preprocessing",
    ]
    df_final = X.loc[:, col]
    data_plot = pd.melt(
        df_final, id_vars=["param_clf", "Window", "Derivative", "Preprocessing"]
    )
    data_plot.rename(columns={"param_clf": "Model", "value": "Accuracy"}, inplace=True)
    data_plot["Model"] = data_plot["Model"].astype(str)
    data_plot.sort_values(
        by=[
            "Model",
        ],
        inplace=True,
    )

    data_plot["Model"] = data_plot["Model"].str.replace(
        "DecisionTreeClassifier()", "CART", regex=False
    )
    data_plot["Model"] = data_plot["Model"].str.replace("SVC()", "SVC", regex=False)
    data_plot["Model"] = data_plot["Model"].str.replace(
        "ExtraTreeClassifier()", "ET", regex=False
    )
    data_plot["Model"] = data_plot["Model"].str.replace(
        "GaussianNB()", "NB", regex=False
    )
    data_plot["Model"] = data_plot["Model"].str.replace(
        "LinearDiscriminantAnalysis()", "LDA", regex=False
    )
    data_plot["Model"] = data_plot["Model"].str.replace(
        "LogisticRegression(max_iter=10000)", "LR", regex=False
    )
    data_plot["Model"] = data_plot["Model"].str.replace(
        "RandomForestClassifier()", "RF", regex=False
    )
    data_plot["Model"] = data_plot["Model"].str.replace(
        "SGDClassifier()", "SGD", regex=False
    )

    return data_plot


def boxplot_preprocessing(X):
    """
    Create a boxplot to visualize the accuracy of different
    preprocessing methods.

    This function sets the context for seaborn plots,
    creates a categorical plot (boxplot)
    to compare the accuracy of different preprocessing
    methods across various window sizes, derivatives, and models. It also adds a horizontal line at y=0.3 for reference.

    Parameters
    ----------
    X : DataFrame
        A pandas DataFrame containing the data to be plotted. The DataFrame should have the
        following columns: 'Window', 'Accuracy', 'Preprocessing', 'Derivative', and 'Model'.

    Returns
    -------
    None
    """
    sn.set_context("notebook", font_scale=1.8)
    plot = sn.catplot(
        x="Window",
        y="Accuracy",
        hue="Preprocessing",
        col="Derivative",
        row="Model",
        kind="box",
        data=X,
        palette="PuBu",
    )
    axes = plot.axes.flatten()
    # iterate through the axes
    for ax in axes:
        ax.axhline(y=0.3, ls="--", c="red")


def baseline_gridsearch_2(
    X,
    y,
    win: float,
    order: float,
):
    """Create a baseline report of the accuray of common ML algorithms
    with different scatter corrections.

    Parameters
    ----------
    X : Features
    y : labels
    win : smoothing window
    order: Derivative order

    Returns
    -------
    gridesearch : dictionary

    """
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=7)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("Savgol", SavitzkyGolay(filter_win=win, deriv_order=order)),
            ("clf", LogisticRegression(max_iter=10000)),
        ]
    )

    print(pipe)

    param_grid = [
        {"clf": [LogisticRegression(max_iter=10000)]},
        {"clf": [RandomForestClassifier()]},
        {"clf": [SGDClassifier()]},
        {"clf": [ExtraTreeClassifier()]},
        {"clf": [GaussianNB()]},
        {"clf": [SVC()]},
        {"clf": [DecisionTreeClassifier()]},
    ]
    grid_search = GridSearchCV(pipe, param_grid, cv=sss, verbose=1, n_jobs=-1)
    grid_search.fit(X, y.ravel())

    return grid_search
