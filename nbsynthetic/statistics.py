"""
Created on 01/09/2022
@author: Javier Marin 
Performs several statistical tests.
"""

import gc
import os
import numpy as np
from scipy.stats import ttest_ind, kstest, wilcoxon
from sklearn.manifold import TSNE
from sklearn.utils import parallel_backend
from sklearn.compose import make_column_selector as selector
from sklearn import metrics
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
import warnings
warnings.filterwarnings("ignore")


def columns_type(df: pd.DataFrame):
    """
    Args:
        df(pd.DataFrame):
            input data

    Returns:
        two lists with numerical and 
        categorical columns names.
    """
    numerical_columns_selector = selector(
        dtype_exclude=CategoricalDtype
    )
    categorical_columns_selector = selector(
        dtype_include=CategoricalDtype
    )
    numerical_columns = numerical_columns_selector(df)
    categorical_columns = categorical_columns_selector(df)
    return numerical_columns, categorical_columns


def reduce_dimensions(
    df_original: pd.DataFrame,
    df_synthetic: pd.DataFrame
):
    """Args:

          df_original (pd.DataFrame): 
            input data frame
          df_synthetic(pd.DataFrame): 
            synthetic dataframe

      Returns:
          embeddings (np.arrays)"""

    with parallel_backend(
        'multiprocessing',
        n_jobs=os.cpu_count() - 1
    ):
        original_emb = TSNE(
            n_components=2,
            perplexity=1,
            learning_rate='auto',
            init='random'
        ).fit_transform(df_original)

        synthetic_emb = TSNE(
            n_components=2,
            perplexity=1,
            learning_rate='auto',
            init='random'
        ).fit_transform(df_synthetic)
        gc.collect()
    return original_emb, synthetic_emb


def mmd_rbf(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    gamma
):
    """
    Maximum Mean Discrepancy (MMD) is a statistical tests 
    to determine if two samples are from different distributions.
    This statistic test measures the distance between the means 
    of the two samples  mapped into a reproducing kernel Hilbert space (RKHS).
    Maximum Mean Discrepancy has found numerous applications in 
    machine learning and nonparametric testing [1][2].

    Maths[3]: 
        Compute the radial basis function (RBF) kernel 
        between two vectors between X and Y.
        k(x,y) = exp(-gamma * ||x-y||^2 / 2)
        where gamma is the inverse of the standard 
        deviation of the RBF. A small gamma value define 
        a Gaussian function with a large variance.

    [1] Ilya Tolstikhin, Bharath K. Sriperumbudur, and Bernhard Schölkopf. 2016. 
    Minimax estimation of maximum mean discrepancy with radial kernels. 
    In Proceedings of the 30th International Conference on Neural 
    Information Processing Systems (NIPS'16). 
    Curran Associates Inc., Red Hook, NY, USA, 1938–1946.

    [2] A. Gretton, K. M. Borgwardt, M. Rasch, B. Schölkopf, and A. Smola. 
    A kernel method for the two sample problem. 
    In B. Schölkopf, J. Platt, and T. Hoffman, editors, Advances in Neural
    Information Processing Systems 19, pages 513–520, Cambridge, MA, 2007. MIT Press.

    [3] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

        Args:

           X: ndarray/pd.DataFrame of shape (n_samples_X, n_features)
           Y: ndarray/pd.DataFrame of shape (n_samples_Y, n_features)
           gamma: float

        Returns:
            Maximum Mean Discrepancy (MMD) value
    """

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()

    return print(f'Maximum Mean Discrepance = {mmd:.5f}')


def t_test(
    X: pd.DataFrame,
    Y: pd.DataFrame
):
    """Compare if the means are equivalent 
    with Student t-test. 
    H0 = null hypothesis  null hypothesis that 
    there is no effective difference between the 
    observed sample mean and the hypothesized 
    or stated population mean. 
    If p < 0.05, H0 is rejected"""
    """Args:

          X: pd.DataFrame of shape (n_samples_X, n_features)
          Y: pd.DataFrame of shape (n_samples_Y, n_features)
                
      Returns:
          list 
            list of features and t-test resulting p-values
          """
    ttest = ttest_ind(
        np.array(X),
        np.array(Y),
        axis=0,
        equal_var=False,
        nan_policy='propagate',
        permutations=None,
        random_state=None,
        alternative='two-sided',
        trim=0
    )
    return np.round(ttest.pvalue, 4),\
        np.round(ttest.statistic, 2)


def Wilcoxon(
    X: pd.DataFrame,
    Y: pd.DataFrame
):
    """
    The Wilcoxon signed-rank test tests the null hypothesis 
    that two related paired samples come from the same 
    distribution. In particular, it tests whether the 
    distribution of the differences x - y is symmetric 
    about zero.  Wilcoxon signed-rank test, which is the 
    nonparametric version of the paired Student’s t-test. 
    This test has less statistical power than the paired t-test,
    although more power when the expectations of the t-test 
    are violated, such as independence, when x or y does 
    not follows a normal distribution or when  x and y haven't 
    the same variance.
    H0 = null hypothesis is that data vectors 
    x1 and x2 are samples from the same 
    distribution. If p < 0.05 H0 is rejected
    """
    """Args:

          X: pd.DataFrame of shape (n_samples_X, n_features)
          Y: pd.DataFrame of shape (n_samples_Y, n_features)
                
      Returns:
          list 
            list of features + Wilcoxon signed-rank 
            test p values
          """
    w = []
    for c in Y.columns:
        w.append([
            'Wilcoxon signed-rank test',
            c,
            f'p_value={wilcoxon(X[c], Y.sample(len(X))[c])[1]:.5f}'
        ]
        )
    return w


def Student_t(
    X: pd.DataFrame,
    Y: pd.DataFrame
):
    """Compares if the means are equivalent 
    with Student t-test. 
    H0 = null hypothesis  null hypothesis that 
    there is no effective difference between the 
    observed sample mean and the hypothesized 
    or stated population mean. 
    If p < 0.05, H0 is rejected"""
    """Args:

          X: pd.DataFrame of shape (n_samples_X, n_features)
          Y: pd.DataFrame of shape (n_samples_Y, n_features)
                
      Returns:
          list 
            list of features + Student_t test p values
          """

    numerical_columns, categorical_columns = columns_type(X)

    if len(categorical_columns) > 0:
        t = []
        for c in Y[categorical_columns].columns:
            t.append(
                [
                    'Student-t test (categorical feature)',
                    c,
                    f'p_value={t_test(X[c], Y[c])[0]:.5f}'
                ]
            )
        return t

    else:
        print("There aren't categorial or boolean variables in your dataset to perform Student-t test.")


def Kolmogorov_Smirnov(
    X: pd.DataFrame,
    Y: pd.DataFrame
):
    """Performs the two-sample Kolmogorov-Smirnov 
    test for goodness of fit. The one-sample test 
    compares the underlying distribution F(x) of 
    a sample against a given distribution G(x). 
    H0 = null hypothesis is that data vectors 
    x1 and x2 are from populations with the same 
    distribution. If p < 0.05 H0 is rejected"""
    """Args:

          X: pd.DataFrame of shape (n_samples_X, n_features)
          Y: pd.DataFrame of shape (n_samples_Y, n_features)
                
      Returns:
          list 
            list of features an t-test p values
          """
    numerical_columns, categorical_columns = columns_type(X)

    if len(numerical_columns) > 0:
        ks = []
        for c in Y[numerical_columns].columns:
            ks.append(
                [
                    'Kolmogorov-Smirnov test (numerical feature)',
                    c,
                    f'p_value={kstest(X[c], Y[c])[1]:.5f}'
                ]
            )

        return ks

    else:
        print("There aren't numerical columns in the dataset to perform KS test.")


def plot_histograms(
    X: pd.DataFrame,
    Y: pd.DataFrame
):
    """ 
      Visually compare the distribution plots
      of each feature and shows the Wilcoxon 
      test values. Use as probability density
      as histnorm.

        Args:

            X: pd.DataFrame of shape(n_samples_X, n_features)
            Y: pd.DataFrame of shape (n_samples_Y, n_features)

        Returns:
            Plotly figure
     """
    try:
        import plotly.figure_factory as ff
    except ImportError:
        print(
            'Histograms cannot be ploted, you need to install plotly '
            '(pip install plotly) in order to execute this function.'
        )
        return

    for c in X.columns:
        fig = ff.create_distplot(
            hist_data=[np.array(X[c]), np.array(Y[c])],
            group_labels=list([f'{c}_Original', f'{c}_Synthetic']),
            show_hist=False,
            show_rug=False
        )
        # Overlay both histograms
        fig.update_layout(
            barmode='overlay',
            autosize=False,
            width=500,
            height=300,
            title=f'Feature = {c}. <br>Wilcoxon test p-value= {wilcoxon(X[c], Y.sample(len(X))[c])[1]:.5f}',
            title_font_size=12
        )
        fig.update_traces(opacity=0.65)
        fig.show()
    return
