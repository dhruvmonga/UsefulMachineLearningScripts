# Copyright 2019, Dhruv Monga, All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

"""Learning Curve Function from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html"""
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=(10,5))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


from sklearn import model_selection
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

class ModelComparer():
    def __init__(self,models,names,X_train,y_train,folds=10,scoring='r2',ltype='reg'):
        if len(models) != len(names):
            print("Number of models must be same as number of names")
            return
        self.models = models
        self.names = names
        self.X_train = X_train
        self.y_train = y_train
        self.folds = folds
        self.score = r2_score
        self.type = ltype
        if scoring == 'roc_auc':
            self.score = roc_auc_score
            self.type = 'cls'
        else:
            self.score = scoring
            self.type = ltype
    
    def calcScores(self):
        self.results = []
        print("Calculating cross val scores")
        for name, model in zip(self.names,self.models):
            print("Calculating cross val score for {}".format(name))
            kfold = model_selection.KFold(n_splits=self.folds, random_state=42)
            cv_results = model_selection.cross_val_score(model, self.X_train, self.y_train, cv=kfold, verbose=1, n_jobs=4)
            self.results.append(cv_results)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        fig = plt.figure()
        fig.suptitle('Model Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(self.results)
        ax.set_xticklabels(self.names)
        plt.xticks(rotation=90)
        plt.show()
        
    def showLearningCurves(self):
        print("Plotting learning curves")
        for name, model in zip(self.names,self.models):
            print("Plotting learning curves for {}".format(model))
            title = 'Learning curves {}'.format(name)
            plot_learning_curve(model, title, self.X_train, self.y_train, cv=self.folds, n_jobs=4)
        plt.show()
    
    def testModelIndependence(self):
        name_df = pd.DataFrame(self.names,columns=['Model Name'])
        res_df = pd.DataFrame(self.results,columns=['cv_'+str(i) for i in range(self.folds)])
        concat_df = pd.concat([name_df,res_df],axis=1)
        concat_df = pd.melt(concat_df,id_vars='Model Name',var_name='Fold',value_name='Score')
        scores = list(list(concat_df[concat_df['Model Name'] == col]['Score'].values) for col in concat_df['Model Name'].unique())
        f, p = stats.f_oneway(*scores)
        
        print ('One-way ANOVA')
        print ('=============')

        print ('F value:', f)
        print ('P value:', p, '\n')
        
        mc = MultiComparison(concat_df['Score'], concat_df['Model Name'])
        result = mc.tukeyhsd()
        
        print(result)
        result.plot_simultaneous()
        plt.show()
