from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt
import graphviz 
import pandas as pd
import numpy as np

class DecisionTreeSurrogate():
    def __init__(self, tree_depth=4):
        self.dr = DecisionTreeRegressor(max_depth=tree_depth)
        
    # define a function to train an model from a base estimator and using a surrogate decision tree to interpret the model
    # created a k fold split to perform the surrogate training
    def train(self, est, X_train, y_train):
        kf = KFold(n_splits=2,shuffle=True)
        kf.get_n_splits(X_train,y_train)
        folds = []
        for train_idx, test_idx in kf.split(X_train,y_train):
            folds.append({
                'X_train':X_train.iloc[train_idx],
                'X_test':X_train.iloc[test_idx],
                'y_train':y_train.iloc[train_idx],
                'y_test':y_train.iloc[test_idx]
            })

        print("Training base estimator on first fold.")
        est.fit(folds[0]['X_train'],folds[0]['y_train'])
        print(est)

        

        y_pred_est0 = est.predict(folds[0]['X_test'])
        score = sqrt(MSE(folds[0]['y_test'],y_pred_est0))
        print("Base estimator score on first fold: {}".format(score))

        print("Training decision tree on second fold and results from base estimator.")
        y_pred_hat = est.predict(folds[1]['X_train'])
        y_pred_hat = pd.DataFrame(y_pred_hat,index=folds[1]['X_train'].index)

        self.dr.fit(folds[1]['X_train'],y_pred_hat)

        y_pred_dr = self.dr.predict(folds[1]['X_test'])

        score = sqrt(MSE(folds[1]['y_test'],y_pred_dr))
        print("Score for decision tree: {}".format(score))

    def visualizeTree(self, feature_names):
        print("Visualizing decision tree")
        dot_data = export_graphviz(self.dr, out_file=None, feature_names=feature_names, filled=True, rounded=True, special_characters=True) 
        graph = graphviz.Source(dot_data)
        display(graph)