""" data science local team  
Mouhamadou Ba <mandiayba@gmail.com>
"""

from __future__ import division
from sklearn.preprocessing import LabelEncoder

import numpy 
import pandas
from sklearn import (metrics, cross_validation, ensemble, preprocessing)

SEED = 42  # always use a seed for randomized procedures

# load files
def load_data(filename, use_labels=True):

    data = pandas.read_csv(open('data/' + filename), dtype=None, delimiter=',', usecols = range(0, 31), header=0)
    if use_labels:
	labels = pandas.read_csv(open('data/' + filename), delimiter=',', usecols=[32], skiprows=0)
    else:
        labels = numpy.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

# ==== data preparation ==== #
def prepareData(X):
    X['ann_obt_permis'] = X['annee_permis'] - X['annee_naissance']
    X.drop(['codepostal'], inplace=True, axis=1)
    X.drop(['annee_permis'], inplace=True, axis=1)
    X.drop(['annee_naissance'], inplace=True, axis=1)
    X.drop(['var14'], inplace=True, axis=1)
    
       # === Label encoding === #
    print X
    # ==== marque, codepostal, energie_veh, profession, var6, var7, var8, var14, var16
    X = MultiColumnLabelEncoder(columns=['marque', 'energie_veh', 'profession', 'var6','var7', 'var8', 'var16']).fit_transform(X)
    print X
    return X

def main():
    """
    Fit models and make predictions.
    We'll use one-hot encoding to transform our categorical features
    into binary features.
    y and X will be numpy array objects.
    """
    #model = linear_model.Lasso()  # the classifier we'll use
    model = ensemble.RandomForestRegressor()

    # === load data in memory === #
    print "loading data"
    y, X = load_data('train.csv')
    #y_test, X_test = load_data('test.csv', use_labels=False)
    
    # prepare data
    X = prepareData(X)

    # === training & metrics === #
    mean_auc = 0.0
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions
        model.fit(X_train, y_train.values.ravel()) 
        preds = model.predict(X_cv)#[:, 1]

	print preds

        # compute MAE metric for this CV fold
        means = metrics.mean_absolute_error(y_cv, preds)
        #roc_auc = metrics.auc(fpr, tpr)
        #print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        #mean_auc += roc_auc

    print "Mean MAE: %f" % (means)

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    ##model.fit(X, y)
    #preds = model.predict(X_test)#[:, 1]
    #filename = raw_input("Enter name for submission file: ")
    #save_results(preds, filename + ".csv")

if __name__ == '__main__':
    main()
