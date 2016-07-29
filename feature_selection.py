"""
Greedy Feature Selection using Logistic Regression as base model
to optimize Area Under the ROC Curve
__author__ : Abhishek
Credits : Miroslaw @ Kaggle
"""


import numpy

import sklearn.linear_model as lm
from sklearn import metrics, preprocessing


class GreedyFeatureSelection(object):

    def __init__(self, model, data, labels, scale=1, verbose=0):
	self._model = model
	if scale == 1:
	    self._data = preprocessing.scale(numpy.array(data))
	else:
	    self._data = numpy.array(data)
	self._labels = labels
	self._verbose = verbose


    def evaluateScore(self):
	model = self._model
	X = self._data
	y = self._labels
	#model = lm.LogisticRegression()
	model.fit(X, y)
	predictions = model.predict(X)
	means = numpy.mean(numpy.abs((y - predictions) / y)) * 100
	return means

    def selectionLoop(self):
	# set
	X = self._data
 	# begin
	score_history = []
	good_features = set([])
	num_features = X.shape[1]
	while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]:
	    scores = []
	    for feature in range(num_features):
	        if feature not in good_features:
	            selected_features = list(good_features) + [feature]

	            Xts = numpy.column_stack(X[:, j] for j in selected_features)

	            score = self.evaluateScore()
	            scores.append((score, feature))

	            if self._verbose:
	                print "Current MAPE : ", numpy.mean(score)

	    good_features.add(sorted(scores)[-1][1])
	    score_history.append(sorted(scores)[-1])
	    if self._verbose:
	        print "Current Features : ", sorted(list(good_features))

	# Remove last added feature
	good_features.remove(score_history[-1][1])
	good_features = sorted(list(good_features))
	if self._verbose:
	    print "Selected Features : ", good_features

	return good_features

    def transform(self):
	good_features = self.selectionLoop()
	#return X[:, good_features]
	return good_features
