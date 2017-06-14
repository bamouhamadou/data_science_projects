from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble.bagging import BaggingClassifier, BaggingRegressor
from sklearn.ensemble.forest import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ElasticNetCV, LassoCV
from sklearn.linear_model.ridge import RidgeCV, RidgeClassifier, RidgeClassifierCV
from sklearn.linear_model.stochastic_gradient import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import Binarizer, FunctionTransformer, Imputer, LabelBinarizer, LabelEncoder, MaxAbsScaler, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from pandas import DataFrame
#from xgboost.sklearn import XGBClassifier, XGBRegressor

import numpy
import pandas






def load_csv(name):
	return pandas.read_csv("data/" + name, na_values = ["N/A", "NA"])

def store_csv(df, name):
	df.to_csv("csv/" + name, index = False)

def store_pkl(obj, name):
	joblib.dump(obj, "pkl/" + name, compress = 9)

def mape(sample_y, preds): 
    #y_true, y_pred = check_array(y_true, y_pred)
    return numpy.mean(numpy.abs((sample_y - preds) / sample_y)) * 100

def mape_on_ref(sample_y, preds): 
    #y_true, y_pred = check_array(y_true, y_pred)
    return numpy.mean((sample_y - preds) / sample_y) * 100


def build_sample(regressor, name):
	# train...
	regressor = regressor.fit(sample_X, sample_y)
	# save model
	store_pkl(regressor, name + ".pkl")
	# predict on train
	preds = regressor.predict(sample_X)
	# create DataFrame
	preds = DataFrame(preds, columns = ["prime_tot_ttc_preds"])
	# mape
	mape_r = mape(sample_y, preds["prime_tot_ttc_preds"])
	# print
	print "MAPE of %s is : %f" % (name, mape_r)
	# predict on test
	preds_on_test = DataFrame(list(zip(sample_id, regressor.predict(sample_t))), columns = ["id", "prime_preds"])
	preds_on_test['id'].astype(int)
	# print on ref
	print "MAPE of %s is : %f" % (name, mape_on_ref(sample_tR, preds_on_test["prime_preds"]))
	# save predictions
	store_csv(preds_on_test, name + ".csv")



sample = load_csv("sample.csv")

print sample.dtypes

sample['id'].astype(int)
sample['ann_obt_permis'] = sample['annee_permis'] - sample['annee_naissance']
sample['duree_permis'] = 2016 - sample['annee_permis']

print(sample.dtypes)

# 14 Feature hidden
mapper = DataFrameMapper([
	(["id"], None),
	(["crm", 
	"ann_obt_permis", 
	"duree_permis", 
	"var1",
	"var2",
	"var3",
	"var4",
	"var5",
	"var17",
	"var18",
	"var19",
	"var20",
	"var21",
	"var22",
	"var9",
	"var10",
	"var11",
	"var12",
	"var13",
	"kmage_annuel",
	"puis_fiscale",
	"anc_veh",
	"var15"], 
	[ContinuousDomain(), StandardScaler()]),
	(["marque"], LabelEncoder()),
	(["energie_veh"], LabelEncoder()), 
        (["profession"], LabelEncoder()), 
        (["var6"], LabelEncoder()),
        (["var7"], LabelEncoder()),
        (["var8"], LabelEncoder()),
        (["var16"], LabelEncoder()),
	(["prime_tot_ttc"], None)
])


sample_mapper = mapper.fit_transform(sample)

print sample_mapper.shape
#print sample_mapper.columns.tolist()

store_pkl(mapper, "mapper.pkl")

sample_X = sample_mapper[0:300000, 1:31]
sample_y = sample_mapper[0:300000, 31]
# we use reference
sample_t = sample_mapper[300000:330000, 1:31]
sample_id = sample_mapper[300000:330000, 0]
# ref
sample_r = load_csv("reference.csv")
sample_tR = sample_r["CODIS"]

print sample_X.shape
print sample_y.shape
print sample_t.shape

#print(auto_X.dtype, auto_y.dtype)

build_sample(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), "DecisionTreeAuto")
build_sample(BaggingRegressor(DecisionTreeRegressor(random_state = 13, min_samples_leaf = 5), random_state = 13, n_estimators = 3, max_features = 0.5), "DecisionTreeEnsembleAuto")
build_sample(ElasticNetCV(random_state = 13), "ElasticNetAuto")
build_sample(ExtraTreesRegressor(random_state = 13, min_samples_leaf = 5), "ExtraTreesAuto")
build_sample(GradientBoostingRegressor(random_state = 13, init = None), "GradientBoostingAuto")
build_sample(LassoCV(random_state = 13), "LassoAuto")
build_sample(LinearRegression(), "LinearRegressionAuto")
build_sample(BaggingRegressor(LinearRegression(), random_state = 13, max_features = 0.5), "LinearRegressionEnsembleAuto")
build_sample(RandomForestRegressor(random_state = 13, min_samples_leaf = 5), "RandomForestAuto")
build_sample(RidgeCV(), "RidgeAuto")
#build_sample(XGBRegressor(objective = "reg:linear"), "XGBAuto")
