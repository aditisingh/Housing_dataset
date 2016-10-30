from sklearn import svm
from sklearn import preprocessing
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data=np.loadtxt('housing.data')

num_instances=len(data)
num_fields=len(data[0])

# crim=[]#per capita crime rate by town
# zn=[]#proportion of residential land zoned for lots over  25,000 sq.ft.
# indus=[]#proportion of non-retail business acres per town
# chas=[]#= 1 if tract bounds river; 0 otherwise)
# nox=[]#nitric oxides concentration (parts per 10 million)
# rm=[]#average number of rooms per dwelling
# age=[]# proportion of owner-occupied units built prior to 1940
# dis=[]#weighted distances to five Boston employment centres
# rad=[]#index of accessibility to radial highways
# tax=[]#full-value property-tax rate per $10,000
# ptratio=[]#pupil to teacher ratio by town
# #b=[]#1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# lstat=[]#lower status of the population
# medv=[]#Median value of owner-occupied homes in $1000's


# for i in xrange(0, num_instances):
#     crim.append(data[i][0])
#     zn.append(data[i][1])
#     indus.append(data[i][2])
#     chas.append(data[i][3])
#     nox.append(data[i][4])
#     rm.append(data[i][5])
#     age.append(data[i][6])
#     dis.append(data[i][7])
#     rad.append(data[i][8])
#     tax.append(data[i][9])
#     ptratio.append(data[i][10])
#     b.append(data[i][11])
#     lstat.append(data[i][12])
#     medv.append(data[i][13])


X=data[:,0:13]
Y=data[:,13]


def one_h_model():
    # create model with one hidden 
    model = Sequential()
    model.add(Dense(26, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=one_h_model, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))