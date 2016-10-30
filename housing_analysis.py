
import numpy as np
from sklearn import svm
from sklearn import preprocessing


data=np.loadtxt('housing.data')

num_instances=len(data)
num_fields=len(data[0])

crim=[]#per capita crime rate by town
zn=[]#proportion of residential land zoned for lots over  25,000 sq.ft.
indus=[]#proportion of non-retail business acres per town
chas=[]#= 1 if tract bounds river; 0 otherwise)
nox=[]#nitric oxides concentration (parts per 10 million)
rm=[]#average number of rooms per dwelling
age=[]# proportion of owner-occupied units built prior to 1940
dis=[]#weighted distances to five Boston employment centres
rad=[]#index of accessibility to radial highways
tax=[]#full-value property-tax rate per $10,000
ptratio=[]#pupil to teacher ratio by town
#b=[]#1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
lstat=[]#lower status of the population
medv=[]#Median value of owner-occupied homes in $1000's


for i in xrange(0, num_instances):
    crim.append(data[i][0])
    zn.append(data[i][1])
    indus.append(data[i][2])
    chas.append(data[i][3])
    nox.append(data[i][4])
    rm.append(data[i][5])
    age.append(data[i][6])
    dis.append(data[i][7])
    rad.append(data[i][8])
    tax.append(data[i][9])
    ptratio.append(data[i][10])
    #b.append(data[i][11])
    lstat.append(data[i][12])
    medv.append(data[i][13])

crim_test=[]
crim_train=[]
zn_test=[]
zn_train=[]
indus_train=[]
indus_test=[]
chas_test=[]
chas_train=[]
nox_test=[]
nox_train=[]
rm_test=[]
rm_train=[]
age_test=[]
age_train=[]
dis_test=[]
dis_train=[]
rad_test=[]
rad_train=[]
tax_test=[]
tax_train=[]
ptratio_test=[]
ptratio_train=[]
lstat_test=[]
lstat_train=[]
medv_train=[]
medv_test=[]

data_train=[]
data_test=[]

label_train=[]
label_test=[]

n=int(0.7*num_instances)
for i in xrange(0,n):
    data_train.append([data[i][0], data[i][1], data[i][2],data[i][3],data[i][4], data[i][5], data[i][6], data[i][7],data[i][8], data[i][9], data[i][10], data[i][12]])
    label_train.append(data[i][13])
    crim_train.append(data[i][0])
    zn_train.append(data[i][1])
    indus_train.append(data[i][2])
    chas_train.append(data[i][3])
    nox_train.append(data[i][4])
    rm_train.append(data[i][5])
    age_train.append(data[i][6])
    dis_train.append(data[i][7])
    rad_train.append(data[i][8])
    tax_train.append(data[i][9])
    ptratio_train.append(data[i][10])
    #b.append(data[i][11])
    lstat_train.append(data[i][12])
    medv_train.append(data[i][13])

for i in xrange(n,num_instances):
    data_test.append([data[i][0], data[i][1], data[i][2],data[i][3],data[i][4], data[i][5], data[i][6], data[i][7],data[i][8], data[i][9], data[i][10], data[i][12]])
    label_test.append(data[i][13])
    crim_test.append(data[i][0])
    zn_test.append(data[i][1])
    indus_test.append(data[i][2])
    chas_test.append(data[i][3])
    nox_test.append(data[i][4])
    rm_test.append(data[i][5])
    age_test.append(data[i][6])
    dis_test.append(data[i][7])
    rad_test.append(data[i][8])
    tax_test.append(data[i][9])
    ptratio_test.append(data[i][10])
    #b.append(data[i][11])
    lstat_test.append(data[i][12])
    medv_test.append(data[i][13])


#preprocessing the data

data_train_scaled = preprocessing.scale(data_train)
data_test_scaled = preprocessing.scale(data_test)


#Linear regression

lin_regr=linear_model.LinearRegression()

lin_regr.fit(data_train_scaled,label_train)
err_lin=(np.mean((lin_regr.predict(data_test_scaled)-label_test)**2))/len(label_test)

# SVM
clf=svm.SVR()
clf.fit(data_train_scaled, label_train)
res=clf.predict(data_test_scaled)
err_svm=(np.mean(res-label_test)**2)/len(label_test)

from sklearn import decomposition
