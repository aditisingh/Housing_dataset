# Housing_dataset

Analysis on a large dataset of housing prices and features influencing them for Boston. The dataset used is the publicly available dataset "housing" by UCI. 
We attempt to find the performance metric weighing multiple numeric statistics from the dataset.  
We have compared performance by methods like linear regression, support vector machines and principal component analysis.  
The neural network implementation performed better than those and hence is being reported.

Attribute Information(no missing data):

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's

Data is normalized, and feature reduction done through PCA.

Results(MSE): 
  
    1. PCA, followed by Linear Regression:155.13
    2. PCA, followed by SVM: 50.9986
    3. Lasso: 156.35
    4. Neural Network: 21.76
