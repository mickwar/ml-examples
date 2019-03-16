import xgboost as xgb
import numpy as np

### Generate data
np.random.seed(0)
n = 5000
k = 5

# Train data
x_train = np.random.randn(n, k)
y_train = np.sin(x_train[:,0]) + np.cos(np.pi * x_train[:,1]) - x_train[:,2] ** 2 - x_train[:,3]
y_train = y_train.reshape((n, 1))

# Test set
m = 1000
x_test = np.random.randn(m, k)
y_test = np.sin(x_test[:,0]) + np.cos(np.pi * x_test[:,1]) - x_test[:,2] ** 2 - x_test[:,3]
y_test = y_test.reshape((m, 1))


dtrain = xgb.DMatrix(x_train, label = y_train)

