import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

### Generate data
np.random.seed(0)
n = 10000
k = 10

x = np.random.randn(n, k)
y = np.sin(x[:,0]) + np.cos(np.pi * x[:,1] + np.abs(x[:,2])) - x[:,3] ** 2 - x[:,4]
y = y.reshape((n, 1))

# Train and test sets
m = 1000
x_train = x[:(n-m),:]
y_train = y[:(n-m),:]
x_test = x[(n-m):,:]
y_test = y[(n-m):,:]


### Model training
dtrain = xgb.DMatrix(x_train, label = y_train)
dtest = xgb.DMatrix(x_test, label = y_test)
param = {'max_depth': 5, 'eta': 1, 'verbosity': 2, 'objective': 'reg:linear'}

num_round = 10
bst = xgb.train(param, dtrain, num_round)


### Predictions
y_pred = bst.predict(dtest).reshape((m,1))


### Plots
# Observed vs fitted
a = min(np.min(y_test), np.min(y_pred))
b = max(np.max(y_test), np.max(y_pred))
plt.clf()
plt.scatter(y_test, y_pred)
plt.plot([a,b],[a,b], color = (1,0.5,0))
plt.show(block = False)


# Importance
# I do my own plot here since xgb.plot_importance creates a new figure each time
# we plot, and it also sorts by the importance value (I want to sort by variable
# instead). Normally, xgb.plot_importance is fine, but in this example it's just
# easier to see that how the first five variables differ from the last five.
def myplot(importance_type = 'weight'):
    imp = bst.get_score(importance_type = importance_type)
    sort_key = sorted(imp.keys())
    sort_val = []
    for key in sorted(imp):
        sort_val.append(imp[key])
    sort_key.reverse()
    sort_val.reverse()
    plt.clf()
    plt.barh(sort_key, sort_val)
    plt.show(block = False)


myplot('weight')
myplot('gain')
myplot('cover')
myplot('total_gain')
myplot('total_cover')

