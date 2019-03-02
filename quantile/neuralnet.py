### Neural network using Keras that returns an estimated mean and variance for
### each observation. (Technically, the mean and log variance based on a normal
### negative log-likelihood loss function).

import numpy as np
import matplotlib.pyplot as plt
import sobol_seq
from scipy.stats import norm
from keras.models import Model
from keras.layers import Dense, Activation, Input, Dropout
from keras import backend as K
from keras import optimizers
from tensorflow import lgamma


### For splitting into train and test sets
def get_train_test_inds(n, train_proportion):
    train_n = round(n * train_proportion)
    full_ind = np.random.permutation(n)
    train_ind = full_ind[:train_n]
    test_ind = full_ind[train_n:]
    return train_ind, test_ind

### Custom loss functions using negative log-likelihood
def make_quantile(q):
    q = np.array(q)
    m = len(q)
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q*e, (q-1)*e), axis = -1)
    return loss


### Generate data, simple
np.random.seed(1)
n = 62000
k = 10
X = np.random.rand(n, k) * 3

xsum = np.sum(X[:,0:4], axis = 1)
xprod = np.sum(X[:,2:6], axis = 1)

Y = np.random.normal(loc = xsum, scale = xprod / 10.)

plt.clf()
plt.scatter(xsum, Y)
plt.show(block = False)

plt.clf()
plt.scatter(xprod, Y)
plt.show(block = False)

# Reshape Y (necessary?)
#Y = Y.reshape(Y.shape[0], 1)

### Separate into training and test sets
np.random.seed(123)
train_ind, test_ind = get_train_test_inds(n, 0.9)

X_train = X[train_ind,]
X_test = X[test_ind,]

Y_train = Y[train_ind]
Y_test = Y[test_ind]

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
qloss = make_quantile(q)

### Build the neural network architecture
inputs = Input(shape=(X_train.shape[1],), name = 'input')
layer = Dense(units = 64, activation = 'relu', name = 'hidden_layer_1')(inputs)
layer = Dropout(0.2)(layer)
layer = Dense(units = 64, activation = 'relu', name = 'hidden_layer_2')(layer)
layer = Dropout(0.2)(layer)
output = Dense(units = len(q), activation = 'linear', name = 'out')(layer)
model = Model(inputs = inputs, outputs = output, name = 'mean_var')

model.compile(
    optimizer = 'adagrad',
    loss = qloss,
    metrics = ['accuracy'])

model.fit(x = {'input' : X_train},
    y = {'out' : Y_train},
    epochs = 20, batch_size = 128, validation_split = 0.1)

### Make predictions
Y_pred = model.predict(x = X_test)
error = np.zeros((len(test_ind), len(q)))
truth = np.zeros((len(test_ind), len(q)))
for i in range(len(q)):
    truth[:,i] = xsum[test_ind] + xprod[test_ind] / 10. * norm.isf(1-q[i])

Y_pred
truth

i = 8
np.sqrt(np.mean((Y_pred[:,i] - truth[:,i])**2))

np.sqrt(np.mean((Y_pred[:,0] - (xsum[test_ind] + error[:,0]))**2))
np.sqrt(np.mean((Y_pred[:,1] - (xsum[test_ind] + error[:,1]))**2))
np.sqrt(np.mean((Y_pred[:,2] - (xsum[test_ind] + error[:,2]))**2))

np.sqrt(np.mean((Y_pred[:,0] - (Y_test + error[:,0]))**2))
np.sqrt(np.mean((Y_pred[:,1] - (Y_test + error[:,1]))**2))
np.sqrt(np.mean((Y_pred[:,2] - (Y_test + error[:,2]))**2))

Y_pred[:,1]
Y_test




# Compare predicted mean with predicted var
plt.clf();
plt.plot(Y_pred_in[:,0], Y_pred_in[:,1], marker='s', linestyle='',
color = (0,0.5,0,0.3))
plt.plot(Y_pred[:,0], Y_pred[:,1], marker='s', linestyle='', color =
(0,0,0.8,0.5))
plt.show(block = False);

# Compare predicted mean with true mean
plt.clf();
a,b = Y_pred_in[:,0].min(), Y_pred_in[:,0].max()
plt.plot([a,b],[a,b], linestyle = '-', color = (1,0,0,0.5))
plt.plot(Y_pred_in[:,0], xprod[train_ind,], marker='s', linestyle='',
color = (0,0.5,0,0.3))
plt.plot(Y_pred[:,0], xprod[test_ind,], marker='s', linestyle='',
color = (0,0,0.8,0.3))
plt.show(block = False);

# Compare predicted sd with true sd
plt.clf();
a,b = 0,1
plt.plot([a,b],[a,b], linestyle = '-', color = (1,0,0,0.5))
plt.plot(np.sqrt(Y_pred_in[:,1]), xsum[train_ind,] / 10., marker='s',
linestyle='', color = (0,0.5,0,0.3))
plt.plot(np.sqrt(Y_pred[:,1]), xsum[test_ind,] / 10., marker='s',
linestyle='', color = (0,0,0.8,0.3))
plt.show(block = False);

# See if the numbers make sense
np.mean(Y_pred_in, 0)
np.mean(Y_pred, 0)
np.var(Y_pred_in, 0)
np.var(Y_pred, 0)

np.mean(Y_train, 0)
np.mean(Y_test, 0)
np.var(Y_train, 0)
np.var(Y_test, 0)

### Plot test samples with -/+ 2 SD bounds
plt.clf();
a,b = Y.min(), Y.max()
plt.plot([a,b],[a,b], linestyle = '-', color = (1,0,0,0.5))
plt.plot(Y_test, Y_pred[:,0] + 2*Y_pred[:,1], marker='s',
linestyle='', color=(1,0,0,0.3))
plt.plot(Y_test, Y_pred[:,0] - 2*Y_pred[:,1], marker='s',
linestyle='', color=(1,0,0,0.3))
#plt.plot(Y_train[:,0], Y_pred_in[:,0], marker='s', linestyle='',
color = (0,0.5,0,0.3))
plt.plot(Y_test, Y_pred[:,0], marker='s', linestyle='', color = (0,0,0.8,0.5))
plt.show(block = False);
