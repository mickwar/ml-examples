### Neural network using Keras that returns an estimated mean and variance for
### each observation. (Technically, the mean and log variance based on a normal
### negative log-likelihood loss function).

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Activation, Input, Dropout, BatchNormalization
from keras import backend as K

from scipy.stats import norm
from keras import optimizers
from tensorflow import lgamma


### Custom loss functions using negative log-likelihood
def make_quantile(q):
    q = np.array(q)
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q*e, (q-1)*e), axis = -1)
    return loss


### Generate data
np.random.seed(1)
n = 50000
k = 10
x = np.random.rand(n, k) * 3

xsum = np.sum(x[:,0:4], axis = 1)
xprod = np.sum(x[:,2:6], axis = 1)

y = np.random.normal(loc = xsum, scale = xprod / 10.)

plt.clf()
_ = plt.scatter(xsum, y)
plt.show(block = False)

plt.clf()
_ = plt.scatter(xprod, y)
plt.show(block = False)

### Separate into training and test sets
m = int(np.round(n * 0.9))
x_train = x[:m,:]
y_train = y[:m]

x_test = x[m:,:]
y_test = y[m:]

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
qloss = make_quantile(q)

### Build the neural network architecture
inputs = Input(shape=(k,))
layer = Dense(units = 64, activation = 'relu')(inputs)
layer = Dropout(0.2)(layer)
layer = Dense(units = 64, activation = 'relu')(layer)
layer = Dropout(0.2)(layer)
output = Dense(units = len(q), activation = 'linear')(layer)
model = Model(inputs = inputs, outputs = output)

model.compile(optimizer = 'adagrad', loss = qloss, metrics = ['accuracy'])

model.fit(x = x_train, y = y_train, epochs = 20, batch_size = 128, validation_split = 0.1)

### Make predictions
Y_pred = model.predict(x = x_test)
error = np.zeros((n-m, len(q)))
truth = np.zeros((n-m, len(q)))
for i in range(len(q)):
    truth[:,i] = xsum[m:] + xprod[m:] / 10. * norm.isf(1-q[i])

plt.clf()
plt.scatter(xsum[m:], y[m:])
plt.scatter(xsum[m:], Y_pred[:,4])
plt.scatter(xsum[m:], Y_pred[:,0])
plt.scatter(xsum[m:], Y_pred[:,8])
plt.show(block = False)

plt.plot(Y_pred[:,0], Y_pred_in[:,1], marker='s', linestyle='',
color = (0,0.5,0,0.3))
plt.plot(Y_pred[:,0], Y_pred[:,1], marker='s', linestyle='', color =
(0,0,0.8,0.5))
plt.show(block = False);




### Generate data
np.random.seed(1)
n = 50000
k = 1
x = np.random.rand(n, k) * 4 * np.pi

y = np.random.normal(loc = np.cos(x), scale = x / 10.)

plt.clf()
_ = plt.scatter(x, y)
plt.show(block = False)

### Separate into training and test sets
m = int(np.round(n * 0.9))
x_train = x[:m,:]
y_train = y[:m]

x_test = x[m:,:]
y_test = y[m:]

q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
qloss = make_quantile(q)

### Build the neural network architecture
inputs = Input(shape=(k,))
layer = Dense(units = 128, activation = 'relu')(inputs)
layer = BatchNormalization()(layer)
layer = Dropout(0.2)(layer)
layer = Dense(units = 128, activation = 'relu')(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.2)(layer)
layer = Dense(units = 128, activation = 'relu')(layer)
layer = BatchNormalization()(layer)
layer = Dropout(0.2)(layer)
output = Dense(units = len(q), activation = 'linear')(layer)
model = Model(inputs = inputs, outputs = output)
model.compile(optimizer = 'adadelta', loss = qloss, metrics = ['accuracy'])
model.fit(x = x_train, y = y_train, epochs = 100, batch_size = 128, validation_split = 0.1)

### Make predictions
Y_pred = model.predict(x = x_test)

plt.clf()
plt.scatter(x[m:], y[m:])
plt.scatter(x[m:], Y_pred[:,4])
plt.scatter(x[m:], Y_pred[:,0])
plt.scatter(x[m:], Y_pred[:,8])
plt.show(block = False)


#error = np.zeros((n-m, len(q)))
truth = np.zeros((n-m, len(q)))
for i in range(len(q)):
    truth[:,i,None] = np.cos(x[m:,:]) + x[m:,:] / 10. * norm.isf(1-q[i])

plt.clf()
plt.scatter(x[m:], Y_pred[:,4], color = (1,0,0))
plt.scatter(x[m:], Y_pred[:,0], color = (1,0,0))
plt.scatter(x[m:], Y_pred[:,8], color = (1,0,0))
plt.scatter(x[m:], truth[:,4], color = (0,0,1))
plt.scatter(x[m:], truth[:,0], color = (0,0,1))
plt.scatter(x[m:], truth[:,8], color = (0,0,1))
plt.show(block = False)
