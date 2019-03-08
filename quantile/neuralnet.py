### Neural network using Keras that returns an estimated mean and variance for
### each observation. (Technically, the mean and log variance based on a normal
### negative log-likelihood loss function).

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras import backend as K
from scipy.stats import norm


### Function for making quantile loss functions (given list of quantiles q)
### This is equivalent to using the negative log-likelihood of the asymmetric Laplace
def make_quantile(q):
    q = np.array(q)
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q*e, (q-1)*e), axis = -1)
    return loss


### Generate sinusoidal data
np.random.seed(1)
n = 1000
k = 1
x = np.random.rand(n, k) * 4 * np.pi
y = np.random.normal(loc = np.cos(x), scale = x / 10.)

plt.clf()
_ = plt.scatter(x, y)
plt.show(block = False)

### Create the loss function at three different quantiles
### Our bounds are akin to 80% prediction intervals (0.9 - 0.1 = 0.8)
q = [0.1, 0.5, 0.9]
qloss = make_quantile(q)

### Build the neural network architecture
inputs = Input(shape=(k,))
layer = Dense(units = 32, activation = 'relu')(inputs)
layer = Dense(units = 32, activation = 'relu')(layer)
output = Dense(units = len(q), activation = 'linear')(layer)
model = Model(inputs = inputs, outputs = output)
model.compile(optimizer = 'adadelta', loss = qloss, metrics = ['accuracy'])
model.fit(x = x, y = y, epochs = 2000)

### Make predictions
x_test = np.linspace(0, 4*np.pi, 100)
Y_pred = model.predict(x = x_test)

truth = np.zeros((x_test.shape[0], len(q)))
for i in range(len(q)):
    truth[:,i] = np.cos(x_test) + x_test / 10. * norm.isf(1-q[i])

plt.clf()
_ = plt.scatter(x, y, label = 'Data')
_ = plt.plot(x_test, truth[:,0], color = (0,0,1), label = 'True quantiles')
_ = plt.plot(x_test, truth[:,1], color = (0,0,1))
_ = plt.plot(x_test, truth[:,2], color = (0,0,1))
_ = plt.plot(x_test, Y_pred[:,0], color = (1,0,0,0.8), lw = 3, label = 'Estimated quantiles')
_ = plt.plot(x_test, Y_pred[:,1], color = (1,0,0,0.8), lw = 3)
_ = plt.plot(x_test, Y_pred[:,2], color = (1,0,0,0.8), lw = 3)
_ = plt.legend(fontsize = 12)
plt.savefig('output.png', bbox_inches = 'tight')
# plt.show(block = False)

