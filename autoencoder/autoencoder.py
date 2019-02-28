from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

### A
# Generate fake data (independent normals)
n = 2000
k = 40
x = np.random.randn(n, k)

# Set up model
l_input = Input(shape = (k,))
l_encode = Dense(units = int(k / 2), activation = 'relu', activity_regularizer = regularizers.l1(0.01))(l_input)
l_encode = Dense(units = int(k / 4), activation = 'relu')(l_encode)
l_decode = Dense(units = int(k / 4), activation = 'relu')(l_encode)
l_decode = Dense(units = int(k / 2), activation = 'sigmoid')(l_decode)
l_output = Dense(units = k, activation = 'linear')(l_decode)
model = Model(inputs = l_input, outputs = l_output)
model.compile(optimizer = 'adadelta', loss = 'mse')
history = model.fit(x = x, y = x, epochs = 500)

# Make predictions and compute MSE
pred = model.predict(x)
mse = np.mean(np.power(pred - x, 2), axis = 1)

# Plot loss over time
plt.clf()
plt.plot(history.history['loss'][50:])
plt.show(block = False)

# Plot "observed" vs "fitted" (some of variables)
plt.clf()
plt.scatter(np.sum(x, 1), np.sum(pred, 1))
plt.show(block = False)

# Scatter plot of MSE
plt.clf()
plt.scatter(range(n), mse)
plt.show(block = False)

# The plots are terrible since any attempt at dimension reduction will always
# result in data loss (since there are k independent variables).


### B
# Generate fake data (correlated normals)
n = 2000
k = 40

mu = np.zeros(k)
sigma = np.zeros((k, k))
rho = 0.99
for i in range(k):
    for j in range(k):
        sigma[i,j] = np.power(rho, np.abs(i-j))

x = np.random.multivariate_normal(mu, sigma, n)

# Set up model
l_input = Input(shape = (k,))
l_encode = Dense(units = int(k / 2), activation = 'relu', activity_regularizer = regularizers.l1(0.01))(l_input)
l_encode = Dense(units = int(k / 4), activation = 'relu')(l_encode)
l_decode = Dense(units = int(k / 4), activation = 'relu')(l_encode)
l_decode = Dense(units = int(k / 2), activation = 'relu')(l_decode)
l_output = Dense(units = k, activation = 'linear')(l_decode)
model = Model(inputs = l_input, outputs = l_output)
model.compile(optimizer = 'adadelta', loss = 'mse')
checkpointer = ModelCheckpoint(filepath = "model.h5", monitor = "loss", save_best_only = True)
history = model.fit(x = x, y = x, epochs = 2500, batch_size = 256, callbacks = [checkpointer])

# Make predictions and compute MSE
pred = model.predict(x)
mse = np.mean(np.power(pred - x, 2), axis = 1)

# Plot loss over time
plt.clf()
plt.plot(history.history['loss'][50:])
plt.show(block = False)

# Plot "observed" vs "fitted" (some of variables)
plt.clf()
plt.scatter(np.sum(x, 1), np.sum(pred, 1))
plt.plot([-100, 100], [-100, 100], color = (1,0,0))
plt.show(block = False)

# Scatter plot of MSE
plt.clf()
plt.scatter(range(n), mse)
plt.show(block = False)


