## Quantile regression with a neural network

### Loss function

Doing quantile regression with a neural network is as easy as choosing the appropriate loss function. The loss function for quantile *q* of a single point can be expressed as

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;L_q(y,&space;\hat{y})&space;=&space;(y-\hat{y})(q-1_{(y-\hat{y}<0)})&space;&=&space;\begin{cases}&space;q&space;|y-\hat{y}|&space;&&space;\mathrm{if}&space;\;&space;y-\hat{y}&space;\ge&space;0&space;\\&space;(1&space;-&space;q)&space;|y-\hat{y}|&space;&&space;\mathrm{if}&space;\;&space;y-\hat{y}&space;<&space;0)&space;\end{cases}&space;\\&space;&=&space;\max\left\{q(y-\hat{y}),&space;-(1-q)(y-\hat{y})&space;\right\}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;L_q(y,&space;\hat{y})&space;=&space;(y-\hat{y})(q-1_{(y-\hat{y}<0)})&space;&=&space;\begin{cases}&space;q&space;|y-\hat{y}|&space;&&space;\mathrm{if}&space;\;&space;y-\hat{y}&space;\ge&space;0&space;\\&space;(1&space;-&space;q)&space;|y-\hat{y}|&space;&&space;\mathrm{if}&space;\;&space;y-\hat{y}&space;<&space;0)&space;\end{cases}&space;\\&space;&=&space;\max\left\{q(y-\hat{y}),&space;-(1-q)(y-\hat{y})&space;\right\}&space;\end{align*}" title="\begin{align*} L_q(y, \hat{y}) = (y-\hat{y})(q-1_{(y-\hat{y}<0)}) &= \begin{cases} q |y-\hat{y}| & \mathrm{if} \; y-\hat{y} \ge 0 \\ (1 - q) |y-\hat{y}| & \mathrm{if} \; y-\hat{y} < 0) \end{cases} \\ &= \max\left\{q(y-\hat{y}), -(1-q)(y-\hat{y}) \right\} \end{align*}" /></a>

where *y* is the true response label and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y}" title="\hat{y}" /></a> is the estimated quantile.

Here is an image of the loss function for several quantiles.
Notice for *q=0.1* that we are punished more for overestimating the truth (the negative side) than underestimating (the positive side).
And the opposite is true for *q=0.7*.

![](loss.png)

This function can be implemented in Python with:

```python
from keras import backend as K
import numpy as np

def make_quantile(q):
    q = np.array(q)
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q*e, (q-1)*e), axis = -1) 
    return loss
```

which can take in a list of quantiles and will return a length-*q* loss function, allowing us to model several quantiles concurrently.

Some references:
- [A helpful stack exchange question](https://stats.stackexchange.com/questions/251600/quantile-regression-loss-function/252029)
- [Deep quantile regression](https://towardsdatascience.com/deep-quantile-regression-c85481548b5a) (which notes the equivalence with the asymmetric Laplace likelihood)
