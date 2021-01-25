---
layout: post

title: Use line_profiler to profile your python code

tip-number: 34
tip-username: dzlab
tip-username-profile: https://github.com/dzlab
tip-tldr: Learn a how to profile and debug performance issue of your python program using line_profiler.
tip-writer-support: https://www.patreon.com/dzlab

categories:
    - en
    - more
---

If your program is slow then before anything it is important to identify the bottelneck or where most of the overhead is coming from. In fact, localting the overhead helps a lot in determining what to do next to improve performance.

In python we can use [line_profiler](https://github.com/pyutils/line_profiler) package to identity where exactly most of time is spent.

```
pip install line-profiler
```

As an example, we will try to profile sklearn `fit` method which is used to train a linear regression model.

First, let's create some toy data for training
```python
from sklearn.datasets import make_regression

X, y = make_regression(random_state=13)
```

Second, create the model to train
```python

from sklearn.linear_model import LinearRegression
est = LinearRegression()
```

Finally, initilize the `LineProfiler` by wrapping the `fit` methond from the Linear Regression model and then running the profile.
```python
from line_profiler import LineProfiler

lp = LineProfiler(est.fit)
print("Run on a single row")
lp.run("est.fit(X, y)")
lp.print_stats()
```

The output of the profiling is very detailed with information about timing and number of hits for every line of code in the profiled program. In the case of `fit` method, it will look like this
```
Run on a single row
Timer unit: 1e-06 s

Total time: 0.022127 s
File: /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_base.py
Function: fit at line 467

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   467                                               def fit(self, X, y, sample_weight=None):
   468                                                   """
   469                                                   Fit linear model.
   470                                           
   471                                                   Parameters
   472                                                   ----------
   473                                                   X : {array-like, sparse matrix} of shape (n_samples, n_features)
   474                                                       Training data
   475                                           
   476                                                   y : array-like of shape (n_samples,) or (n_samples, n_targets)
   477                                                       Target values. Will be cast to X's dtype if necessary
   478                                           
   479                                                   sample_weight : array-like of shape (n_samples,), default=None
   480                                                       Individual weights for each sample
   481                                           
   482                                                       .. versionadded:: 0.17
   483                                                          parameter *sample_weight* support to LinearRegression.
   484                                           
   485                                                   Returns
   486                                                   -------
   487                                                   self : returns an instance of self.
   488                                                   """
   489                                           
   490         1         10.0     10.0      0.0          n_jobs_ = self.n_jobs
   491         1          4.0      4.0      0.0          X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
   492         1       1515.0   1515.0      6.8                           y_numeric=True, multi_output=True)
   493                                           
   494         1          4.0      4.0      0.0          if sample_weight is not None:
   495                                                       sample_weight = _check_sample_weight(sample_weight, X,
   496                                                                                            dtype=X.dtype)
   497                                           
   498         1          5.0      5.0      0.0          X, y, X_offset, y_offset, X_scale = self._preprocess_data(
   499         1          4.0      4.0      0.0              X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
   500         1          3.0      3.0      0.0              copy=self.copy_X, sample_weight=sample_weight,
   501         1       1229.0   1229.0      5.6              return_mean=True)
   502                                           
   503         1          4.0      4.0      0.0          if sample_weight is not None:
   504                                                       # Sample weight can be implemented via a simple rescaling.
   505                                                       X, y = _rescale_data(X, y, sample_weight)
   506                                           
   507         1          7.0      7.0      0.0          if sp.issparse(X):
   508                                                       X_offset_scale = X_offset / X_scale
   509                                           
   510                                                       def matvec(b):
   511                                                           return X.dot(b) - b.dot(X_offset_scale)
   512                                           
   513                                                       def rmatvec(b):
   514                                                           return X.T.dot(b) - X_offset_scale * np.sum(b)
   515                                           
   516                                                       X_centered = sparse.linalg.LinearOperator(shape=X.shape,
   517                                                                                                 matvec=matvec,
   518                                                                                                 rmatvec=rmatvec)
   519                                           
   520                                                       if y.ndim < 2:
   521                                                           out = sparse_lsqr(X_centered, y)
   522                                                           self.coef_ = out[0]
   523                                                           self._residues = out[3]
   524                                                       else:
   525                                                           # sparse_lstsq cannot handle y with shape (M, K)
   526                                                           outs = Parallel(n_jobs=n_jobs_)(
   527                                                               delayed(sparse_lsqr)(X_centered, y[:, j].ravel())
   528                                                               for j in range(y.shape[1]))
   529                                                           self.coef_ = np.vstack([out[0] for out in outs])
   530                                                           self._residues = np.vstack([out[3] for out in outs])
   531                                                   else:
   532                                                       self.coef_, self._residues, self.rank_, self.singular_ = \
   533         1      19208.0  19208.0     86.8                  linalg.lstsq(X, y)
   534         1          8.0      8.0      0.0              self.coef_ = self.coef_.T
   535                                           
   536         1          5.0      5.0      0.0          if y.ndim == 1:
   537         1         42.0     42.0      0.2              self.coef_ = np.ravel(self.coef_)
   538         1         76.0     76.0      0.3          self._set_intercept(X_offset, y_offset, X_scale)
   539         1          3.0      3.0      0.0          return self
```

With such output it is easy to locate where most of the time was spent, in this case there is a considerable amout of time spent in safety checks that sklearn usually do for instance to avoid divisions by zero.
```
   491         1          4.0      4.0      0.0          X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
   492         1       1515.0   1515.0      6.8                           y_numeric=True, multi_output=True)
```

Then, another bottlenck happens in data preprocessing
```
   498         1          5.0      5.0      0.0          X, y, X_offset, y_offset, X_scale = self._preprocess_data(
   499         1          4.0      4.0      0.0              X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
   500         1          3.0      3.0      0.0              copy=self.copy_X, sample_weight=sample_weight,
   501         1       1229.0   1229.0      5.6              return_mean=True)
```

Finally, we can see that most of the time was spent in the actual training which seems that sklearn used numpy's [lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
```
   532                                                       self.coef_, self._residues, self.rank_, self.singular_ = \
   533         1      19208.0  19208.0     86.8                  linalg.lstsq(X, y)
```

In this toy profiling example, we have identified the overhead inside sklearn's `fit` as due to safety checks and data preprocessing. This means that we can go faster if we avoid those two steps (e.g. we know the data is perfect and does not need checks or normalization) by directly using numpy's `lstsq`.