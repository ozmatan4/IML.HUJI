from __future__ import annotations
from typing import NoReturn
from importlib_metadata import re
# from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv

from IMLearn import BaseEstimator
import IMLearn

# import IMLearn.metrics.loss_functions


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            X = np.insert(X, 0, 1., axis= 1)
        self.coefs_ = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        self.fitted_ = True




    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        if self.fitted_:
            if self.include_intercept_:
                X = np.insert(X, 0, 1., axis= 1)           
            return X @ self.coefs_ 

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """

        if self.fitted_:
            return IMLearn.metrics.loss_functions.mean_square_error(y, self._predict)
        
        


# # TR = LinearRegression()
# # X = np.array([[9]])
# # X.reshape(3,3)
# # y = np.array([[3]])
# # TR.fit(X, y)
# lj = LinearRegression(True)
# x = np.array([[0,0,0], [1,0,9], [0,1,0]]).T
# print(np.shape(x))
# y =  np.array([[0,0.5,2]]).T

# print("x\n", x, "\n", "y\n", y)
# lj._fit(x,y)
# print("coefs:\n", lj.coefs_)