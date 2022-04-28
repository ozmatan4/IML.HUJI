from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, return_counts = np.unique(y, return_counts=True)
        self.pi_ = return_counts/y

        self.mu_ = np.zeros(np.size(self.classes_), X.shape(1))
        self.cov_ = np.zeros(X.shape(1), X.shape(1))

        for i, cl in enumerate(self.classes_):
            self.mu_[i] = X[y == cl].mean(axis=0)
            self.vars_[i] = X[y == cl].var(axis=0)



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
        return np.argmax(self.likelihood(X), axis=1)
        

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # z = np.sqrt(np.power(2*np.pi, X.shape(1))*np.linalg.det(self.cov_))
        # expArr = np.array([np.exp((0.5) * np.diag((x - self.mu_) @ self._cov_inv @ (x - self.mu_).T)) for x in X])
        # likelihoods = (1/z)*expArr
        # return likelihoods


        likelihoods = np.zeros((X.shape[0], self.classes_.size))
        for i in range(self.classes_.size):
            z = np.sqrt(np.power(2*np.pi, X.shape[1]) * np.linalg.det(self.vars_[i, :, :]))
            likelihoods[:, i] = (1/z) * np.exp(-0.5 * np.diag((X - self.mu_[i]) @ np.linalg.inv(self.vars_[i, :, :]) @ (X - self.mu_[i]).T))
        return likelihoods



    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        raise NotImplementedError()
