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
        self.pi_ = return_counts/np.size(y)

        if len(X.shape)==1:
            self.mu_ = np.zeros((np.size(self.classes_), 1))
            self.vars_ = np.zeros((self.classes_.size, 1))

        else:
            self.mu_ = np.zeros((np.size(self.classes_), X.shape[1]))
            self.vars_ = np.zeros((self.classes_.size, X.shape[1]))

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

        likelihoods = np.zeros((X.shape[0], self.classes_.size))
        for i in range(self.classes_.size):
            varMat = self.vars_[i] * np.identity(X.shape[1])
            z = np.sqrt(np.power(2*np.pi, X.shape[1]) * np.linalg.det(varMat))
            inv_var = np.linalg.inv(varMat)
            likelihoods[:, i] = (1/z) * np.exp(-0.5 * np.diag((X - self.mu_[i]) @ inv_var @ (X - self.mu_[i]).T))
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
        misclassification_error(y, self._predict(X))    




