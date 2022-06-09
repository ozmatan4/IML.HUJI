from __future__ import annotations
from turtle import color
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    def f(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.random.uniform(-1.2, 2, n_samples)
    epsilon = np.random.normal(0, noise, n_samples)
    y = f(X) + epsilon
    xDF, yDF = pd.DataFrame(X), pd.Series(y)
    [xTrain, yTrain, xTest, yTest] = split_train_test(xDF, yDF, 2/3)
    
    xTrain, xTest = np.array(xTrain.squeeze()), np.array(xTest.squeeze())
    yTrain, yTest = np.array(yTrain), np.array(yTest)

    fig = go.Figure([go.Scatter(x=X, y=f(X), mode="markers", name="True-Noiseless model", marker_color="black"),
                     go.Scatter(x=xTrain, y=yTrain, mode="markers", name="Train model", marker_color="red"),
                     go.Scatter(x=xTest, y=yTest, mode="markers", name="Test model", marker_color="blue")])
    fig.update_layout(title=f"3 plots, Noiseless-black, Train-red, Test-blue, Num Samples: {n_samples},  noise: {noise}", xaxis_title="X", yaxis_title="y")
    # fig.write_image(f"polynomial.model.selection.{n_samples}.samples.and.{noise}.noise.png")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    K = 11
    trainMSE_array, validationMSE_array = [], []
    for k in range(K):
        trainMSE, validationMSE = cross_validate(PolynomialFitting(k), xTrain, yTrain, mean_square_error, 5)
        trainMSE_array.append(trainMSE)
        validationMSE_array.append(validationMSE)
    
    kArr = np.arange(K)
    fig = go.Figure([go.Scatter(x=kArr, y=trainMSE_array, mode="markers + lines", name="Train errors"),
                       go.Scatter(x=kArr, y=validationMSE_array, mode="markers + lines", name="Validation errors")])
    fig.update_layout(title=f"Train and Validation MSE for 5-Fold CV, and K=10 - Noise = {noise}, Samples number = {n_samples}",
                        xaxis_title="K", yaxis_title="MSE Value")
    fig.show()



    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    kStar = validationMSE_array.index(min(validationMSE_array))
    kStarModel = PolynomialFitting(kStar)
    kStarModel.fit(xTrain, yTrain)
    print(f"samples amount:{n_samples}, noise:{noise}")
    print("k* = ", kStar)
    print("test error = ", round(mean_square_error(yTest, kStarModel.predict(xTest)), 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    xTrain, yTrain = X[:n_samples], y[:n_samples]
    xTest, yTest = X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdaRange = np.linspace(0.0001, 0.7, n_evaluations)

    ridgeTrainErrors, ridgeValidationErrors = [], []
    for lam in lambdaRange:

        curRidge = RidgeRegression(lam)
        trainResult, validationResult = cross_validate(curRidge, xTrain, yTrain, mean_square_error, 5)

        ridgeTrainErrors.append(trainResult)
        ridgeValidationErrors.append(validationResult)

    figRidge = go.Figure([go.Scatter(x=lambdaRange, y=ridgeTrainErrors, mode="lines", name="Training Errors"),
                           go.Scatter(x=lambdaRange, y=ridgeValidationErrors, mode="lines", name="Validation Errors")])
    figRidge.update_layout(title=f"Ridge errors for {n_evaluations} lambdas in range 0.0001, 0.7, and {n_samples} Samples",
                            xaxis_title="Lambdas", yaxis_title="Error").show()


    lassoTrainErrors, lassoValidationErrors = [], []

    for lam in lambdaRange:

        curLasso = Lasso(alpha=lam)
        trainResult, validationResult = cross_validate(curLasso, xTrain, yTrain, mean_square_error, 5)
                
        lassoTrainErrors.append(trainResult)
        lassoValidationErrors.append(validationResult)


    figLasso = go.Figure([go.Scatter(x=lambdaRange, y=lassoTrainErrors, mode="lines", name="Training Errors"),
                           go.Scatter(x=lambdaRange, y=lassoValidationErrors, mode="lines", name="Validation Errors")])

    figLasso.update_layout(title=f"Lasso errors for {n_evaluations} lambdas in range 0.0001, 0.7, and {n_samples} Samples",
                            xaxis_title="Lambdas", yaxis_title="Error").show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridgeMinInd = np.argmin(np.array(ridgeValidationErrors))
    selectedRidgeLambda = lambdaRange[ridgeMinInd]
    ridge = RidgeRegression(ridgeMinInd)
    ridge.fit(xTrain, yTrain)


    lassoMinInd = np.argmin(np.array(lassoValidationErrors))
    selectedLassoLambda = lambdaRange[lassoMinInd]
    lasso = Lasso(lassoMinInd)
    lasso.fit(xTrain, yTrain)

    lineaRegression = LinearRegression().fit(xTrain, yTrain)
    ridgeError = mean_square_error(yTest, ridge.predict(xTest))
    lassoError = mean_square_error(yTest, lasso.predict(xTest))
    linearRigError = mean_square_error(yTest, lineaRegression.predict(xTest))
    print("Best Ridge Regularization: " + str(selectedRidgeLambda))
    print("Best Lasso Regularization: " + str(selectedLassoLambda))
    print("Test Error of fitted Ridge: " + str(ridgeError))
    print("Test Error of fitted Lasso: " + str(lassoError))
    print("Test Error of fitted Linear Regression: " + str(linearRigError))






if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()

