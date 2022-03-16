# import os
# print(os.getcwd(), "\n")
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)
    uniVG = UnivariateGaussian()
    uniVG.fit(samples)

    print((uniVG.mu_, uniVG.var_))


    # Question 2 - Empirically showing sample mean is consistent
    samplesArr = np.array({x:(np.abs(mu - uniVG.fit(np.random.normal(mu, sigma, x)).mu_)) for x in range(10, 1000, 10)})
    # go.Figure()
    print(samplesArr)
    


    # # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


# def main():
#     test_univariate_gaussian()



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
