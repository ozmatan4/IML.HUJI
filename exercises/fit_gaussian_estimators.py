# import os
# print(os.getcwd(), "\n")
from turtle import distance
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)
    uniVG = UnivariateGaussian()
    uniVG.fit(samples)


    # Question 2 - Empirically showing sample mean is consistent
    Xarr = np.array([x for x in range(10, 1010, 10)])
    uniVarArr = np.array([UnivariateGaussian().fit(np.random.normal(mu, sigma, x)) for x in range(10, 1010, 10)])
    samplesArr = np.array([np.abs(mu - x.mu_) for x in uniVarArr])
    # go.Figure()
    plt.plot(Xarr, samplesArr)
    plt.title("Q2 - distance from real mean")
    plt.xlabel("Samples number")
    plt.ylabel("distance between mu to mu hat")

    plt.show()
    


    # # Question 3 - Plotting Empirical PDF of fitted mode
    # go.Figure()
    sortSample = np.sort(samples)
    pdfArr = uniVG.pdf(sortSample)
    indexArr = range(0, 1000, 1)
    plt.scatter(sortSample, pdfArr)
    plt.title("Q3 - pdf of the samples")
    plt.xlabel("Sample index")
    plt.ylabel("pdf of the samples")

    plt.show()
    



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
