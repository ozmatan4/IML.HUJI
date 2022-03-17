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
    print("(", uniVG.mu_, uniVG.var_, ")")


    # Question 2 - Empirically showing sample mean is consistent
    
    Xarr = np.array([x for x in range(10, 1010, 10)])
    secondUVG = UnivariateGaussian()
    samplesArr = np.array([np.abs(mu - secondUVG.fit(np.random.normal(mu, sigma, x)).mu_) for x in range(10, 1010, 10)])
 
    go.Figure(go.Scatter(x=Xarr, y=samplesArr, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{Q2 - distance from real mean}$",
                  xaxis_title="$m\\text{Samples number}$",
                  yaxis_title="$m\\text{distance between mu to mu hat}$",
                  height=300)).show()
    


    # # Question 3 - Plotting Empirical PDF of fitted mode
    # go.Figure()
    sortSample = np.sort(samples)
    pdfArr = uniVG.pdf(sortSample)
    indexArr = range(0, 1000, 1)
    go.Figure(go.Scatter(x=sortSample, y=pdfArr, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{Q3 - pdf of the samples}$",
                  xaxis_title="$m\\text{Sample index}$",
                  yaxis_title="$m\\text{pdf of the samples}$",
                  height=300)).show()
    


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.matrix([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    samplesArr = np.random.multivariate_normal(mu, cov, 1000)

    multiVG = MultivariateGaussian()
    multiVG = multiVG.fit(samplesArr)
    print(multiVG.mu_)
    print(multiVG.cov_)


    # Question 5 - Likelihood evaluation
    # muArr = []
    # for f1 in np.linspace(-10, 10, 200):
    #     for f3 in np.linspace(-10, 10, 200):
    #         muArr.append([f1, 0, f3, 0])

    # print(muArr)


    # fig = go.Figure(data=go.Heatmap(
    #                z=[[1, None, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
    #                x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    #                y=['Morning', 'Afternoon', 'Evening'],
    #                hoverongaps = False))
    # fig.show()
   


    # Question 6 - Maximum likelihood
    # raise NotImplementedError()




if __name__ == '__main__':
    a= np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1, -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print(np.sum(a)/a.size)
    print(a.var())
    
    print(UnivariateGaussian().log_likelihood(a.mean(),a.var(), a))


    np.random.seed(0)
    # test_univariate_gaussian()
    # test_multivariate_gaussian()

