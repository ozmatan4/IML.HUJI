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
    print((uniVG.mu_, uniVG.var_))


    # Question 2 - Empirically showing sample mean is consistent
    
    Xarr = np.linspace(10,1000,100)
    secondUVG = UnivariateGaussian()
    samplesArr = np.array([np.abs(mu - secondUVG.fit(samples[:x]).mu_) for x in range(10, 1010, 10)])
 
    go.Figure(go.Scatter(x=Xarr, y=samplesArr, mode='markers+lines'),
              layout=go.Layout(
                  title=r"$\text{Q2 - distance from real mean}$",
                  xaxis_title="$m\\text{Samples number}$",
                  yaxis_title="$m\\text{distance between mu to mu hat}$",
                  height=300)).show()
    


    # # Question 3 - Plotting Empirical PDF of fitted mode
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
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    lLH_Array = np.zeros((f1.size, f3.size))
    for i, val1 in enumerate(f1):
        for j, val3 in enumerate(f3):
            lLH_Array[i, j] = MultivariateGaussian().log_likelihood(np.array([val1, 0, val3, 0]), cov, samplesArr) 

    go.Figure(data=go.Heatmap(
                   z=lLH_Array,
                   x=f1,
                   y=f3),
                   layout=go.Layout(
                   title=r"$\text{log likelihood}$",
                   xaxis_title="$\\text{f1}$",
                   yaxis_title="$\\text{f3}$",
                   height=600, width=600)).show()
    
   
    # Question 6 - Maximum likelihood
    theta = np.unravel_index(np.argmax(lLH_Array, axis=None), lLH_Array.shape)
    print(np.argmax(lLH_Array, axis=None))
    print("maxF1: ", f1[theta[0]], "maxF3: ",  f3[theta[1]])



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()  
    test_multivariate_gaussian()

