from distutils.log import error
from turtle import width
from IMLearn.utils import split_train_test
from matplotlib.pyplot import ylabel
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.drop(columns=["id", "lat", "long"], axis=1, inplace=True)
    df.dropna()

    nonNegTitle  = ["price", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]
    posTitle = ["bedrooms", "bathrooms", "sqft_living", "waterfront", "floors", "sqft_basement", "yr_renovated"]

    for title in nonNegTitle:
        df = df[df[title] > 0]
    for title in posTitle:
        df = df[df[title] >= 0]

    df = df[df["view"].isin(range(5))]
    df = df[df["condition"].isin(range(1, 6))]
    df = df[df["grade"].isin(range(1, 14))]

    df.loc[df.yr_renovated == 0, "yr_renovated"] = df["yr_built"] 


    df["date"] = (pd.to_datetime(df["date"]).dt.year - 2000)

    df["yr_built"] = df["yr_built"] - df["yr_built"].min()
    
    df["yr_renovated"] = df["yr_renovated"] - df["yr_renovated"].min()

    allTitle = nonNegTitle + posTitle + ["view", "condition", "grade"]

    # for title in allTitle:


    df = df[df["price"] < 1300000]
    df = df[df["bedrooms"] < 8]
    df = df[df["bathrooms"] < 5]
    df = df[df["sqft_living"] < 5500]
    df = df[df["sqft_lot"] < 10000]
    df = df[df["floors"] < 5]
    df = df[df["sqft_above"] < 6000]
    df = df[df["sqft_basement"] < 1700]
    df = df[df["sqft_living15"] < 5000]
    df = df[df["sqft_lot15"] < 60000]

    df = pd.get_dummies(df, prefix="zip_is_", columns=["zipcode"])


    y = df["price"]
    x = df.drop(columns=["price"])    



    return x, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """


    ylabelSD = np.std(y)

    for title in X:
        curruntCol = X[title].values
        curTitleSD = np.std(curruntCol)
        covCurY = np.cov(curruntCol, y, bias=True)[0][1]
        P_Correlation = covCurY / (ylabelSD*curTitleSD)

        plot = go.Figure(go.Scatter(x=curruntCol, y=y, mode='markers'),
                layout=go.Layout(
                    title="the correlations of the prices with " + str(title) + "is: " + str(P_Correlation),
                    xaxis_title=str(title),
                    yaxis_title="Prices- y lable"))

        plot.write_image(output_path + "/" + title + ".png")







if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data("/Users/ozmatan4/Documents/Studies/huji/year3/semesterB/IML/exercises/EX1/IML.HUJI/datasets/house_prices.csv")

    # x.to_csv("/Users/ozmatan4/Documents/Studies/huji/year3/semesterB/IML/exercises/EX1/IML.HUJI/datasets/house_prices2.csv")


    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(x, y, "/Users/ozmatan4/Documents/Studies/huji/year3/semesterB/IML/exercises/ex2_images")

    # Question 3 - Split samples into training- and testing sets.
    [train_x, train_y, test_x, test_y] = split_train_test(x, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    
    meanLossArr = []
    varLossArr = []

    lR = LinearRegression()

    allSamples = train_x
    allSamples["price"] = train_y
    pArr= np.arange(10,101,1)
    for p in pArr:
        tempLos = []
        for repeat in range(10):
            tempAllSamples= allSamples.sample(frac=(p/100))
            tempTrainY = tempAllSamples["price"]
            tempTrainX  = tempAllSamples.drop(columns=["price"])
            lR.fit(tempTrainX, tempTrainY)
            tempLos.append(lR.loss(np.array(test_x), np.array(test_y)))
            
        meanLossArr.append(np.mean(tempLos))
        varLossArr.append(np.std(tempLos))
    
    pArr = np.array(pArr)
    meanLossArr = np.array(meanLossArr)
    varLossArr = np.array(varLossArr)

    
    plot = go.Figure((go.Scatter(x=pArr, y=meanLossArr,
                                mode="markers+lines", name="loss for samples", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                          go.Scatter(x=pArr, y=meanLossArr - 2 * varLossArr,
                                     fill=None, mode="lines", name="p samples", line=dict(color="lightgrey")),
                          go.Scatter(x=pArr, y=meanLossArr + 2 * varLossArr,
                                     fill='tonexty', mode="lines", name="loss mean and var", line=dict(color="lightgrey"))),
                                     layout=go.Layout(
                            title="loss for samples",
                            xaxis_title="p samples",
                            yaxis_title="loss mean and var"))
    plot.show()

    # plot.write_image("/Users/ozmatan4/Documents/Studies/huji/year3/semesterB/IML/exercises/ex2_images" + "/" "new.png")











