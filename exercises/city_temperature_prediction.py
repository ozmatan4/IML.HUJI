import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting, polynomial_fitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna()
    # period = pd.Period(df["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear

    df.drop(columns=["Date", "Day"])

    df = df[df["Temp"] > -10]
    df = df[df["Temp"]< 45]

    # month = df["Month"]
    # country = df["Country"]
    # df = pd.get_dummies(df, prefix="Country_is", columns=["Country"])
    # df = pd.get_dummies(df, prefix="City_is", columns=["City"])
    # df = pd.get_dummies(df, prefix="Month_is", columns=["Month"])
    # df["Month"] = month
    # df["Country"] = country

    # y_temperature = df["Temp"]
    # df= df.drop(columns=["Temp"])

    return df



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("/Users/ozmatan4/Documents/Studies/huji/year3/semesterB/IML/exercises/EX1/IML.HUJI/datasets/City_Temperature.csv")
    


    # Question 2 - Exploring data for specific country
    # subsetDF = df[df["Country_is_Israel"]==1]
    subsetDF = df[df["Country"]=="Israel"]


    plot = px.scatter(x=subsetDF["DayOfYear"], y=subsetDF["Temp"], color=(subsetDF["Year"].astype(str)),
                     title="Israel temperature per day and years by color",
                     labels={"x":"Day of year", "y":"Temperature"})
    plot.show()


    subsetDF2 = subsetDF.groupby("Month")
    subsetDF2 = subsetDF2.agg("std")

    plot = px.bar(subsetDF2, x=subsetDF2.index, y="Temp",
                 title="Standard deviation - temperature by month",
                 labels={"Temp":"temperature Standard deviation"})
    plot.show()


    # Question 3 - Exploring differences between countries
    countryMonthGB = df.groupby(["Country", "Month"])
    countryMonthGB = countryMonthGB.agg(aver=("Temp", "mean"), err=("Temp", "std"))
    countryMonthGB = countryMonthGB.reset_index()

    plot = px.line(countryMonthGB, x="Month", y="aver", error_y="err", color="Country",
                  title="average and standard deviation of the temperature",
                  labels={"mean":"mean Temperature"})
    plot.show()



    # Question 4 - Fitting model for different values of `k`
    y = subsetDF["Temp"]
    x = subsetDF.drop(columns=["Temp"])
    [train_x, train_y, test_x, test_y] = split_train_test(x, y, 0.75)


    LossArr = []
    
    kArr= np.arange(1,11,1)
    for k in kArr:
        polyFit = PolynomialFitting(k)
        polyFit.fit(train_x["DayOfYear"], train_y)
        tempLos = round(polyFit.loss(test_x["DayOfYear"], test_y), 2)
        LossArr.append(tempLos)
    
    kArr = np.array(kArr)
    LossArr = np.array(LossArr)
    print(LossArr)
    fig = px.bar(x=kArr, y=LossArr, title="error recorded for each value of k.",
                 labels={"x":"K of the polynomials", "y":"Loss"})
    fig.show()




    # Question 5 - Evaluating fitted model on different countries
    





















