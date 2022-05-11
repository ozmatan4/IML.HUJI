import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump

from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    symbols = np.array(["circle", "x"])
    boolY = np.where(test_y < 0, 1, 0)

    # Question 1: Train- and test errors of AdaBoost in noiseless case

    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)

    trainLossArr = [adaBoost.partial_loss(train_X, train_y, t+1) for t in range(n_learners)]
    testLossArr = [adaBoost.partial_loss(test_X, test_y, t+1) for t in range(n_learners)]

    trainLossArr = np.array(trainLossArr)
    testLossArr = np.array(testLossArr)
    learnersNumber = np.arange(1, n_learners+1)


    fig = go.Figure([go.Scatter(x=learnersNumber, y=trainLossArr, mode='lines', name='Train Loss'), 
                go.Scatter(x=learnersNumber, y=testLossArr, mode='lines', name='Test Loss')])
    fig.update_layout(title=rf"$\text{{Errors for learners number }}$")
    fig.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{learner} Learners" for learner in T],
                          horizontal_spacing=0.01, vertical_spacing=0.03)
    

    for i, learner in enumerate(T):
        def pred(X):
            return adaBoost.partial_predict(X, learner)
        fig.add_traces([decision_surface(pred, lims[0], lims[1], showscale=True),
                          go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=test_y, symbol=symbols[boolY],
                                                 colorscale=[custom[0], custom[-1]],
                                                 line=dict(color="black", width=1))),],
                         rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=f"  Quastion 2 - the noise is: {noise} -"
                              f"  Decision Boundary to number of learners 5, 50, 100, 250",
                        margin_t=100).update_xaxes(visible=False).update_yaxes(visible=False).show()
 


    # Question 3: Decision surface of best performing ensemble

    minIndex = np.argmin(testLossArr)
    bestLearnersSize = minIndex + 1
    
    fig = go.Figure([decision_surface(lambda X: adaBoost.partial_predict(X, bestLearnersSize),
                                        lims[0], lims[1], showscale=False),
                       go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=test_y, symbol=symbols[boolY],
                                              colorscale=[custom[0], custom[-1]],
                                              line=dict(color="black", width=2)))])
    fig.update_layout(title=f"Quastion 3 - the noise is: {noise} - Best Ensemble decision Boundary, Size: "
                              f" {bestLearnersSize} Accuracy : {1 - testLossArr[minIndex]}")
    fig.update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 4: Decision surface with weighted samples

    D = adaBoost.D_
    D = (D / np.max(D)) * 5

    go.Figure([decision_surface(adaBoost.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y, symbol=symbols[boolY], colorscale=custom, line=dict(color="black", width=1), size=D))],
              layout=go.Layout(title=f"Quastion 4 - Decision surface with weighted")).show()





if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)



