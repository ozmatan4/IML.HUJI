from numpy import array
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    def callback_function(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(x, y))

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration

        losses = []
        percepObj = Perceptron(callback=callback_function) 
        percepObj._fit(X, y)

        # print(np.shape(X))

        # Plot figure of loss as function of fitting iteration
        x_axis = np.arange(len(losses))
        y_axis = np.array(losses)
        plt = go.Figure(data=go.Scatter(x=x_axis, y=y_axis))
        plt.update_layout(
            title=n+" Perceptron fitting",
            xaxis_title="Iter number",
            yaxis_title="Loss"
        ).show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        file = "/Users/ozmatan4/Documents/Studies/huji/year3/semesterB/IML/exercises/EX1/IML.HUJI/datasets/" + f
        X, y = load_dataset(file)

        # Fit models and predict over training set
            # Linear Discriminant Analysis (LDA) classifier Object
        ldaObj = LDA()
        ldaObj.fit(X, y)

            # Gaussian Naive-Bayes classifier Object
        gnbObj = GaussianNaiveBayes()
        gnbObj.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplotss
        from IMLearn.metrics import accuracy
        ldaPred = ldaObj.predict(X)
        gnbPred = gnbObj.predict(X)

        ldaTitle = "Accuracy LDA: " + str(accuracy(y, ldaPred))
        gnbTitle = "Accuracy GNB: " + str(accuracy(y, gnbPred))
        fig = make_subplots(rows=1, cols=2, subplot_titles=(gnbTitle, ldaTitle))

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=ldaPred, symbol=y)),
                        row=1, col=2)

        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=gnbPred, symbol=y)),
                        row=1, col=1)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=ldaObj.mu_[:,0], y=ldaObj.mu_[:, 1], mode="markers",
                       marker=dict(color="black", symbol='x')), row=1, col=2)

        fig.add_trace(
            go.Scatter(x=gnbObj.mu_[:,0], y=gnbObj.mu_[:, 1], mode="markers",
                       marker=dict(color="black", symbol='x')), row=1, col=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(np.size(ldaObj.classes_)):
            fig.add_trace(get_ellipse(ldaObj.mu_[i], ldaObj.cov_), row=1,col=2)

            varMat = gnbObj.vars_[i] * np.identity(X.shape[1])
            fig.add_trace(get_ellipse(gnbObj.mu_[i], varMat), row=1, col=1)

        fig.update_layout(height=600, width=1200, title_text=f, showlegend=False).show()



if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()

    X = np.array([[1,1], [1,2], [2,3], [2,4], [3,3], [3,4]])
    y = np.array([0,0,1,1,1,1])

    objMod = GaussianNaiveBayes()
    objMod.fit(X,y)
    print(objMod.vars_)
    # print(objMod.mu_)
