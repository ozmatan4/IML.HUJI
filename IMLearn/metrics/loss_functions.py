from venv import main
import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    mse = np.mean(np.square(y_true - y_pred))
    return mse


# y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
# y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
# print(mean_square_error(y_true, y_pred))

def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """

    misVec = y_pred[y_pred==y_true]

    # someVec = y_pred + y_true
    # misVec = someVec[someVec==0]

    return (np.size(misVec)/y_pred.size if normalize else np.size(misVec))

# y_true = np.array([1,2,3,3,5])
# y_pred = np.array([1,2,2,3,5])
# print(misclassification_error(y_true, y_pred))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    truePredArr = y_true[y_true==y_pred]
    return (truePredArr.size/y_true.size)

# y_true = np.array([1,2,3,4,5])
# y_pred = np.array([1,2,2,3,5])
# print(accuracy(y_true, y_pred))

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()



