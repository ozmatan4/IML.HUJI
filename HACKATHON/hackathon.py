import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
import re
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor



def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True) -> float:
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
    number_of_misclass = np.sum(y_true != y_pred)
    if normalize:
        return (1 / y_true.shape[0]) * number_of_misclass
    return number_of_misclass

def load_data_part1(filename_feat: str, filename_label: str):
    """
    Load dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to dataset

    Returns
    -------
    Design matrix and response vector - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename_feat, low_memory=False)
    df_label = pd.read_csv(filename_label, low_memory=False)
    df_new = df.loc[:, ['אבחנה-Age', 'אבחנה-M -metastases mark (TNM)', 'אבחנה-Positive nodes',
                        'אבחנה-Stage', 'אבחנה-Surgery name1', 'אבחנה-Surgery sum',
                        'surgery before or after-Actual activity']]
    df_new['y_response'] = df_label

    df_new = df_new[df_new['אבחנה-Age'] > 0]

    df_new = pd.get_dummies(df_new, prefix='TNM: ', columns=['אבחנה-M -metastases mark (TNM)'])

    df_new['אבחנה-Surgery sum'] = df_new['אבחנה-Surgery sum'].fillna(0)

    positiveNodesMean = int(df_new['אבחנה-Positive nodes'].mean())
    df_new['אבחנה-Positive nodes'] = df_new['אבחנה-Positive nodes'].fillna(
        positiveNodesMean)  # Todo:change to 0 ????
    df_new = df_new[df_new['אבחנה-Positive nodes'] >= 0]

    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].replace(to_replace='Not yet Established',
                                                          value='2', regex=True)
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].replace(
        to_replace=["Stage", "a", 'b', 'c', 'LA', 'is'], value=['', '.25', '.5', '.75', '2', '.5'],
        regex=True)
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].astype('float')
    stageMean = (float)(df_new['אבחנה-Stage'].mean())
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].fillna(stageMean)
    df_new = pd.get_dummies(df_new, prefix='Surgery Name: ', columns=['אבחנה-Surgery name1'])
    df_new = pd.get_dummies(df_new, prefix='Surgery Activity: ',
                            columns=['surgery before or after-Actual activity'])
    # print(df_new.shape)

    return df_new


def load_data_test_part1(filename_feat: str):
    """
    Load dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to dataset

    Returns
    -------
    Design matrix and response vector - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename_feat, low_memory=False)
    df_new = df.loc[:, ['אבחנה-Age', 'אבחנה-M -metastases mark (TNM)', 'אבחנה-Positive nodes',
                        'אבחנה-Stage', 'אבחנה-Surgery name1', 'אבחנה-Surgery sum',
                        'surgery before or after-Actual activity']]

    df_new = df_new[df_new['אבחנה-Age'] > 0]

    df_new = pd.get_dummies(df_new, prefix='TNM: ', columns=['אבחנה-M -metastases mark (TNM)'])

    df_new['אבחנה-Surgery sum'] = df_new['אבחנה-Surgery sum'].fillna(0)

    positiveNodesMean = int(df_new['אבחנה-Positive nodes'].mean())
    df_new['אבחנה-Positive nodes'] = df_new['אבחנה-Positive nodes'].fillna(
        positiveNodesMean)
    df_new = df_new[df_new['אבחנה-Positive nodes'] >= 0]

    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].replace(to_replace='Not yet Established',
                                                          value='2', regex=True)
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].replace(
        to_replace=["Stage", "a", 'b', 'c', 'LA', 'is'], value=['', '.25', '.5', '.75', '2', '.5'],
        regex=True)
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].astype('float')
    stageMean = (float)(df_new['אבחנה-Stage'].mean())
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].fillna(stageMean)
    df_new = pd.get_dummies(df_new, prefix='Surgery Name: ', columns=['אבחנה-Surgery name1'])
    df_new = pd.get_dummies(df_new, prefix='Surgery Activity: ',
                            columns=['surgery before or after-Actual activity'])
    # print(df_new.shape)
    return df_new


def part1():
    np.random.seed(0)
    # Load and preprocessing of dataset
    filename_feat = "HACKATHON/Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/train.feats.csv"
    filename_label0 = "HACKATHON/Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/train.labels.0.csv"
    filename_label1 = "HACKATHON/Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/train.labels.1.csv"
    filename_test_feat = "HACKATHON/Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/test.feats.csv"

    # Part 1: Predicting Metastases
    df1 = load_data_part1(filename_feat, filename_label0)
    df1_test = load_data_test_part1(filename_test_feat)

    # Split samples into training- and testing sets.
    train1, test1 = train_test_split(df1, test_size=0.2, random_state=42, shuffle=True)
    # test1_y = test1["y_response"].squeeze()
    # test1_X = np.array(test1.drop(columns=["y_response"])).squeeze()
    #
    # train1_y = train1["y_response"]
    # train1_X = train1.drop(columns=["y_response"])

    allDiseases = ['ADR - Adrenals', 'LYM - Lymph nodes', 'BON - Bones',
                   'HEP - Hepatic', 'PUL - Pulmonary', 'PLE - Pleura',
                   'SKI - Skin', 'BRA - Brain', 'PER - Peritoneum',
                   'MAR - Bone Marrow', 'OTH - Other']
    # allTrainsy = []
    # allTesty = []
    # for disease in allDiseases:
    #     r = re.compile(r'.*({}).*'.format(disease))
    #     allTrainsy.append(train1_y.apply(lambda x: int(bool(r.match(x)))))
    #     allTesty.append(test1_y.apply(lambda x: int(bool(r.match(x)))))


    # DecisionTreeLoss = []
    # KNNLoss = []
    # for j in range(11):
    #     test1_y = np.array(allTesty[j])
    #     models = [DecisionTreeClassifier(max_depth=4), KNeighborsClassifier(n_neighbors=4)]
    #     model_names = ["Desicion Tree (Depth 5)", "KNN"]
    #
    #     y = np.array(allTrainsy[j].squeeze())
    #     X = np.array(train1_X.squeeze())
    #     for i, model in enumerate(models):
    #         model.fit(X, y)
    #         y_pred = model.predict(test1_X)
    #         loss = misclassification_error(test1_y, y_pred)
    #         if i == 0:
    #             DecisionTreeLoss.append(loss)
    #         else:
    #             KNNLoss.append(loss)
    #
    # DecisionTreeLoss = np.array(DecisionTreeLoss)
    # KNNLoss = np.array(KNNLoss)

    # print(DecisionTreeLoss)
    # print(KNNLoss)
    # print(DecisionTreeLoss.mean())
    # print(KNNLoss.mean())

    # evaluation
    data1_y = df1["y_response"].squeeze()
    data1_X = np.array(df1.drop(columns=["y_response"])).squeeze()
    DecisionTreePred = [[] for i in range(df1_test.shape[0])]
    DecisionTree = DecisionTreeClassifier(max_depth=4)

    all_trains_y = []
    for disease in allDiseases:
        r = re.compile(r'.*({}).*'.format(disease))
        all_trains_y.append(data1_y.apply(lambda x: int(bool(r.match(x)))))

    df1_zeros_test = np.zeros((df1_test.shape[0], (data1_X.shape[1] - df1_test.shape[1])))
    df1_test = np.append(df1_test, df1_zeros_test, axis=1)
    for j in range(len(all_trains_y)):
        y = np.array(all_trains_y[j].squeeze())
        X = np.array(data1_X.squeeze())
        DecisionTree.fit(X, y)
        y_pred = DecisionTree.predict(df1_test)
        args = np.argwhere(y_pred == 1).flatten()
        for arg in args:
            DecisionTreePred[arg].append(allDiseases[j])

    for i in range(len(DecisionTreePred)):
        DecisionTreePred[i] = str(DecisionTreePred[i])

    csv_pred = pd.DataFrame(DecisionTreePred, columns=['אבחנה-Location of distal metastases'])
    csv_pred.to_csv('HACKATHON/predicitions.csv', index=False)

    # go.Figure([
    #     go.Scatter(name='Noiseless set', x=x, y=DecisionTreeLoss, mode='markers',
    #                marker_color='rgb(152,171,150)')]).show()

    # fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
    #                     horizontal_spacing=0.01, vertical_spacing=.03)
    # for i, m in enumerate(models):
    #     fig.add_traces([decision_surface(m.fit(X, y).predict, lims[0], lims[1], showscale=False),
    #                     go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
    #                                marker=dict(color=y, symbol=symbols[y],
    #                                            colorscale=[custom[0], custom[-1]],
    #                                            line=dict(color="black", width=1)))],
    #                    rows=(i // 3) + 1, cols=(i % 3) + 1)

    # fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - {title} Dataset}}$",
    #                   margin=dict(t=100)) \
    #     .update_xaxes(visible=False).update_yaxes(visible=False)

    # fig = go.Figure(
    #     layout=go.Layout(title=rf"$\textbf{{(3) ROC Curves Of Models -  Dataset}}$",
    #                      margin=dict(t=100)))
    # for i, model in enumerate(models):
    #     fpr, tpr, th = metrics.roc_curve(test1_y, model.predict_proba(test1_X)[:, 1])
    #     fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=model_names[i]))
    #
    # fig.show()



def load_data(filename_feat: str, filename_label: str, filename_test_feat : str):
    """
    Load dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to dataset

    Returns
    -------
    Design matrix and response vector - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename_feat, low_memory=False)
    df_label = pd.read_csv(filename_label, low_memory=False)
    df_test = pd.read_csv(filename_test_feat, low_memory=False)
    ones = np.zeros(df_test.shape[0])
    ones = pd.DataFrame(ones, columns=['אבחנה-Tumor size'])
    df_label = df_label.append(ones, ignore_index = True)
    df_test_row_index = df.shape[0]

    df = df.append(df_test, ignore_index = True)

    df_new = df.loc[:, ['אבחנה-Age', 'אבחנה-M -metastases mark (TNM)', 'אבחנה-Positive nodes',
                        'אבחנה-Stage', 'אבחנה-Surgery name1', 'אבחנה-Surgery sum',
                        'surgery before or after-Actual activity']]

    

    df_new['y_response'] = df_label

    df_new = df_new[df_new['אבחנה-Age'] > 0]

    df_new = pd.get_dummies(df_new, prefix='TNM: ', columns=['אבחנה-M -metastases mark (TNM)'])

    df_new['אבחנה-Surgery sum'] = df_new['אבחנה-Surgery sum'].fillna(0)

    positiveNodesMean = int(df_new['אבחנה-Positive nodes'].mean())
    df_new['אבחנה-Positive nodes'] = df_new['אבחנה-Positive nodes'].fillna(
        positiveNodesMean)  # Todo:change to 0 ????
    df_new = df_new[df_new['אבחנה-Positive nodes'] >= 0]

    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].replace(to_replace='Not yet Established',
                                                          value='2', regex=True)
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].replace(
        to_replace=["Stage", "a", 'b', 'c', 'LA', 'is'], value=['', '.25', '.5', '.75', '2', '.5'],
        regex=True)
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].astype('float')
    stageMean = (float)(df_new['אבחנה-Stage'].mean())
    df_new['אבחנה-Stage'] = df_new['אבחנה-Stage'].fillna(stageMean)
    df_new = pd.get_dummies(df_new, prefix='Surgery Name: ', columns=['אבחנה-Surgery name1'])
    df_new = pd.get_dummies(df_new, prefix='Surgery Activity: ', columns=['surgery before or after-Actual activity'])

    df_new_lable = df_new['y_response']
    df_new_lable = df_new_lable.iloc[:df_test_row_index]

    df_new = df_new.drop(columns=['y_response'])

    df_train = df_new.iloc[:df_test_row_index,:]

    df_new_test = df_new.iloc[df_test_row_index:,:]
    

    return df_train, df_new_test, df_new_lable


def cross_validate(estimator, X: np.ndarray, y: np.ndarray,scoring, cv: int = 5):

    trainScore, validationScore = 0, 0
    foldSize= int(X.shape[0]/cv)
    for k in range(cv):
        xTrainFold, yTrainFold = np.concatenate((X[: k*foldSize], X[k*foldSize + foldSize:]), axis=0),\
             np.concatenate((y[: k*foldSize], y[k*foldSize + foldSize:]), axis=0)
        xValidateFold, yValidateFold = X[k*foldSize : k*foldSize + foldSize], y[k*foldSize : k*foldSize + foldSize]
        estimator.fit(xTrainFold, yTrainFold)
        trainScore += scoring(yTrainFold, estimator.predict(xTrainFold))
        validationScore += scoring(yValidateFold, estimator.predict(xValidateFold))
    return trainScore/cv, validationScore/cv


def part2():
    # Load and preprocessing of dataset
    filename_feat = "HACKATHON/Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/train.feats.csv"
    filename_label1 = "HACKATHON/Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/train.labels.1.csv"
    filename_test_feat = "HACKATHON/Data and Supplementary Material-20220602/Mission 2 - Breast Cancer/test.feats.csv"


    # Part 2: Predicting Tumor Size
    df2, df_unlabeled_Test, df_lable = load_data(filename_feat, filename_label1, filename_test_feat)

    df2["y_response"] = df_lable


    # df2.to_csv("HACKATHON/df2.csv", index=False)

    # Split samples into training- and testing sets.
    train2, test2 = train_test_split(df2, test_size=0.2,
                                   random_state=42,shuffle=True)

    # # Split samples into training- and validation sets.
    # train2, valid2 = train_test_split(train2 , test_size=0.25,
    #                                random_state=42,shuffle=True)

    
    #arrange data
    train2_y = (train2["y_response"]).squeeze()
    train2_X = np.array((train2.drop(columns=["y_response"])).squeeze())

    test2_y = np.array((test2["y_response"]).squeeze())
    test2_X = np.array((test2.drop(columns=["y_response"])).squeeze())

    regresor = RandomForestRegressor(max_depth=25, random_state=0)
    # regresor.fit(train2_X, train2_y)

    trainError, validationError = cross_validate(regresor, train2_X, train2_y, mean_squared_error, 10)

    print("the train error is: " + str(trainError))
    print("the validation error is: " + str(validationError))

    Y_pred = regresor.predict(test2_X)
    print("the test error is: " + str(mean_squared_error(Y_pred, test2_y)))


    x_unlabeled_Test = np.array(df_unlabeled_Test.squeeze())
    unlabeled_train_pred = regresor.predict(x_unlabeled_Test)

    modify_unlabeled_train_pred = pd.DataFrame(np.array(unlabeled_train_pred), columns=['אבחנה-Tumor size'])

    modify_unlabeled_train_pred.to_csv("HACKATHON/df_unlabeled_Test.csv", index=False)




if __name__ == '__main__':
    np.random.seed(0)
    part1()
    part2()

