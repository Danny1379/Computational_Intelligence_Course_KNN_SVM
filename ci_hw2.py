import numpy as np
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from numpy import nan


def load_data():
    csv_raw = io.StringIO("adult.dat")
    df = pd.read_csv("adult.dat", header=None)
    print(df)
    return df


"""

Preprocessing:

methods used : 

  removal of unknown data

  turning categorical data to be able to use in our models

  binning integer data for easier use and taking care of the bias for larger numbers

  separating number and categorical data to be able to implement above mentioned methods  



"""


def preprocess(df):
    df_nan = df[0:].replace('?', nan)
    df2 = df_nan
    df2.dropna(inplace=True)
    # separating Results from Data for training our model
    X_raw = df2[df2.columns[:14]]
    Y_raw = df2[df2.columns[14]]
    return X_raw, Y_raw


def encode_y(y_raw):
    # SKLearn Label Encoder for string values also used on Y so it becomes binary
    lencoder = LabelEncoder()
    Y = lencoder.fit_transform(y_raw)
    Y = np.asarray(Y)
    return Y


def encode_x(X_raw):
    # divide data between string type and int types
    X_String = X_raw.select_dtypes(include=[object])
    X_int = X_raw.select_dtypes(exclude=[object])
    # one hot encoding categorical data to be usable
    enc = OneHotEncoder(handle_unknown='ignore')
    encoded = enc.fit(X_String)
    X_String_Onehot = encoded.transform(X_String).toarray()
    # binning integer data into bins of 4 integers
    est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
    est.fit(X_int)
    X_binned = est.transform(X_int)
    # joining both Feature vectors of int and String that have been turned into usable data
    X_new = X_binned
    X_new = np.append(X_binned, X_String_Onehot, axis=1)

    # test to see if correct shape
    print("test joined shape : ", X_new.shape)

    return X_new


def svm_predict(X_new, Y):
    # splitting data for training and test use
    x_train, x_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10, random_state=0)

    # support vector machine using SKLearn SVM
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    # checking model Accuracy
    predict_svm = clf.predict(x_test)
    accuracy = get_accuracy(predict_svm, y_test)
    print("svm accuracy is :", accuracy)


# find the distance between to points in M dimension plane
def euclidean_distance(X1, X2):
    return np.sqrt(np.sum((X1 - X2) ** 2, 1))


# return K smallest indexes of X_train
def predict_distance(X_test, X_train, k):
    Distances = np.zeros(len(X_train))
    Distances = euclidean_distance(X_test, X_train)
    indexes = np.argpartition(Distances, k)[:k]
    return indexes


# predicted class is returned based on given distances
# this is for binary if it was not binary we would need extra function parameter and diffrent implementation
def select_class(min_indexes, y_train):
    ones = 0
    zeroes = 0
    for i in range(len(min_indexes)):
        if y_train[min_indexes[i]]:
            ones += 1
        else:
            zeroes += 1
    return 1 if ones > zeroes else 0


def predict(X_test, X_train, y_train, k):
    answers = np.zeros(len(X_test))
    for i in range(len(X_test)):
        temp = predict_distance(X_test[i], X_train, k)
        answers[i] = select_class(temp, y_train)
    return answers


def get_accuracy(prediction, y):
    return np.mean(prediction == y)


def knn(x_new, Y):
    x_train, x_test, y_train, y_test = train_test_split(x_new, Y, test_size=0.10, random_state=0)
    Accuracys = np.zeros(11)
    for k in range(3, 14):
        if k % 2 != 0:
            predictions = predict(x_test, x_train, y_train, k)
            Accuracys[k - 3] = get_accuracy(predictions, y_test)
            print("KNN for K", k, "finished with Accuracy of", Accuracys[k - 3])
    min_index = np.argmax(Accuracys)
    print("the best KNN accuracy is : ", Accuracys[min_index], "for K :", min_index + 3)


def main():
    df = load_data()
    x_raw, y_raw = preprocess(df)
    y = encode_y(y_raw)
    x = encode_x(x_raw)
    svm_predict(x, y)
    knn(x, y)


if __name__ == '__main__':
    main()
