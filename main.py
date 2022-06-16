"""
                    Abdullah Baig - 231485698
"""
from math import sqrt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


def euclidean_distance(test_row, row):
    distance = 0.0
    for i in range(len(test_row)):
        distance += (test_row[i] - row[i]) ** 2
    return sqrt(distance)


# def get_neighbours()

def most_common(lst):
    temp = list()
    for label, distance in lst:
        temp.append(label[0])
    return max(set(temp), key=temp.count)


def knn_classifier(x_test, x_train, y_test, y_train):
    k = 3
    distances = list()

    predicted_results = list()

    x_test = list(x_test.values)
    x_train = list(x_train.values)
    y_test = list(y_test.values)
    y_train = list(y_train.values)

    for i in x_test:

        for j in range(len(x_train)):
            distances.append((y_train[j], euclidean_distance(i, x_train[j])))

        predicted_outcome = most_common(sorted(distances, key=lambda x: x[1])[:k])
        predicted_results.append(predicted_outcome)
        distances = list()

    return predicted_results


if __name__ == "__main__":
    df = pd.read_csv('diabetes.csv')

    x = df.drop(columns=["Outcome"], axis=1)
    y = df.filter(["Outcome"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=41)

    predicted = knn_classifier(x_test, x_train, y_test, y_train)
    y_test = list(map(lambda x: list(x)[0], list(y_test.values)))

    print("Confusion Matrix: \n", confusion_matrix(y_test, predicted))
    print("\nAccuracy: ", accuracy_score(y_test, predicted) * 100)
