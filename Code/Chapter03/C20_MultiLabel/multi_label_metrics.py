import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss

def exact_match_ratio(y_true, y_pred):
    print(accuracy_score(y_true, y_pred))

def zero_one(y_true, y_pred):
    print(zero_one_loss(y_true, y_pred))


def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]

def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]

def Recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]

def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]

def Hamming_Loss(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = np.size(y_true[i] == y_pred[i])
        q = np.count_nonzero(y_true[i] == y_pred[i])
        count += p - q
    return count / (y_true.shape[0] * y_true.shape[1])


if __name__ == '__main__':
    y_true = np.array([[0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
    y_pred = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
    exact_match_ratio(y_true, y_pred)
    zero_one(y_true, y_pred)
    print(Accuracy(y_true, y_pred))
    print(Precision(y_true, y_pred))
    print(Recall(y_true, y_pred))
    print(F1Measure(y_true, y_pred))
    print(precision_score(y_true=y_true, y_pred=y_pred, average='samples'))
    print(recall_score(y_true=y_true, y_pred=y_pred, average='samples'))
    print(f1_score(y_true,y_pred,average='samples'))
    print(Hamming_Loss(y_true, y_pred))  # 0.5
    # print(hamming_loss(y_true, y_pred))  # 0.5