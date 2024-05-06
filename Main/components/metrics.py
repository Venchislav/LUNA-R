import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def confusion_matrix(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            tp += 1
        elif y_true[i] == y_pred[i] == 0:
            tn += 1
        elif (y_true[i] == 0) and (y_pred[i] == 1):
            fp += 1
        else:
            fn += 1

    return np.array([[tp, fp], [fn, tn]])


def plot_confusion_matrix(matrix):
    plt.title('Confusion Matrix')
    matrix = pd.DataFrame(matrix, index=[i for i in "10"], columns=[i for i in "10"])
    sns.heatmap(matrix, annot=True)
    plt.show()


matrix = confusion_matrix([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
plot_confusion_matrix(matrix)
