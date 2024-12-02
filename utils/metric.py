import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(y_test, y_pred, label_name):
    acc = accuracy_score(y_test.loc[:, [label_name]], y_pred)
    ppv = precision_score(y_test.loc[:, [label_name]], y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test.loc[:, [label_name]], y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test.loc[:, [label_name]], y_pred, average='weighted', zero_division=1)
    return acc, ppv, recall, f1

# # Pandora
# def evaluate(y_test, y_pred, label_name):
#     # Extract the true labels as a flat array
#     y_true = y_test.loc[:, [label_name]].values.flatten()
#     y_pred = y_pred.values.flatten()

#     # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)
#     TP = np.sum((y_true == 1) & (y_pred == 1))
#     FP = np.sum((y_true == 0) & (y_pred == 1))
#     FN = np.sum((y_true == 1) & (y_pred == 0))
#     TN = np.sum((y_true == 0) & (y_pred == 0))

#     # Calculate Accuracy
#     acc = (TP + TN) / (TP + TN + FP + FN)

#     # Calculate Precision (PPV)
#     ppv = TP / (TP + FP) if (TP + FP) > 0 else 1

#     # Calculate Recall (Sensitivity)
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 1

#     # Calculate F1 Score
#     f1 = 2 * (ppv * recall) / (ppv + recall) if (ppv + recall) > 0 else 0

#     return acc, ppv, recall, f1

def roc_auc_curve(tprs, aucs, mean_fpr, label_name):
    plt.figure()
    lw = 2
    plt.plot(mean_fpr, np.mean(tprs, axis=0), color='blue', lw=lw, label=' (area = %0.2f)'.format(label_name) % np.mean(aucs))
    plt.fill_between(mean_fpr, np.percentile(tprs, 5, axis=0), np.percentile(tprs, 95, axis=0), color='blue', alpha=0.2, label="95% CI")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for {}'.format(label_name))
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_{}.png'.format(label_name))
    return
