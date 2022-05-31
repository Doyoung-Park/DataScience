import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def roc_curve_auc_score(X_test, y_test, y_pred_probabilities, classifier_name):
    
    y_pred_prob = y_pred_probabilities[:,1]
    fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob)

    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr,tpr,label = f'{classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} - ROC Curve')
    plt.show()

    return print(f'AUC Score (ROC) : {roc_auc_score(y_test, y_pred_prob)}\n')