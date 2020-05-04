from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt

def ROC_curve(predictions, test_labels):
    # Plot an ROC curve
    new_labels = label_binarize(test_labels, classes=[digit for digit in range(10)])
    bin_pred = label_binarize(predictions, classes=[digit for digit in range(10)])
    colors = ['r', 'tab:orange', 'y', 'g', 'b', 'tab:purple', 'c', 'm', 'k', 'tab:olive',]
    for i in range(10):
        # Get false positive rate, true positive rate
        fpr, tpr, _ = roc_curve(new_labels[:, i], bin_pred[:, i])
        # Area under roc curve
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], label='Digit {0}: area = {1:0.5f}'.format(i, roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def report(test_labels, predictions):
    # Plot ROC Curve
    ROC_curve(predictions, test_labels)

    # MSE
    mse = mean_squared_error(test_labels, predictions)
    print("Mean Squared Error:", mse)

    # Confusion Matrix
    confusion_mtrx = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:")
    print(confusion_mtrx)
    print('Report : ')
    print(classification_report(test_labels, predictions, digits=5)) 
