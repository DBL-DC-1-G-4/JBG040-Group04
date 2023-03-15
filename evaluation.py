from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, roc_auc_score, roc_curve,auc,RocCurveDisplay,precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
from pathlib import Path
from datetime import datetime

def evaluation (pred_labels, true_labels, pred_probs):
    now = datetime.now()
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))
    
    print("Recall score:", recall_score(true_labels, pred_labels, average="macro"))
    print("Precision score:", precision_score(true_labels, pred_labels, average="macro"))
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='micro')
    macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
    weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

    # create the table data for precision, recall and f1
    table_data = [['Class', 'Precision', 'Recall', 'F1-score']]
    for i in range(len(precision)):
        table_data.append(['Class {}'.format(i+1), '{:.2f}'.format(precision[i]), '{:.2f}'.format(recall[i]), '{:.2f}'.format(f1_score[i])])

    table_data.append(['Micro-average', '{:.2f}'.format(micro_precision), '{:.2f}'.format(micro_recall), '{:.2f}'.format(micro_f1_score)])
    table_data.append(['Macro-average', '{:.2f}'.format(macro_precision), '{:.2f}'.format(macro_recall), '{:.2f}'.format(macro_f1_score)])
    table_data.append(['Weighted-average', '{:.2f}'.format(weighted_precision), '{:.2f}'.format(weighted_recall), '{:.2f}'.format(weighted_f1_score)])

    # create the table
    fig, ax = plt.subplots()
    table = ax.table(cellText=table_data, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    ax.axis('off')
    fig.savefig(Path("artifacts") / f"table_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


    #Reshape the predicted probabilities array to have shape (n_samples, n_classes)
    pred_probs = pred_probs.reshape((-1, 6))

    # Calculate the AUC-ROC score OVR
    auc_roc_score_ovr = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
    print("AUC-ROC-OVR score:", auc_roc_score_ovr)
    # Calculate the AUC-ROC score OVO
    auc_roc_score_ovo = roc_auc_score(true_labels, pred_probs, multi_class='ovo')
    print("AUC-ROC-OVO score:", auc_roc_score_ovo)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    fig = plt.figure(figsize=(8,6), dpi=80)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
    disp.plot()
    plt.title("Confusion matrix")
    disp.figure_.savefig(Path("artifacts") / f"confmatr_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")
    

    # new encoding scheme 0 - no finding, 1 - low risk, 2 - high risk
    # remember now does not show missclassifying low risk dieases between them, same for high risk
    class_map = {3: 0,
                 2: 2, 
                 5: 2, 
                 0: 1, 
                 4: 1, 
                 1: 1}
    true_labels_mapped = [class_map[label] for label in true_labels]
    pred_labels_mapped = [class_map[label] for label in pred_labels]
    fig = plt.figure(figsize=(8,6), dpi=80)
    cm = confusion_matrix(true_labels_mapped, pred_labels_mapped)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.title("Confusion matrix of severity")
    disp.figure_.savefig(Path("artifacts") / f"confmatrseverity_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")
    

def roc(true_labels,pred_probs):
               # Binarize the true labels
            n_classes=6
            y_true = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5])


            # Fit the OneVsRestClassifier on the predicted probabilities
            classifier = OneVsRestClassifier(RandomForestClassifier())
            classifier.fit(pred_probs, y_true)

            # Compute the ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), pred_probs.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Plot the ROC curve for each class
            fig = plt.figure(figsize=(8,6), dpi=80)
            lw = 2
            colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'black']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

            # Plot the micro-average ROC curve
            plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curve')
            plt.legend(loc="lower right")
            fig.savefig(Path("artifacts") / f"ROC_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")