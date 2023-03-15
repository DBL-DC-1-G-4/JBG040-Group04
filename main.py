# Custom imports
from batch_sampler import BatchSampler
from image_dataset import ImageDataset
from net import Net
from train_test import train_model, test_model

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List

# evaluation imports
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, roc_auc_score, roc_curve,auc,RocCurveDisplay,precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path("../data/X_trainshort.npy"), Path("../data/Y_trainshort.npy"))
    test_dataset = ImageDataset(Path("../data/X_testshort.npy"), Path("../data/Y_testshort.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)
    
    m = torch.jit.script(Net(n_classes=6))
    torch.jit.save(m, 'initial.pt')
    
    # This line is equivalent to the previous
    
    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    
    for e in range(n_epochs):
        if activeloop:

            # Training:
            model.train()
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            print(losses)
            print(type(losses))
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            losses = test_model(model, test_sampler, loss_function, device)

            # Create an empty numpy array to store the predicted probabilities for each test image
            n_classes=6
            pred_probs = np.zeros((len(test_dataset), n_classes))
            model.eval()
            
            # Create a dataloader for the test data
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

            # Create lists to store the predicted labels and ground truth labels
            pred_labels = []
            true_labels = []
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")
            cm = []

            # Create an empty numpy array to store the predicted probabilities for each test image
            pred_probs = np.zeros((len(test_dataset), 6))

            # Iterate over the test data and make predictions
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    # Move the images and labels to the device (GPU/CPU) used for training
                    images = images.to(device)
                    labels = labels.to(device)

                    # Make predictions on the test images
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    probs = torch.softmax(outputs, dim=1)

                    # Store the predicted labels and true labels for the current test image
                    pred_labels.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

                    # Store the predicted probabilities for the current test image
                    pred_probs[i] = probs.cpu().numpy()
            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()
            # Reshape the predicted probabilities array to have shape (n_samples, n_classes)
            pred_probs = pred_probs.reshape((-1, 6))

            # Calculate the AUC-ROC score OVR
            auc_roc_score_ovr = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
            print("AUC-ROC-OVR score:", auc_roc_score_ovr)
            # Calculate the AUC-ROC score OVO
            auc_roc_score_ovo = roc_auc_score(true_labels, pred_probs, multi_class='ovo')
            print("AUC-ROC-OVO score:", auc_roc_score_ovo)

            # Calculate the precision, recall, and f1-score for each class
            precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)

            # calculate micro-average, macro-average, and weighted-average values
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


            # Print the precision, recall, and f1-score for each class
            for i in range(len(precision)):
                print(f"Class {i}: precision={precision[i]}, recall={recall[i]}, f1-score={f1_score[i]}")


            # Binarize the true labels
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


    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    
    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")
    
    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # create the table
    fig, ax = plt.subplots()
    table = ax.table(cellText=table_data, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    ax.axis('off')
    fig.savefig(Path("artifacts") / f"table_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # save ROC curve
    fig = plt.figure(figsize=(8,6), dpi=80)
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'black']
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    fig.savefig(Path("artifacts") / f"ROC_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # save confusion matrix
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


    # save superimposed plot of losses for epochs
    fig = plt.figure(figsize=(8,6), dpi=80)
    plt.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    plt.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("superimposed loss")
    fig.savefig(Path("artifacts") / f"loss_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)
