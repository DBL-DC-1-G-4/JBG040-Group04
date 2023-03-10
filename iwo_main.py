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
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, roc_auc_score, roc_curve,auc,RocCurveDisplay,precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
    test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"))
    
    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay=0.001) ##change from SGD-->ADAM ,weight_decay=0.0005
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
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
            #my addition 
            model.train()
            #end
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            
            losses = test_model(model, test_sampler, loss_function, device)
            ### My addition of checking the predictions

            n_classes=6
            # Create an empty numpy array to store the predicted probabilities for each test image
            pred_probs = np.zeros((len(test_dataset), n_classes))

            model.eval()

            # Create a dataloader for the test data
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

            # Create lists to store the predicted labels and ground truth labels
            pred_labels = []
            true_labels = []
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
                    #probs = torch.softmax(outputs, dim=1)

                    # Store the predicted labels and true labels for the current test image
                    pred_labels.extend(predicted.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

                    # Store the predicted probabilities for the current test image
                    #pred_probs[i] = probs.cpu().numpy()

            #sklearn function for a confusion matrix
            print("Recall score:", recall_score(true_labels, pred_labels, average="macro"))
            print("Precision score:", precision_score(true_labels, pred_labels, average="macro"))

            cm = confusion_matrix(true_labels, pred_labels)
            print(cm)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
            disp.plot()
            plt.title("Confusion matrix")
            plt.show()

            # Reshape the predicted probabilities array to have shape (n_samples, n_classes)
           # pred_probs = pred_probs.reshape((-1, 6))

            # # Calculate the AUC-ROC score OVR
            # auc_roc_score_ovr = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
            # print("AUC-ROC-OVR score:", auc_roc_score_ovr)
            # # Calculate the AUC-ROC score OVO
            # auc_roc_score_ovo = roc_auc_score(true_labels, pred_probs, multi_class='ovo')
            # print("AUC-ROC-OVO score:", auc_roc_score_ovo)

            # # Calculate the precision, recall, and f1-score for each class
            # precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)

            # # Print the precision, recall, and f1-score for each class
            # for i in range(len(precision)):
            #     print(f"Class {i}: precision={precision[i]}, recall={recall[i]}, f1-score={f1_score[i]}")


            # # Binarize the true labels
            # y_true = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5])


            # # Fit the OneVsRestClassifier on the predicted probabilities
            # classifier = OneVsRestClassifier(RandomForestClassifier())
            # classifier.fit(pred_probs, y_true)

            # # Compute the ROC curve and ROC area for each class
            # fpr = dict()
            # tpr = dict()
            # roc_auc = dict()
            # for i in range(n_classes):
            #     fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred_probs[:, i])
            #     roc_auc[i] = auc(fpr[i], tpr[i])

            # # Compute micro-average ROC curve and ROC area
            # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), pred_probs.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # # Plot the ROC curve for each class
            # plt.figure()
            # lw = 2
            # colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'black']
            # for i, color in zip(range(n_classes), colors):
            #     plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

            # # Plot the micro-average ROC curve
            # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
            # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Multi-class ROC Curve')
            # plt.legend(loc="lower right")
            # plt.show()




            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            # plotext.show() #cm if doesnt work

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

