# Custom imports
from batch_sampler import BatchSampler
from image_dataset import ImageDataset
from vgg import VGG
from baseline import Net
from resnet import *
from train_test import train_model, test_model
from augmentation import augment
from balancer import balance
from augmentedBalance import augmentedBalance

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
from evaluation import evaluation
from validation_split import validation_split
from tqdm import tqdm
import random

# Dictionary with label names
label_dict = {0:"Atelectasis", 1: "Effusion", 2: "Infiltration" ,3: "No finding",4: "Nodule",5: "Pneumothorax"}

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    augmentation = args.augmentation
    augmentation_bal=args.augmentation_bal
    balancing = args.balanced
    validation_ratio = args.validation_ratio

    directory = "data/"
    
    if(augmentation>0):
        print("Running on augmented data!")
        if not Path("data/augmented/").exists():
            os.mkdir(Path("data/augmented/"))
        augment(pVersion=4) #Change augmentation version here
        directory = "data/augmented/"
    if(augmentation_bal>0):
        print("Running on balanced and then augmented data!")
        if not Path("data/balanced_and_augmented/").exists():
            os.mkdir(Path("data/balanced_and_augmented/"))
        augmentedBalance(pVersion=4) #Change augmentation version here
        directory = "data/balanced_and_augmented/"
    
    if(balancing>0):
        print("Running on balanced data!")
        if not Path("data/balanced/").exists():
            os.mkdir(Path("data/balanced/"))
        balance()
        directory = "data/balanced/"
    if augmentation > 0 and balancing > 0:
        print("Invalid arguments, both Augmentation and Balancing > 0")
        return 0

    # Construct the validation datasets
    validation_split(validation_ratio, directory)
    print(directory)

    # Load all of the datasets
    train_dataset = ImageDataset(Path(directory+"X_train_split.npy"), Path(directory+"Y_train_split.npy"))
    val_dataset = ImageDataset(Path(directory+"X_validation_split.npy"), Path(directory+"Y_validation_split.npy"))

    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

    print(len(train_dataset))
    
    
    # Load the Neural Net. NOTE: set number of distinct labels here
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=6)
    #model = VGG(n_classes=6)
    #model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001) ##change from SGD-->ADAM ,weight_decay=0.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True,min_lr=0.00001)
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
        batch_size=batch_size, dataset=train_dataset, balanced=balancing)
    
    val_sampler = BatchSampler(
            batch_size=batch_size, dataset=val_dataset, balanced=balancing)
    
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=1)        
    mean_losses_train: List[torch.Tensor] = []
    mean_losses_val: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    
    n_classes=6

    for e in range(n_epochs):
        if activeloop:
            # Training:
            #my addition 
            
            #end
            
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1}/{n_epochs} training done, loss on train set: {mean_loss}\n")
            
            # Validation:
            losses = test_model(model, val_sampler, loss_function, device)
 
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_val.append(mean_loss)
            print(f"\nEpoch {e + 1}/{n_epochs} validation done, loss on validation set: {mean_loss}\n")
            scheduler.step(mean_loss)
            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.scatter(mean_losses_val, label="validation")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            # plotext.show() #cm if doesnt work


    # Testing:

    losses = test_model(model, test_sampler, loss_function, device)
    # Calculating and printing statistics:
    mean_loss_test = sum(losses) / len(losses)
    print(f"Final Test Loss: {mean_loss_test}\n")

    
    # Create an empty numpy array to store the predicted probabilities for each test image
    pred_probs = np.zeros((len(test_dataset), n_classes))


 

    # Create lists to store the predicted labels and ground truth labels
    pred_labels = []
    true_labels = []

    # Create an empty numpy array to store the predicted probabilities for each test image
    pred_probs = []

    # Iterate over the test data and make predictions
    with torch.no_grad():
        for (images, labels) in tqdm(test_sampler):
            
            # Move the images and labels to the device (GPU/CPU) used for training
            images = images.to(device)
            labels = labels.to(device)

            # Make predictions on the test images
            outputs = model.forward(images)
            
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            # Store the predicted labels and true labels for the current test image
            pred_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Store the predicted probabilities for the current test image
            pred_probs.extend(probs.cpu().numpy())
                

    #sklearn function for a confusion matrix
    evaluation(pred_labels,true_labels,pred_probs)

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
    ax1.axhline(y = mean_loss_test.cpu(), color = 'r', linestyle = 'dashed')
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_val], label="Validation", color="green")
    ax2.axhline(y = mean_loss_test.cpu(), color = 'r', linestyle = 'dashed')
   
    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # Adding random samples
    random_samples = [random.randrange(0,len(pred_probs)) for _ in range(5)]
    print("###################################")
    print("Random samples of model predictions")
    print("###################################")
    for el in random_samples:
        predicted = label_dict[pred_labels[el]]
        actual = label_dict[true_labels[el]]
        probs = [f"{label_dict[disease]}: {round(prob*100,2)}%" for disease, prob in enumerate(pred_probs[el])]
        print()
        print(f"Predicted label: {predicted}, actual label: {actual}")
        print("With class probabilities:")
        for p in probs:
            print(p, end = ', ')
        print()
        print("###################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=5, type=int
    )
    parser.add_argument(
        "--augmentation", help="whether the model should be run on augmented data", default=0, type=int
    )
    parser.add_argument(
        "--augmentation_bal", help="whether the model should be run on augmented data balanced", default=0, type=int
    )
    parser.add_argument(
        "--validation_ratio", help="how big should the validation set be", default=0.2, type=float
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument("--balanced", help="whether to balance batches for class labels", default=1, type=int)
    args = parser.parse_args()

    main(args)
