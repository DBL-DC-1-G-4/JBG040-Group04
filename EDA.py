import numpy as np
import matplotlib.pyplot as plt #used for the plot
import random #used for the random images
from pathlib import Path
import os
from os import path
if not Path("EDA_figures/").exists():
    os.mkdir(Path("EDA_figures/"))

#load the data
X_test = np.load("data/X_test.npy")
X_train = np.load("data/X_train.npy")
Y_test = np.load("data/Y_test.npy")
Y_train = np.load("data/Y_train.npy")


#bar charts of the Y data
unique_Y_train, counts_Y_train = np.unique(Y_train, return_counts=True) #count the occurences from each value in Y_train
unique_Y_test, counts_Y_test = np.unique(Y_test, return_counts=True)

bar_Y_train = plt.bar(["Atelectasis", "Effusion", "Infiltration" ,"No finding","Nodule","Pneumothorax"], counts_Y_train) #display the occurences in a bar chart

plt.title('Division of the Training Class Labels')
plt.savefig('EDA_figures/bar_Y_training')
plt.show()
bar_Y_test = plt.bar(["Atelectasis", "Effusion", "Infiltration" ,"No finding","Nodule","Pneumothorax"], counts_Y_test)
plt.title('Division of the Test Class Labels')
plt.savefig('EDA_figures/bar_Y_test')
plt.show()


#image display
label_dict = {0:"Atelectasis", 1: "Effusion", 2: "Infiltration" ,3: "No finding",4: "Nodule",5: "Pneumothorax"}
mono_font = {'fontname':'monospace'} #used for changing the font in the random picture titles

def showTenRandomImages(image_data : np.array, label_data : np.array = None) -> None:
    """Show ten (pseudo-)random images in the input array and optionally show the corresponding label
    Input: image_data is X, label_data is Y"""
    #make sure that both inputs are from the same set (training or test)
    if label_data.all() != None:
        assert len(image_data) == len(label_data), "Images and labels do not correspond."

    fig = plt.figure(figsize=(5, 2))
    for i in range(10):
        ax = plt.subplot2grid((2, 5), (int(i / 5), i - int(i / 5) * 5))
        rand_im = random.randrange(0, len(image_data)) #generate what image to display
        label = label_data[rand_im]
        ax.imshow(image_data[rand_im][0], cmap='Greys') #add the image to the subplot
        ax.set_title(f"{label_dict[label]}", fontsize=7, **mono_font)
        ax.axis('off')
    plt.show()

showRand = showTenRandomImages(X_test, Y_test)
showRand