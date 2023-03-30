from sklearn.model_selection import train_test_split
from image_dataset import ImageDataset
from pathlib import Path
import numpy as np


def validation_split(test_size=0.2, path="data/"):
    train_dataset = ImageDataset(Path(f"{path}X_train.npy"), Path(f"{path}Y_train.npy"))
    X_train, X_validation, Y_train, Y_validation = train_test_split(train_dataset.imgs, 
                                                                    train_dataset.targets,
                                                                    test_size = test_size,
                                                                    random_state = 1,
                                                                    stratify = train_dataset.targets)
    np.save(f"{path}X_train_split.npy", X_train)
    np.save(f"{path}X_validation_split.npy", X_validation)
    np.save(f"{path}Y_train_split.npy", Y_train)
    np.save(f"{path}Y_validation_split.npy", Y_validation)
