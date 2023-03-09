import numpy as np

#load the data
X_test = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\X_test.npy")
X_train = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\X_train.npy")
Y_test = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\Y_test.npy")
Y_train = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\Y_train.npy")

clean_X_train = np.delete(X_train, [6536, 4976, 13954], axis=0)
clean_Y_train = np.delete(Y_train, [6536, 4976, 13954])
np.save('Clean X train', clean_X_train)
np.save('Clean Y train', clean_Y_train)
