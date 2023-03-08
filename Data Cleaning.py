import numpy as np

#load the data
X_test = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\X_test.npy")
X_train = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\X_train.npy")
Y_test = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\Y_test.npy")
Y_train = np.load(r"C:\Users\20212392\OneDrive - TU Eindhoven\Uni\Y2\Q3\Data Challenge 1\code_template\dc1\data\Y_train.npy")

#check if there are images that only have one shade (all black/all white/all gray)
# monochromatic = []
# for _ in range(len(X_train)):
#     if np.all(X_train[_][0] == X_train[_][0][0]):
#         monochromatic.append(X_train)
#

vals, inverse, count = np.unique(X_train, axis=0, return_inverse=True, return_counts=True)

#print(len(X_test) - len(np.unique(X_test, axis=0))) #Conclusion: test set has no duplicates
# print(len(X_train) - len(vals)) #Conclusion: training set has 3 duplicates

vals_repeated = vals[np.where(count > 1)[0]] #images that are in X_train multiple times

indices = [] #list with the indices where there are duplicates
for j in range(len(vals_repeated)):
    for i in range(len(X_train)):
        if np.all(vals_repeated[j] == X_train[i]):
            indices.append(i)

clean_X_train = np.delete(X_train, [6536, 4976, 13954], axis=0)
clean_Y_train = np.delete(Y_train, [6536, 4976, 13954])
np.save('Clean X train', clean_X_train)
np.save('Clean Y train', clean_Y_train)
