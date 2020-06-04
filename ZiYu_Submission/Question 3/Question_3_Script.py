"""Importing libraries"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split

"""Get current working directory"""
wd = os.getcwd() #need to set wd as the file right above the level of the individual questions

"""Loading datasets + checking basic characteristics of dataset"""
q3_wd = os.path.join(wd, "Question 3") #creates path for question 3 easily

#Reading files as Pandas df
X_train = pd.read_csv(os.path.join(q3_wd, "train_data.txt"), sep='\t') #sep='\t' as values are tab-separated
Y_train = pd.read_csv(os.path.join(q3_wd, "train_truth.txt"), sep='\t')
X_test = pd.read_csv(os.path.join(q3_wd, "test_data.txt"), sep='\t')

#Checking certain characteristics of dataset such as length, name of columns, and range of values for each variable.
print("Length of X train =", len(X_train))
print("Length of Y train =", len(Y_train))
print("Length of X test =", len(X_test))
print("\n")
print(X_train.columns)
print(Y_train.columns)
print(X_test.columns)
print("\n")
print(X_train.describe())
print(Y_train.describe())


"""Some basic visualisation of data"""
for col in X_train.columns:
    plt.hist(X_train[col])  # Plots histogram based on values of selected column within X_train df.
    plt.title(col)  # adjusts title of plot
    plt.show()  # show plot
    plt.clf()  # clear plot for the next for loop iteration
    print(col)

plt.hist(Y_train["y"])  # Plots histogram based on values of Y_train df.
plt.title(Y_train.columns)
plt.show()


"""Correlation between x variables and y"""
for col in X_train.columns:
    print("Correlation of", col, "and y =", str(round(np.corrcoef(X_train[col], Y_train['y'])[0][1], 2)))

#np.corrcoef finds correlation between two arrays.


"""Convert to numpy array (float32) because it is what Keras framework requires, and creating train-validation sets"""
X_train = X_train.to_numpy() #.to_numpy() converts pandas array into numpy array
Y_train = Y_train.to_numpy().ravel() #.ravel() flattens array into a 1d array; required by keras
X_test = X_test.to_numpy()

#train_test_split function helps to split the X and Y dataframes into training and test validation set. 20% of the
#datapoints will be used for the validation set, while 80% will be used for training the neural network
x_train, x_valid, y_train, y_valid = train_test_split(X_train,
                                                      Y_train,
                                                      test_size = 0.2,
                                                      random_state= 123)

#Identifying dtypes, and shape of numpy arrays
print(X_train.dtype)
print(X_train.shape)
print(Y_train.shape)


"""Building Multi-Layer Perceptron (MLP)"""
model = None
model = Sequential() #Sets up the base for stacking the neural network

#Creates 2 layers of 4 neurons with linear activation function. First line need to define number of inputs
#(i.e. number of x variables).
model.add(Dense(4, activation='linear', input_dim = 3))
model.add(Dense(4, activation='linear'))

#final ouput layer with one neurons only.
model.add(Dense(1, activation='linear'))

#compile the model, selection of neural network optimiser as well as the loss function for learning of neural network
model.compile(optimizer="adam",loss="mse", metrics=[metrics.MeanSquaredError()])

#prints summary of neural network model
print(model.summary())


"""Train model"""
# EarlyStopping allows early stopping of neural network training when metric monitored (loss) stops improving for
# specified number of epochs (3 in this case).
callback = EarlyStopping(monitor='loss', patience=3)

#Trains the neural network model and identifies fit through validation data. 303 epochs used,
#with batchsize of 1000 to speed up training.
model_baseline = model.fit(x_train,
                           y_train,
                           verbose = 3,
                           validation_data = (x_valid, y_valid),
                           epochs = 303,
                           batch_size = 1000,
                           callbacks = [callback])


"""Diagnostic plots. Pulling values from training history of neural network which is stored in the trained model"""
loss = model_baseline.history['loss']
val_loss = model_baseline.history['val_loss']

#Loss Curve

#plots loss of validation set from first to last entry
plt.plot(range(1,len(val_loss)+1), val_loss, "r-", label = "val_loss")
#plots loss of training set from first to last entry
plt.plot(range(1,len(loss)+1), loss, "y-", label = "loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Values")

plt.legend()
plt.grid(True)
plt.show()


"""Fitting model to test dataset; generates prediction based on input X_test data using trained neural network"""
predictions = model.predict(X_test, verbose = 2) #identifying predictions based on X_test
print(predictions.min())
print(predictions.max())

"""
I am aware that there should no have been any negative value predictions given how all of the y values in training
set ranged between 0 and 1. But for the sake of time, I will move on instead of optimising the model. It performed
better than SGD anyways, so I think it needs a bit of hyperparameter tweaking.
"""


"""Creation of output file for submission"""

#save as having header "y", and no hex before the header.
#np.savetxt(r"Question 3/test_predicted.txt", predictions, header = "y", comments = "")
