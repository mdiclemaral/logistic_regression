"""
Maral Dicle Maral 2020700120 May 2021

Implementation of logistic regression from scratch.
Both batch gradient decent and mini-batch algorithms are implemented.
Runs the algorithm with cross validation.

"""
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


def file_handler(path):  # Extracts data from the given csv file to the numpy arrays

    df = pd.read_csv(path, sep=',', header=None)
    df = df[df[18].isin(['saab', 'van'])]

    Y = df[18].str.contains('van').astype(int)# saab = class0     van = class1
    Y = Y[:, np.newaxis]
    data_set = df.values
    X = data_set[:, :-1].astype(float)
    n = X.shape[1]

    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1;

    for i in range(0, n):
        X[:,i] = ((X[:,i] - X[:,i].min())/(X[:,i].max() - X[:,i].min()))

    data_size = Y.shape[0]
    for_t0 = np.ones((data_size,1))
    X = np.concatenate([for_t0, X], axis=1)
    return X, Y


def sigmoid(a): #Sigmoid function
    s  = 1 / (1 + np.exp(-a))
    return s


def predict(w, X_test): #Predicts Y values of the test data
    list = []
    for i in range(0, X_test.shape[0]):
        p = np.dot(X_test[i], w)
        z = sigmoid(p)
        if z>= 0.5:
            y_pred = 1
        else:
            y_pred = -1
        list.append(y_pred)
    return list


def E(x, y, w):  # Logistic error and derivative function

    loss = np.log(1 + np.exp(-1*y[0] * np.dot(w, x)))
    derivative = -1 * (y[0]*x) / (1 + np.exp(y[0] * np.dot(w, x)))
    return derivative, loss


def batch_gradient(step_size, X, Y):  # Batch Gradient Decent Algorithm

    n = X.shape[1]
    m = X.shape[0]
    w= np.random.rand(n)
    losses = []
    new_error = 0
    count = 0
    while(True):
        old_error = new_error
        rand_var = np.arange(m)
        np.random.shuffle(rand_var)
        X = X[rand_var]
        Y = Y[rand_var]
        loss = 0
        v = 0
        for i in range(0, m):
            y = Y[i]
            x = X[i]
            dot = np.dot(w, x)
            exp = np.exp(y * dot)
            loss_i = np.log(1+1/exp)
            g = (y * x) / (1 + exp)
            v += g
            loss += loss_i
        losses.append(loss/m)
        new_error = loss/m
        w = w + (step_size * v/m)
        if abs(new_error - old_error) < 0.0001:
            break
        count+=1
    return w, losses


def mini_batch(step_size, X, Y, b_size):  # Mini batch algorithm, depending on b_size, it can be used as schoastic gradient decent
    n = X.shape[1]
    m = X.shape[0]
    #w = np.zeros((n))
    w = np.random.rand(n)
    losses = []
    count = 0
    new_error = 0
    while(True):
        old_error = new_error
        rand_var = np.arange(X.shape[0])
        np.random.shuffle(rand_var)
        X = X[rand_var]
        Y = Y[rand_var]

        for i in range(0, X.shape[0], b_size):
            X_sub = X[i:i + b_size]
            Y_sub = Y[i:i + b_size]
            v = 0
            loss = 0
            for i in range(0, Y_sub.shape[0]):
                g, loss_i = E(X_sub[i], Y_sub[i], w)
                v += g
                loss += loss_i
            new_error = loss / m
            direction = -1 * v / m
            w = w + (step_size * direction)
        losses.append(loss / m)
        count += 1
        if abs(new_error - old_error) < 0.00001:
            break
    return w, losses

def cross_val(x, y, gradient_type, step_size):  # 5 fold cross validation

    valid = [True]*x.shape[0]
    s = round(x.shape[0]/5)
    total_accuracy = 0

    for i in range(0,5):
        if i < 4:
            valid[s*i:s*(i+1)] = [False for x in valid[s*i:s*(i+1)]]
            test_x = x[s*i:s*(i+1),:]
            test_y = y[s * i:s * (i + 1), :]

            train_x = x[valid]
            train_y = y[valid]

            valid = [True] * x.shape[0]
        else:
            valid[s * i:] = [False for x in valid[s * i:]]
            test_x = x[s * i:, :]
            test_y = y[s * i:, :]

            train_x = x[valid]
            train_y = y[valid]

        if gradient_type == "batch":
            w, loss = batch_gradient(step_size, train_x, train_y)
        else:
            w, loss = mini_batch(step_size, train_x, train_y, 1)

        y_pred = predict(w, test_x)
        y_test = test_y.T
        count = 0

file_name = 'vehicle.csv'
X, Y = file_handler(file_name)

part = sys.argv[1]
step = sys.argv[2]

if part == "part1":

    if step == "step1": #batch
        cross_val(X, Y, "batch", 0.8)
    elif step == "step2": #stoc
        cross_val(X, Y, "stochastic", 0.8)
else:
    print("Please enter a valid part name")
"""
#### THIS PART IS WITHOUT CROSS VALIDATION  ###

per = int(Y.size * 80 / 100)

training_x, test_x = X[:per, :], X[per:, :]
training_y, test_y = Y[:per, :], Y[per:, :]

#w, loss = batch_gradient(0.8, training_x, training_y)
w, loss = mini_batch(0.5, X, Y, 50)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(loss)
plt.savefig("mini-batch_0.5_50b.png")
y_pred = predict(w,test_x)
y_test = test_y.T

count = 0
for i in range(len(y_pred)):
   if y_pred[i] == y_test[0][i]:
       count+=1
print("accuracy: ",count/len(y_pred))
"""

