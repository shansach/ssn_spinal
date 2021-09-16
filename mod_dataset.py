import math
import os
import numpy as np

from sklearn.model_selection import train_test_split






if __name__=="__main__":


    train_x = np.load('train_x_mel40.npy')
    train_y = np.load('Y_train.npy')
    x_test = np.load('test_x_mel40.npy')
    y_test = np.load('Y_test.npy')
    

    train_x = train_x[:,:,:500,:]
    x_test =x_test[:,:,:500,:]


    x_train , x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.3)
    print(train_x.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    np.save("x_train_ssn", x_train)
    np.save("y_train_ssn", y_train)
    np.save("x_val_ssn", x_val)
    np.save("y_val_ssn", y_val)
    np.save("x_test_ssn", x_test)
    np.save("y_test_ssn", y_test)










