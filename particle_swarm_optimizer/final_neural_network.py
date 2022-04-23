from config_nn import tanh, sigmoid
from config_nn.model import Model
from config_nn.layers import Layer
from config_nn.losses import  BinaryCrossEntropyLoss
from config_nn.pipeline import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv('/home/sachin269/Desktop/INV_KINEMATICS/Kinamatics-using-DL-main/Dataset 6dof/dataset_myplace_with_constraints_no6_merged_plus3.csv',  encoding = 'utf8')

def accuracy(y, y_hat):
    y = np.argmax(y, axis=1)
    y_hat = np.argmax(y_hat, axis=1)

    return np.mean(y==y_hat)
size = 12
angles = 6 
cross_val = 1

        
df = df.drop(['Unnamed: 0'], axis = 1)
        
x_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
y_scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))       
X = df.iloc[:,:size]; 
y = df.iloc[:,size:]; 
        
X_s = x_scaler.fit_transform(X)
y_s = y_scaler.fit_transform(y)
        
        #X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size = 0.2)
        
        #           cross-validatio

            #	print('train: %s, test: %s' % (df[train], df[test]))

X_train = X_s[:,:-1]
X_test = X_s[:,:-1]
y_train = y_s[:, -1]
y_test = y_s[:, -1] 
            

model = Model()
model.add_layer(Layer(2, 10, tanh))
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 1, sigmoid))


model.compile(BinaryCrossEntropyLoss, DataLoader, accuracy, batches_per_epoch = X_train.shape[0] // 32 + 1, n_workers=50, c1=1., c2=2.)

model.fit(X_train, y_train, 100)

y_hat = model.predict(X_test) 

print('Accuracy on test:', accuracy(y_test, y_hat))

            # training_loss = model.history['loss']
            # test_loss = model.history['val_loss']


                    
            # # Get training and test accuracy histories
            # training_acc = model.history['accuracy']
            # test_acc = model.history['val_accuracy']
            
            # # Create count of the number of epochs
            # epoch_count = range(1, len(training_loss) + 1)


# print("DONE")            