# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:54:26 2024

@author: Andrew
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def train_supervised(training_inputs, training_outputs, network, epochs, learning_rate, 
                     wtdecay,batch_size, loss_function, print_interval, device): 
    #initalize dataset
    train_dataset = torch.utils.data.TensorDataset(training_inputs,training_outputs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    end_time = 1e6
    if epochs < 0:
        #add timed training option if negative epochs passed, consider it seconds
        end_time = -epochs
        epochs = int(1e9)
        
    track_losses = []
    start = time.time()
    print('timer started')
    network = network.to(device)
    #train
    for epoch in range(1, epochs):
        for batch_idx, (X, y) in enumerate(train_loader):
            data = X
            output = network(data)
            loss = loss_function(output, y.to(device).unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        training_loss = loss.item()**.5
        track_losses.append(training_loss)
        
        if time.time()-start > end_time:
            print('epoch: %4d training loss:%10.3e time:%7.1f'%(epoch, training_loss, time.time()-start))
            break
        if epoch % print_interval == 0:
            print('epoch: %4d training loss:%10.3e time:%7.1f'%(epoch, training_loss, time.time()-start))
    return network, track_losses

class vega(nn.Module):
    def __init__(self, input_size):
        super(vega, self).__init__()
        self.hidden_size = 10
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.encode = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            self.dropout,
            self.activate,
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )
        
    def forward(self, x):
        y = self.encode(x)
        return y
    
def model_eval_supervised(X_train, X_test, y_train, y_test, model, device=None):
    # Move data to the specified device if needed
    if device:
        X_train_th = torch.FloatTensor(X_train).to(device)
        X_test_th = torch.FloatTensor(X_test).to(device)
    
    # Evaluate model predictions on training and testing data
    with torch.no_grad():
        if isinstance(model, torch.nn.Module):  # If the model is a PyTorch model
            y_train_pred = model(X_train_th).cpu().detach().numpy()
            y_test_pred = model(X_test_th).cpu().detach().numpy()
        else:  # If the model is a scikit-learn model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
    
    # Reshape predictions to handle 1D output
    
    if y_train.ndim == 1 or y_train.shape[1] == 1:
        y_train_pred = y_train_pred.reshape(-1, 1)
        y_test_pred = y_test_pred.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    
    # Compute evaluation metrics for training and testing sets
    metrics_train = {}
    metrics_test = {}
    
    for i in range(y_train_pred.shape[1]):
        metrics_train[f'MAE_{i}'] = mean_absolute_error(y_train_pred[:, i], y_train[:, i])
        metrics_train[f'MSE_{i}'] = mean_squared_error(y_train_pred[:, i], y_train[:, i])
        metrics_train[f'RMSE_{i}'] = np.sqrt(mean_squared_error(y_train_pred[:, i], y_train[:, i]))
        metrics_train[f'R2_{i}'] = r2_score(y_train_pred[:, i], y_train[:, i])
        
        metrics_test[f'MAE_{i}'] = mean_absolute_error(y_test_pred[:, i], y_test[:, i])
        metrics_test[f'MSE_{i}'] = mean_squared_error(y_test_pred[:, i], y_test[:, i])
        metrics_test[f'RMSE_{i}'] = np.sqrt(mean_squared_error(y_test_pred[:, i], y_test[:, i]))
        metrics_test[f'R2_{i}'] = r2_score(y_test_pred[:, i], y_test[:, i])
    
    # Compute mean values of evaluation metrics
    mean_metrics_train = {key: np.mean(value) for key, value in metrics_train.items()}
    mean_metrics_test = {key: np.mean(value) for key, value in metrics_test.items()}
    
    return mean_metrics_train, mean_metrics_test

def plt_perf(y_test_pred, y_test, title):
    plt.figure(figsize=(12, 6))
    
    # Scatter plot of predicted vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_test_pred, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title} Testing Data - Predicted vs Actual')
    
    # Histogram of residuals
    plt.subplot(1, 2, 2)
    residuals_test = y_test - y_test_pred
    plt.hist(residuals_test, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{title} Residuals Histogram')
    
    plt.tight_layout()
    plt.show()
#%%
data = pd.read_excel('data/default of credit card clients.xls',header=1)

if torch.cuda.is_available():
    print('CUDA available')
    print(torch.cuda.get_device_name(0))
else:
    print('CUDA *not* available')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.FloatTensor(np.array(data.iloc[:,:-1])).to(device)
y = torch.Tensor(np.array(data.iloc[:,-1])).to(device)

seed = torch.randint(10, 8545, (1,)).item()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed-1)

epochs = 5000
learning_rate = 2e-3
wtdecay = 1e-6
batch_size = 1024
print_interval = 50
loss_function = nn.BCELoss()

torch.manual_seed(seed)
#%%
network = vega(x.shape[1]).to(device)
#%%
network, losses = train_supervised(x_train, y_train, network, epochs, learning_rate,
                                   wtdecay,batch_size,loss_function,print_interval,device)
#%%
y_train = np.array(y_train.cpu().detach())
y_test = np.array(y_test.cpu().detach())
x_train = np.array(x_train.cpu().detach())
x_test = np.array(x_test.cpu().detach())
[train_metrics,test_metrics] = model_eval_supervised(x_train,x_test,y_train,y_test,network,device)

#%%
# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Test the model
y_pred = model.predict(x_test)

# Evaluate the model's performance
[lm_train_metrics,lm_test_metrics] = model_eval_supervised(x_train,x_test,y_train,y_test,model,device)

#%%
metrics = ['MAE_0', 'MSE_0', 'RMSE_0', 'R2_0']
num_metrics = len(metrics)

fig, axs = plt.subplots(num_metrics, 2, figsize=(12, 8))

for i, metric in enumerate(metrics):
    axs[i, 0].bar(['NN'], [train_metrics[f'{metric}']], color='blue', label='Neural Network')
    axs[i, 0].bar(['LR'], [lm_train_metrics[f'{metric}']], color='orange', label='Linear Regression')
    axs[i, 0].set_ylabel(metric)
    axs[i, 0].set_title(f'Train {metric}')

    axs[i, 1].bar(['NN'], [test_metrics[f'{metric}']], color='blue', label='Neural Network')
    axs[i, 1].bar(['LR'], [lm_test_metrics[f'{metric}']], color='orange', label='Linear Regression')
    axs[i, 1].set_ylabel(metric)
    axs[i, 1].set_title(f'Test {metric}')

for ax in axs.flat:
    ax.legend()

plt.tight_layout()
plt.show()

# Plot predictions for testing data

y_test_pred = model.predict(x_test)
plt_perf(y_test_pred,y_test,'linear model')

y_test_pred = network(torch.FloatTensor(x_test).to(device)).cpu().detach().numpy()
plt_perf(y_test_pred.squeeze(1),y_test,'neural network')

