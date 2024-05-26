import torch
import numpy as np


class NeuralNetwork:
    def __init__(self , input_dim , hidden_dim , output_dim , loss_function = 'mse'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_size = output_dim
        self.loss_func = loss_function
        
        self.w1 = np.random.randn(self.input_dim , self.hidden_dim)
        self.b1 = np.zeros((1 , self.hidden_dim))
        self.w2 = np.random.randn(self.hidden_dim , self.output_dim)
        self.b2 = np.zeros((1 , self.output_dim))
        
    def feedforward(self , x): 
        self.z1 = np.dot(x , self.w1) + self.b1
        self.a1 = torch.sigmoid(self.z1)
        self.z2 = np.dot(self.z2 , self.w2) + self.b2
        
        if self.loss_func == "binary_cross_entropy":
            self.a2 = torch.softmax(self.z2)
            
        else : 
            self.a2 = torch.sigmoid(self.z2)
            
        return self.a2  
    
    def backward(self , X , y , learning_rate):
        
        m = X.shape[0]
        
        if self.loss_func == "mse" or self.loss_func == "binary_cross_entropy":
            self.dz2 = self.a2 - y # aka loss
            
        elif self.loss_func == "log_loss" : 
            self.dz2 = -(y/self.a2 - (1-y)/(1-self.a2))
            
        else:
            raise ValueError("Invalid Loss Function")
    
        # gradients    
        self.dw2 = (1 / m) * np.dot(self.a1.T, self.dz2)
        self.db2 = (1 / m) * np.sum(self.dz2, axis=0, keepdims=True)
        self.dz1 = np.dot(self.dz2, self.weights2.T) * self.sigmoid_derivative(self.a1)
        self.dw1 = (1 / m) * np.dot(X.T, self.dz1)
        self.db1 = (1 / m) * np.sum(self.dz1, axis=0, keepdims=True)
        
        # Update
        self.weights2 -= learning_rate * self.dw2
        self.bias2 -= learning_rate * self.db2
        self.weights1 -= learning_rate * self.dw1
        self.bias1 -= learning_rate * self.db1  