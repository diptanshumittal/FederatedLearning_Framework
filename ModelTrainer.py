import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim
from ModelCode import CNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


BATCH_SIZE = 1000

train_data = datasets.MNIST(root = 'data', train = True, transform = ToTensor(), download = True)
train_iterator = data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)

test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor())
test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)


def train(model, iterator, optimizer, criterion): 
    epoch_loss = 0
    model.train()                
    for (x, y) in iterator:   
        x, y = x.to(device), y.to(device)        
        optimizer.zero_grad()                
        y_pred = model(x) 
        loss = criterion(y_pred, y)         
        loss.backward()              
        optimizer.step()                     
        epoch_loss += loss.item()
    return epoch_loss/len(iterator)         

def evaluate(model, iterator, criterion):      
    epoch_loss = 0
    model.eval()                                 
    with torch.no_grad():                        
        for (x, y) in iterator:  
            x, y = x.to(device), y.to(device)          
            y_pred = model(x)           
            loss = criterion(y_pred, y)      
            epoch_loss += loss.item()
        return epoch_loss/len(iterator) 

def pred(model, iterator, criterion):   
    correct = 0
    model.eval()                                 
    with torch.no_grad():                        
        for (x, y) in iterator: 
            x, y = x.to(device), y.to(device)          
            y_pred = model(x)
            y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
            y = y.cpu().detach().numpy() 
            for i in range(len(y_pred)):
                if y_pred[i] == y[i]:
                    correct += 1
    return correct


model = CNN()  
model.to(device) 
trainloss = []  
testloss = []

epochs = 1
optimizer = optim.Adam(model.parameters(), lr=0.01 )                  
criterion = nn.CrossEntropyLoss()                                            
for epoch in range(epochs):      
    trainloss.append(train(model, train_iterator, optimizer, criterion))
    testloss.append(evaluate(model, test_iterator, criterion))
    
torch.save(model, "model.pb")

