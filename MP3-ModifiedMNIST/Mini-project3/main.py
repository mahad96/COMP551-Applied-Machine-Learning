"""
Mini-Project 3: COMP551 Winter 2019
Benjamin MacLellan, John McGowan, Mahad Khan

Convolutional Neural Network for digit detection in MNIST dataset
"""

## library imports
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

## custom-made imports
from models import Net1, Net2, Net3, ResNet
from dataset_loaders import ModifiedMNISTDataSet, test_val_split

plt.close('all')
print(os.listdir('input/'))
t1 = time.time()

# Data import
train_images_path, train_labels_path = ('input/train_images.pkl', 'input/train_labels.csv')
test_images_path = 'input/test_images.pkl'

dataset = ModifiedMNISTDataSet(train_images_path, train_labels_path, test_images_path)

# number of classes
classes = tuple(range(0,10))

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 50
learning_rate = 0.001
val_split = 0.1
shuffle_data = True
seed = 0
save_model = False
model_name = 'trained_models/model_ben_mar16_'

# Split data into validation and train partitions
train_sampler, valid_sampler = test_val_split(dataset, batch_size, val_split, shuffle_data, seed)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)


# select model
model = Net1(num_classes) 


# Create model and specify loss/optimization functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t2 = time.time()
print('Time to load data and split = {:.4f} seconds'.format(t2-t1))

total_step = len(train_loader)
print('Beginning training with {} epochs'.format(num_epochs))

## run training
loss_list, batch_list, epoch_list = [], [], []
for epoch in range(num_epochs):
    for batch_i, (images, labels) in enumerate(train_loader):

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track progress
        if (batch_i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, batch_i+1, total_step, loss.item()))
        loss_list.append(loss.item())
        batch_list.append(batch_i+1)
        epoch_list.append(epoch)
        
t3 = time.time()
print('Time to train model = {:.4f} seconds'.format(t3-t2))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_i, (images, labels) in enumerate(val_loader):

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

t4 = time.time()
print('Time to evaluate model = {:.4f} seconds'.format(t4-t3))

acc = 100 * correct / total
save_name = model_name+'_accuracy_{:2.0f}'.format(acc)+'.ckpt'
if save_model:
    torch.save(model.state_dict(), save_name)

# save the training results
d = {'Batch':batch_list,'Epoch':epoch_list,'Loss':loss_list}
df = pd.DataFrame(d)
df.to_pickle(model_name+'_accuracy_{:2.0f}'.format(acc)+'TRAININGINFO')

#plots loss
plt.title('Loss over time')
plt.plot(loss_list)
plt.xlabel('Time')
plt.ylabel('Loss')
plt.show()


"""
# if we need to reload a model to do analysis
model = Net3()
model.load_state_dict(torch.load('trained_models/model_ben_mar15_accuracy_82.ckpt'))
model.eval()
"""


## run model on test images and save submission csv
test_id, test_pred = [], []
for image_id in range(0, dataset.test_images.shape[0]):
    if image_id%100 == 0:
        print('Test image {} of {}'.format(image_id,dataset.test_images.shape[0]))
    images = dataset.get_test(image_id)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    test_id.append(image_id)
    test_pred.append(predicted.item())
d = {'Id':test_id,'Category':test_pred}
df = pd.DataFrame(d)
df.to_csv('submissions/submissionv3.csv', index=False)


## to plot the kernels on the first layer
kernels = model.layer1[0].weight.detach()
fig, axarr = plt.subplots(1,kernels.size(0))
for idx in range(kernels.size(0)):
    axarr[idx].imshow(kernels[idx].squeeze())
    axarr[idx].get_xaxis().set_visible(False)
    axarr[idx].get_yaxis().set_visible(False)
