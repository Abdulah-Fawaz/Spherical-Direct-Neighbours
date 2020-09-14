#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:02:40 2020

@author: fa19
"""


import numpy as np
import torch
import torch.nn as nn


import matplotlib.pyplot as plt

import os
import sys
sys.path.append('/home/fa19/Documents/dHCP_Data') # i append this path because dta loaders and make datautils are located there. 

from make_data_utils import raw_to_rot

from MyDataLoader import My_dHCP_Data

from test_models import Linear_simple, Convolutional_simple, Simple_Convolution_with_dropout, Conv_with_nebs, FC_one


import nibabel as nb 


"""

GET DEVICE & MODEL

"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device is ', device)


#model = DownNet_mat_2ring(3,1).to(device)  #downNet_mat_2ring means it has a "kernel size" of "5" instead of "3" i.e a double ring around each vertex


"""

Training Parameters

"""

### ignore this is the alternative if i wanted a classifier

#def classifier_loss(guess, true):
#    total_loss = 0        
#    proportion = [0.13,0.87] #prop that are 0, 1
#
#    for i in range(len(guess)):
#        g = guess[i]
#        t = true[i]
#        loss = nn.BCELoss()(g,t)
#        n = proportion[torch.argmax(t).item()] 
#        
#        loss = loss * n
#        total_loss += loss
#    total_loss = total_loss / len(guess)
#    return total_loss


batch_size = 1 # batch size can be any number

learning_rate = 1e-3


def weighted_mse_loss(input, target, k=1 ):
    return torch.mean((1+(target<37)* k )* (input - target) ** 2)

criterion = nn.L1Loss()

numberOfEpochs = 150


print("Batch Size is ", batch_size)
print("Loss Function is ", criterion)
print("Learning Rate is ", learning_rate)
print("Total Number of Epochs is ", numberOfEpochs)


""" 

LIST OF OPTIMIZERS


"""



#optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

#optimizer = torch.optim.Adadelta(model.parameters())




"""

Data Parameters

"""



#####################################################################
########### END OF PARAMETERS ############################




print("Loading Data")






#full_file_arr = np.load('scan_age_regression_full_shuffled_18-08-2020.npy', allow_pickle = True)

full_file_arr = np.load('scan_age_corrected_input_arr.npy', allow_pickle = True)

K = 5
splits = np.arange(0,K+1)/K * len(full_file_arr)
splits.round()
splits = splits.astype(int)



test_results = []

rotated_test_results = []




train_losses_list = []
validation_losses_list = []
rotated_validation_losses_list = []

#import copy

list_of_all_labels = []

list_of_all_predictions = []

overall_results = []

for k in range(K):
    test_set = full_file_arr[ splits[k]:splits[k+1] ]
    
    train_set = np.concatenate( [full_file_arr[:splits[k]], full_file_arr[splits[k+1]:] ])
    
    
    
    
    train_ds = My_dHCP_Data(train_set, number_of_warps = 0, rotations=False, smoothing = False, normalisation='std', parity_choice='both')
    
    test_ds = My_dHCP_Data(test_set, number_of_warps = 0, rotations=False, smoothing = False, normalisation='std', parity_choice='left')
    
    
    MyTrainLoader  = torch.utils.data.DataLoader(train_ds,batch_size,   shuffle=True, num_workers=2)


    MyTestLoader = torch.utils.data.DataLoader(test_ds, 1 ,shuffle=False, num_workers=2)
    print("Data Sucessfully Loaded")  
        

    MyTrainLoader2 =  torch.utils.data.DataLoader(train_ds,1,   shuffle=True, num_workers=1)

    #model = Linear_simple(40962*4,1).to(device)
    #model = Convolutional_simple(4,1).to(device)
    model = FC_one(4,1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay = 0.01)




    print("Beginning Training")
    validation_losses = []
    train_losses =[]
    rotated_validation_losses = []
    
    for epoch in range(numberOfEpochs):
        running_losses  = []

        for i, batch in enumerate(MyTrainLoader):    
            model.train()
            images = batch['image']
            #images = images.reshape(images.size(0), -1)
            
            images = images.to(device)
            labels = batch['label'].cuda()

            estimates = model(images)
            
            loss = criterion(estimates, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            running_losses.append(loss.item())
        if epoch % 10 == 0:
            print(epoch, np.mean(running_losses))
        train_losses.append(np.mean(running_losses))

        if epoch%10 ==0:
            with torch.no_grad():
                running_losses  = []
                for i, batch in enumerate(MyTestLoader):    
                    images = batch['image']
                    #images = images.reshape(images.size(0), -1)
                    
                    images = images.to(device)
                    labels = batch['label'].cuda()
        
                    estimates = model(images)
                    
                    loss = criterion(estimates, labels)
        
                    running_losses.append(loss.item())
                print('validation ', np.mean(running_losses))
                





    test_outputs = []
    test_labels = []
    
    for i, batch in enumerate(MyTestLoader):
        test_images = batch['image']
            
        
        test_images = test_images.to(device)
        test_label = batch['label'].to(device)
    
    #    test_labels = test_labels.unsqueeze(1)
    
        test_output = model(test_images)
    
        test_outputs.append(test_output.item())
        test_labels.append(test_label.item())
    
     
    print('average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 
    overall_results.append(np.mean(np.abs(np.array(test_outputs)-np.array(test_labels))))
    list_of_all_predictions.extend(test_outputs)
    list_of_all_labels.extend(test_labels)
    
    plt.scatter(x = test_labels, y = test_outputs)
    plt.plot(np.arange(30,45), np.arange(30,45))
    plt.show()



    test_outputs = []
    test_labels = []

    for i, batch in enumerate(MyTrainLoader2):
        test_images = batch['image']


        test_images = test_images.to(device)
        test_label = batch['label'].to(device)

    #    test_labels = test_labels.unsqueeze(1)

        test_output = model(test_images)

        test_outputs.append(test_output.item())
        test_labels.append(test_label.item())


    print('average absolute error ', np.mean(np.abs(np.array(test_outputs)-np.array(test_labels)))) 



    plt.scatter(x = test_labels, y = test_outputs)
    plt.plot(np.arange(30,45), np.arange(30,45))
    plt.show()
    
    
    
plt.scatter(x = list_of_all_labels, y = list_of_all_predictions)
plt.plot(np.arange(25,45), np.arange(25,45))
#plt.savefig('exp_3_fig_all')
#
plt.show()
#
#np.save('experiment_3_predictions.npy', [list_of_all_labels, list_of_all_predictions])
#torch.save(model, 'model_exp_3')
#
#
#np.save('exp3_results',overall_results)




"""

I lengthened epoch from 100 to 150 
and changed constant of weighted mse loss from 5 to 1


ex2->3

changed to just l1loss, no weighting
changed to Unet - like convolution 


"""