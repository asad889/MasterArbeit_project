import pandas as pd     						        #to read the file out from exce
import torch		 						            # to make tensors
from torch.utils.data import Dataset 					#to upload Data set
from sklearn.preprocessing import StandardScaler  		#for scaling values while reading file
import torch.nn as nn 							        #for implementing basic neural network
import torch.nn.functional as F
import torch.optim as optim 						    #for optimizer
from torchvision import datasets, transforms  			#for torchvision import dataset and transform
from torch.utils.data import DataLoader, Dataset    	#for Dataloader and dataset upload
import numpy as np    							        #fundemetal package for scientific computing on python



#Reading files from actual Dataset
file_out = pd.read_csv('data_usage_train_several_server.csv')	#readinng .csv files
x_features = file_out.iloc[:400, 1:8].values				    #Extracting Features
y_features = file_out.iloc[:400, 8].values				    #Extracting Labels
z_x_y_features = file_out.iloc[:400, 1:9].values
#Scaling is performed
sc = StandardScaler()							#Scaling is required to scale when Data Column will be added		
#x_train = sc.fit_transform(x)
x_train = x_features 								
y_train = y_features
z_x_y_train = sc.fit_transform(z_x_y_features)

#Creating tensors
X_train_features = torch.tensor(x_train, dtype=torch.float32)	#tensors for features
Y_train_features = torch.tensor(y_train, dtype=torch.float32)				#Tensors for Labels
Z_X_Y_train_features = torch.tensor(z_x_y_train, dtype=torch.float32)

#creating arguements
class Arguments():
    def __init__(self):
        self.dataset_count = 400     					#no. of Data sample
        self.clients = 4       					#no of clients, we need
        self.rounds = 10						#no.of rounds
        self.epochs = 2
        self.local_batches = 64
        self.lr = 0.01
        self.C = 0.9
        self.drop_rate = 0.1
        self.torch_seed = 0
        self.log_interval = 2
        self.iid = 'iid'
        self.split_size = int(self.dataset_count / self.clients)
args = Arguments()        

#making function to divide data equally among Clients

def rsc_alloc_IID(x_features, clients):
    rsc_alloc_p_clients = int(len(x_features)/ clients)
    users_dict, indeces = {}, [i for i in range(len(x_features))]
    for i in range(clients):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, rsc_alloc_p_clients, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict



