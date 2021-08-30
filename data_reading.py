import pandas as pd     						        #to read the file out from exce
import torch as T	 						            # to make tensors
from torch.utils.data import Dataset 					#to upload Data set
from sklearn.preprocessing import StandardScaler  		#for scaling values while reading file
import torch.nn as nn 							        #for implementing basic neural network
import torch.nn.functional as F
import torch.optim as optim 						    #for optimizer
from torchvision import datasets, transforms  			#for torchvision import dataset and transform
from torch.utils.data import DataLoader, Dataset    	#for Dataloader and dataset upload
import numpy as np
import seaborn as sns   #fundemetal package for scientific computing on python


df = pd.read_csv('data_usage_train_several_server.csv')
#Reading files from actual Dataset
class Res_all(T.utils.data.Dataset):
    def __init__(self, src_file):
        all_xy = src_file

        tmp_x = all_xy.iloc[:10000, 1:8].values  # all rows, cols [0,6)
        tmp_y = all_xy.iloc[:10000, 8].values  # 1-D required

        self.x_data = \
            T.tensor(tmp_x, dtype=T.float32)
        self.y_data = \
            T.tensor(tmp_y, dtype=T.int64)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        trgts = self.y_data[idx]
        sample = {
            'predictors': preds,
            'targets': trgts
        }
        return sample
class TEST_all(T.utils.data.Dataset):
    def __init__(self, src_file):
        all_xy = src_file

        tmp_x = all_xy.iloc[10001:10500, 1:8].values  # all rows, cols [0,6)
        tmp_y = all_xy.iloc[10001:10500, 8].values  # 1-D required

        self.x_data = \
            T.tensor(tmp_x, dtype=T.float32)
        self.y_data = \
            T.tensor(tmp_y, dtype=T.int64)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        trgts = self.y_data[idx]
        sample = {
            'predictors': preds,
            'targets': trgts
        }
        return sample

ds = TEST_all(df)
train_ds = Res_all(df)

#creating arguements
class Arguments():
    def __init__(self):
        self.dataset_count = 400     					#no. of Data sample
        self.clients = 4       					#no of clients, we need
        self.rounds = 10						#no.of rounds
        self.epochs = 2
        self.local_batches = 10
        self.lr = 0.01
        self.C = 0.9
        self.drop_rate = 0.1
        self.torch_seed = 0
        self.log_interval = 2
        self.iid = 'iid'
        self.split_size = int(self.dataset_count / self.clients)
args = Arguments()        

#making function to divide data equally among Clients


Z_X_Y_train_features = T.utils.data.DataLoader(train_ds,batch_size=args.local_batches, shuffle=False)


def rsc_alloc_IID(x_features, clients):
    rsc_alloc_p_clients = int(len(Z_X_Y_train_features)/ clients)
    users_dict, indeces = {}, [i for i in range(len(Z_X_Y_train_features))]
    for i in range(clients):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, rsc_alloc_p_clients, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict


for x in Z_X_Y_train_features:
    print(x)
def bunch(inde,num_of_images):
    sam = inde / num_of_images
    return sam