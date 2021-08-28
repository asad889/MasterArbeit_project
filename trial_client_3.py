import torch

from data_reading import *

#creating client dictionary
client_3 = {}
train_group = rsc_alloc_IID(x_features, args.clients)
trainset_ind_list = list(train_group[2])


client_3['trainset']    = X_train_features[trainset_ind_list]
client_3['Target']      = Y_train_features[trainset_ind_list]


client_3['dataset']     = Z_X_Y_train_features[trainset_ind_list]
#client_1['samples'] = len(trainset_ind_list) / args.images


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(7, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)


    def forward(self, x):
        x = F.relu( self.fc1 (x))
        x = F.relu( self.fc2 (x))
        x = torch.sigmoid(self.fc3 (x))

        return x

torch.manual_seed(args.torch_seed)

client_3['model'] = Net()
client_3['optim'] = optim.SGD(client_3['model'].parameters(), lr=args.lr)
client_3['model'].train


for epoch in range(1, args.epochs + 1):
    for batch_idx, data in enumerate(client_3['dataset']):
        client_3['optim'].zero_grad()
        output = client_3['model'](data[0:7])

        output = torch.squeeze(output)
        criterion = nn.BCELoss()
        loss = criterion(output, data[7])
        loss.backward()
        client_3['optim'].step()
        print('client Train Epoch: {} [{}/{}         ({:.0f}%)]\tLoss: {:.6f}'.format(
                         epoch, batch_idx , len(client_3['trainset']) ,
                              100. * batch_idx / len(client_3['trainset']), loss))
client_3_model = client_3['model']
#print(client_2_model)
#global_dict = client_2_model.state_dict()
#print (client_2_model.state_dict())

#torch.save(client_1_model.state_dict(),"fedavg_1.pt")