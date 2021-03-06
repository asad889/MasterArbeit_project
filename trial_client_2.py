import torch

from data_reading import *

#creating client dictionary
client_2 = {}
train_group = rsc_alloc_IID(x_features, args.clients)
trainset_ind_list = list(train_group[1])


client_2['trainset']    = X_train_features[trainset_ind_list]
client_2['Target']      = Y_train_features[trainset_ind_list]


client_2['dataset']     = Z_X_Y_train_features[trainset_ind_list]
#client_1['samples'] = len(trainset_ind_list) / args.images


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(7, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 6)


    def forward(self, x):
        x = F.relu( self.fc1 (x))
        x = F.relu( self.fc2 (x))
        x = torch.sigmoid(self.fc3 (x))

        return x

torch.manual_seed(args.torch_seed)

client_2['model'] = Net()
client_2['optim'] = optim.SGD(client_2['model'].parameters(), lr=args.lr)
client_2['model'].train


for epoch in range(1, args.epochs + 1):
    for batch_idx, data in enumerate(client_2['dataset']):
        client_2['optim'].zero_grad()
        output = client_2['model'](data[0:7])
        target = data[7]
        creteria= nn.L1Loss()
        loss = creteria(output,target)
        loss.backward()
        client_2['optim'].step()
        print('client Train Epoch: {} [{}/{}         ({:.0f}%)]\tLoss: {:.6f}'.format(
                         epoch, batch_idx , len(client_2['trainset']) ,
                              100. * batch_idx / len(client_2['trainset']), loss))

client_2_model = client_2['model']
torch.save(client_2_model.state_dict(),"fedavg_2.pt")