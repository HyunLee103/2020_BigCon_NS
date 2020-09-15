import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.decomposition import PCA

torch.manual_seed(2020)
np.random.seed(2020)

class AutoEncoder(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(AutoEncoder, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.fc_in(x)
        hidden = torch.sigmoid(x)
        out = self.fc_out(hidden)
        
        return torch.sigmoid(out), hidden

class DataFrameDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        return torch.from_numpy(self.data.iloc[idx,:].astype(np.int16).values)


def train_AE(data,components,epochs):

    reduction_cols = ['day_','hour','min','mcode_freq_gr','show_order']
    columns = [col for col in data.columns for rcol in reduction_cols if rcol in col]

    input = data.loc[:,columns]

    AE = AutoEncoder(components,input.shape[1])

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(AE.parameters())

    data = DataFrameDataset(input)

    dataloader = DataLoader(data,batch_size=64,shuffle=True)

    for epoch in range(1, epochs+1):

        running_loss = 0.0

        for i, x in enumerate(dataloader,0):

            optimizer.zero_grad()

            x = x.float()

            pred, _ = AE(x)
            loss = criterion(pred, x)

            loss.backward()

            running_loss += loss.item()

            optimizer.step()

        print(f'{epoch}epoch loss: {running_loss/len(dataloader):.3f}')

    torch.save(AE.state_dict(),f'AE_{components}')

def by_AE(data,components,path='AE'):
    reduction_cols = ['day_','hour','min','mcode_freq_gr','show_order']
    columns = [col for col in data.columns for rcol in reduction_cols if rcol in col]

    input = data.loc[:,columns]

    AE = AutoEncoder(components,input.shape[1])
    AE.load_state_dict(torch.load(f'{path}_{components}'))

    input = DataFrameDataset(input)

    dataloader = DataLoader(input,batch_size=64,shuffle=False)

    with torch.no_grad():

        hiddens = []

        for x in dataloader:
             
             x = x.float()

             _, hidden = AE(x) # hidden 

             hiddens.append(hidden.numpy()) # hidden - type) torch.Tensor
    
    data.drop(columns,axis=1,inplace=True)

    result = pd.DataFrame([hidden for sets in hiddens for hidden in sets])

    return pd.concat([data,result],axis=1)

def by_PCA(data,components=0.95):
    reduction_cols = ['day_','hour','min','mcode_freq_gr','show_order']
    columns = [col for col in data.columns for rcol in reduction_cols if rcol in col]

    input = data.loc[:,columns]

    pca = PCA(n_components=components)
    result = pca.fit_transform(input)

    data.drop(columns,axis=1,inplace=True)

    result = pd.DataFrame(result)

    return pd.concat([data,result],axis=1)
