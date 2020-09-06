import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.decomposition import PCA

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

        return torch.from_numpy(self.data.iloc[idx,:].values)


def train_AE(data,components,epochs):

    reduction_cols = ['day_','hour','min','mcode_freq_gr','show_order']
    columns = [col for col in data.columns for rcol in reduction_cols if rcol in col]

    data = data.loc[:,columns]

    AE = AutoEncoder(components,data.shape[1])

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(AE.parameters())

    data = DataFrameDataset(data)

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

        print(f'{epoch}epoch loss: {running_loss/len(dataloader)}')

    torch.save(AE.state_dict(),'AE')

    
def by_AE(data,path='AE'):
    reduction_cols = ['day_','hour','min','mcode_freq_gr','show_order']
    columns = [col for col in data.columns for rcol in reduction_cols if rcol in col]

    AE = AutoEncoder(30,data.shape[1])
    #AE.load_state_dict(path)

    data = DataFrameDataset(data)

    dataloader = DataLoader(data,batch_size=64,shuffle=False)

    with torch.no_grad():

        hiddens = []

        for x in dataloader:
             
             x = x.float()

             _, hidden = AE(x)

             hiddens += hidden

    return pd.DataFrame(hiddens)

def by_PCA(data,components=0.95):
    pca = PCA(n_components=components)
    result = pca.fit_transform(data)

    return pd.DataFrame(result)

'''
def dim_reduction(data,mode,components): 
    # default) 대상 - day, hour, min, mcode_freq_gr, show_order

    reduction_cols = ['day_','hour','min','mcode_freq_gr','show_order']
    columns = [col for col in data.columns for rcol in reduction_cols if rcol in col]

    if mode == 'AE':
        reduction = by_AE(data.loc[:,columns],components,100)

    elif mode == 'PCA':
        reduction = by_PCA(data.loc[:,columns],components)

    else:
        raise NotImplementedError

    data.drop(columns,axis=1,inplace=True)

    return pd.concat([data,reduction],axis=1)
'''
