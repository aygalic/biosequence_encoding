from .. import config

import torch
import numpy as np

from sklearn.model_selection import train_test_split


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, data1 ,transform = None):
        self.transform = transform
        self.data1 = data1
        self.datanum = len(data1)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        
        out_data1 = torch.tensor(self.data1[idx]).float() 
        if self.transform:
            out_data1 = self.transform(out_data1)

        return out_data1

def format_dataset(data, metadata, test_size = 0):
    print(data.shape)
    feature_num = data.shape[1]
    data = data.reshape(-1,1,feature_num)
    print(data.shape)

    batch_size = 32 # was 32 originally

    print('train data:',len(data))
    data_set = Mydatasets(data1 = data)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size = batch_size, shuffle=False)

    return data_set, dataloader

def encode_recon_dataset(dataloader, model, DEVICE):
    en_lat = []
    en_reconstruction = []

    model.eval()
    for _, inputs in enumerate(dataloader):
        if model.is_variational:
            latent_1, _ = model.encode(inputs.to(DEVICE))
            data_recon, _, _ = model(inputs.to(DEVICE))
        else:
            latent_1 = model.encode(inputs.to(DEVICE))
            data_recon = model(inputs.to(DEVICE))

        
        for elem in latent_1.cpu().detach().numpy():
            en_lat.append(elem)

        for elem in data_recon.cpu().detach().numpy():
            en_reconstruction.append(elem)

        
    encode_out = np.array(en_lat)
    if(not model.use_convolution):
        encode_out = encode_out.squeeze(axis=1)
    reconstruction_out = np.array(en_reconstruction).squeeze(axis=1)

    return encode_out, reconstruction_out

