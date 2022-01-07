import torch
import os
import pickle
import random
import pywt
import numpy as np
import pandas as pd
import scipy.fftpack as t
import scipy.signal as sig
from torch_geometric.data import InMemoryDataset, Data

NB_NEIG = 5 # Number of neighbors per node in the graph
FS = 250    # Sample frequency of the EEG
General = ['GNSZ', 'ABSZ', 'TNSZ']
Focal = ['FNSZ', 'CPSZ']

func_corr = lambda i, j, x: np.abs(np.correlate(x[i,:], x[j,:]))
func_vect = np.vectorize(func_corr, excluded=['x'])

# Create edge between electrodes using Euclidian distances between them
def create_edge_index_dist():
    weight = pd.read_csv('edge.csv', header = None)
    weight = np.array(weight)
    indices = np.diag_indices(weight.shape[0])
    weight[indices[0], indices[1]] = np.zeros(weight.shape[0])
    neig_val = np.sort(weight,axis=1)[:,-NB_NEIG]
    neig_val = neig_val.reshape((-1,1))
    weight[weight<neig_val] = 0
    edge_weight = torch.from_numpy(weight[weight>0])
    index = np.nonzero(weight)
    edge_index = torch.cat((torch.from_numpy(index[0]).unsqueeze(0),torch.from_numpy(index[1]).unsqueeze(0)),0)
    return edge_index, edge_weight


# Create edge between electrodes using correlation between them
def create_edge_index_corr(x):
    N = x.shape[0]
    corr = np.fromfunction(func_vect, shape=(N,N), dtype=int, x=x)
    neig_val = np.sort(corr,axis=1)[:,-NB_NEIG]
    neig_val = neig_val.reshape((-1,1))
    corr[corr<neig_val] = 0
    edge_weight = torch.from_numpy(corr[corr>0])
    edge_weight /= np.linalg.norm(edge_weight)
    index = np.nonzero(corr)
    edge_index = torch.cat((torch.from_numpy(index[0]).unsqueeze(0),torch.from_numpy(index[1]).unsqueeze(0)),0)
    return edge_index, edge_weight

# Extract interesting features from time domain
def feature_extractor_time(x):
    x = torch.from_numpy(x)
    avg = torch.mean(x, dim=1).reshape(-1,1)
    rect_avg = torch.mean(torch.abs(x), dim=1).reshape(-1,1)
    peak2peak = (torch.max(x,dim=1)[0] - torch.min(x,dim=1)[0]).reshape(-1,1)
    std = torch.std(x, dim=1).reshape(-1,1)
    diffs = x - avg
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), dim=1).reshape(-1,1)
    kurtoses = (torch.mean(torch.pow(zscores, 4.0), dim=1) - 3.0).reshape(-1,1)
    var = torch.sum(torch.abs(x[:,1:]-x[:,:-1]), dim=1)
    var /= (torch.max(x,dim=1)[0]-torch.min(x,dim=1)[0] * (x.shape[1]-1))
    var = var.reshape(-1,1)
    return torch.cat((peak2peak, std, skews, kurtoses, var), dim=1)

def concat_PSD(f, PSD):
    f_4 = f<4
    f_8 = f<8
    f_16 = f<16
    f_32 = f<32
    f_56 = f<56

    delta = torch.from_numpy(np.sum(PSD[:,f_4], axis=1)).reshape(-1,1)
    theta = torch.from_numpy(np.sum(PSD[:,~f_4*f_8], axis=1)).reshape(-1,1)
    alpha = torch.from_numpy(np.sum(PSD[:,~f_8*f_16], axis=1)).reshape(-1,1)
    beta = torch.from_numpy(np.sum(PSD[:,~f_16*f_32], axis=1)).reshape(-1,1)
    gamma = torch.from_numpy(np.sum(PSD[:,~f_32*f_56], axis=1)).reshape(-1,1)
    return torch.cat((delta, theta, alpha, beta, gamma), dim=1)

def extract_cwt(x):
    nb_width = 20
    CWT = np.zeros((x.shape[0], nb_width*x.shape[1]))
    for i in range(x.shape[0]):
        cwt_tmp = sig.cwt(x[i,:], sig.ricker, widths=np.logspace(-2, 4, nb_width, base = 2))
        cwt_tmp = cwt_tmp[:,:].reshape(1,-1).squeeze()
        CWT[i,:] = cwt_tmp
    return torch.from_numpy(CWT)

def extract_dwt(x):
    w = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(x, w, level=5)
    for iter, coeff in enumerate(coeffs):
        if iter == 0:
            dwt = coeff
        elif iter < 4:
            dwt = np.concatenate((dwt, coeff), axis=1)
    return torch.from_numpy(dwt)

class EEG(InMemoryDataset):
    def __init__(self, root, seizure_types, balanced='True', mode='FFT', edges='Corr',
    duration = 12, transform=None, pre_transform=None):

        self.root = root
        self.seizure_types = seizure_types
        self.balanced = True if balanced=='True' else False
        self.mode = mode
        self.edges = edges
        self.weight = []
        self.train_mask = None
        self.duration = int(duration)
        
        super(EEG, self).__init__(root, transform, pre_transform)

        self.transform, self.pre_transform = transform, pre_transform
        self.data, self.slices, self.weight, self.train_mask = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'processed_data.pt'

    def download(self):
        raise NotImplementedError('No download')

    def process(self):
        print("Creating dataset")
        data_list = []
        len_data = self.duration * FS
        nb_data = 0
        # Create notch filter to remove 60Hz noise
        freq_noise, qual_factor = 60, 20
        b_notch, a_notch = sig.iirnotch(freq_noise, qual_factor, FS)
        sos = sig.butter(4, 100, 'lowpass', fs=FS, output='sos')

        if self.edges == 'Dist':
            edge_index, edge_weight = create_edge_index_dist()

        # Read data into huge `Data` list.
        for type in self.seizure_types:
            data_type = []
            for set in self.raw_paths :
                for dir_file in os.listdir(os.path.join(set, type)) :
                    if np.random.rand() < 0.15:
                        continue
                    file = os.path.join(set, type, dir_file)
                    eeg = pickle.load(open(file, 'rb'))
                    x = eeg.data
                    x = sig.filtfilt(b_notch, a_notch, x)
                    x = sig.sosfilt(sos, x)
                    x = (x-x.mean())/x.std()
                    y = self.seizure_types.index(eeg.seizure_type)
                 
                    overlap = len_data
                    if self.duration > 5:
                        overlap = 2*FS
                    for k in range(len_data, x.shape[1], overlap):
                        x_val = x[:,k-len_data:k]

                        # Cross-correlation edges
                        if self.edges == 'Corr':
                            edge_index, edge_weight = create_edge_index_corr(x_val)
                            
                        # Time serie
                        if self.mode == 'Normal':
                            X = torch.from_numpy(x_val)
                        # Discrete Cosinus Transform
                        if self.mode == 'DCT':
                            X = t.dct(x_val, norm='ortho')
                            sorted = np.sort(abs(X))
                            thresh = sorted[:,-int(X.shape[1]/3)].reshape((-1,1))
                            X = X[abs(X) > thresh].reshape((19,-1))
                            X = torch.from_numpy(X)
                            X = (X-X.mean())/X.std()
                            features = feature_extractor_time(x_val)
                            X = torch.cat((features, X), dim=1)
                        # Fast Fourier Transform
                        if self.mode == 'FFT':
                            # rand_ind = torch.randperm(115)
                            for l in range(0, len_data, 2*FS):
                                X = torch.from_numpy(t.fft(x_val[:,l:l+2*FS]))
                                tfreq = t.fftfreq(2*FS, 1/FS)
                                X = X.abs()[:,(tfreq>0)*(tfreq<58)]
                                if l == 0:
                                    X1 = X
                                else:
                                    X1 = torch.cat((X1, X), dim=1)
                            X = (X1-X1.mean())/X1.std()
                            features = feature_extractor_time(x_val)
                            X = torch.cat((features, X), dim=1)
                        if self.mode == 'DWT':
                            X = extract_dwt(x_val)
                            X = (X-X.mean())/X.std()
                            features = feature_extractor_time(x_val)
                            X = torch.cat((features, X), dim=1)
                        if self.mode == 'CWT':
                            X = extract_cwt(x_val)
                        
                        data = Data(x=X, edge_index = edge_index, edge_weight=edge_weight, y=y, id=int(eeg.patient_id))
                        data_type.append(data)

            # if BALANCED = True, remove data to have the same number of data per class
            if self.balanced == True:
                if nb_data == 0:
                    nb_data = len(data_type)
                    data_list.append(data_type)
                if len(data_type) > nb_data:
                    nb_del = len(data_type) - nb_data
                    del_elem = random.sample(data_type, k=nb_del)
                    data_type = [elem for elem in data_type if elem not in del_elem]
                    data_list.append(data_type)
                if len(data_type) < nb_data:
                    nb_del = nb_data - len(data_type)
                    nb_data = len(data_type)
                    for current_list in data_list:
                        ind = data_list.index(current_list)
                        del_elem = random.sample(current_list, k=nb_del)
                        current_list = [elem for elem in current_list if elem not in del_elem]
                        data_list[ind] = current_list
                    data_list.append(data_type)
            else:
                data_list.append(data_type)
        tmp = []

        # Calculate weight of each class and train/test dataset
        nb_data = 0
        nb_type = len(self.seizure_types)
        w = torch.zeros(nb_type, dtype=torch.double)
        for current_list in data_list:
            tmp = tmp+current_list
            nb_data += len(current_list)
            w[current_list[0].y] += len(current_list)
        w = w / nb_data
        self.weight = (1/w).clone().detach()

        self.data, self.slices = self.collate(tmp)

        for seiz in range(nb_type):
            s_data = self.data.y==seiz
            unique_id, counts = self.data.id[s_data].unique(return_counts=True)
            _, ind = torch.sort(counts, descending=True)
            counts = counts[ind]
            unique_id = unique_id[ind]
            train_id = unique_id[0].unsqueeze(0)
            ratio =  counts[0]/len(self.data.id[s_data])
            i=1
            while ratio < 0.8:
                ratio += counts[i]/len(self.data.id[s_data])
                if ratio == 1:
                    break
                train_id = torch.cat((train_id, unique_id[i].unsqueeze(0)))
                i += 1
            if len(unique_id) == 1:
                train_ind_tmp = s_data[s_data] * (torch.rand(s_data[s_data].shape)<0.8)
            else:
                train_ind_tmp = torch.sum(torch.stack([(self.data.id[s_data] == i) for i in train_id]), dim=0, dtype=torch.bool)
            #print(train_id)
            print("train ratio : ", len(train_ind_tmp[train_ind_tmp])/len(train_ind_tmp))
            if seiz==0:
                train_ind = train_ind_tmp
            else:
                train_ind = torch.cat((train_ind, train_ind_tmp))
            
        self.train_mask = train_ind
        print("Saving dataset")
        torch.save((self.data, self.slices, self.weight, self.train_mask), self.processed_paths[0])
