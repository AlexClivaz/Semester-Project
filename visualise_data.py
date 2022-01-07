import os
import numpy as np
import scipy.signal as sig
import scipy.fftpack as t
import collections
import pickle
import pywt
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


seizure_types = ['FNSZ','GNSZ','CPSZ','TNSZ','ABSZ']
raw_paths = ['./data/v1.5.2/raw/dev', './data/v1.5.2/raw/train']
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])
rng = np.random.RandomState(20)

FS = 250

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
    return torch.cat((avg, rect_avg, peak2peak, std, skews, kurtoses, var), dim=1)

def extract_cwt(x):
    # plt.figure(3)
    nb_width = 50
    cwt = np.zeros((x.shape[0]*nb_width, x.shape[1]))
    plt.figure(7)
    for i in range(2):
        cwt_tmp = sig.cwt(x[i,:], sig.ricker, widths=np.logspace(-2, 4, nb_width, base = 2))
        # cwt_tmp = cwt_tmp.reshape(1,-1).squeeze()
        # cwt[i*nb_width:(i+1)*nb_width,:] = cwt_tmp
        #plt.subplot(4,5,i+1)
        # plt.imshow(cwt_tmp, cmap='PRGn', aspect='auto',vmax=abs(cwt_tmp).max(), vmin=-abs(cwt_tmp).max())
        # plt.xlabel("Time [s]")
        # plt.ylabel("Width")
    # plt.savefig('./data/CWT_4/'+s_type+'_'+str(np.around(np.random.rand(),3))+'.png')
    # cwt = (cwt-cwt.mean())/cwt.std()
    # test = cwt.reshape(19,-1)
    # pca = PCA(n_components=nb_width)
    # cwt_t = pca.fit_transform(cwt)

    # cwt_n = pca.inverse_transform(cwt_t)
    # test_n = cwt_n.reshape(19,-1)
    # plt.figure(6)
    # plt.plot(test[0,:])
    # plt.figure(7)
    # plt.plot(test_n[0,:])
    # plt.show()
    
    return cwt_tmp

def extract_dwt(x):
    w = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(x, w, level=5)
    plt.figure(6)
    plt.subplot(7,1,1)
    plt.plot(np.linspace(0,1,500),x[6,:].T)
    plt.ylabel("Voltage [uV]")
    for iter, coeff in enumerate(coeffs):
        plt.subplot(7,1,iter+2)
        if iter == 0:
            dwt = coeff
        elif iter < 4:
            dwt = np.concatenate((dwt, coeff), axis=1)
        a = pywt.idwt(coeff[6,:].T, None, w)
        plt.plot(np.linspace(0,1,a.shape[0]),a)
        plt.ylabel("Voltage [uV]")
        print(coeff.shape)
    plt.xlabel("Time [s]")
    print(dwt.shape)
    plt.figure(5)
    plt.plot(dwt[3:6,:].T)
    return dwt

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


freq_noise, qual_factor = 60, 35
b_notch, a_notch = sig.iirnotch(freq_noise, qual_factor, FS)
sos = sig.butter(4, 90, 'lowpass', fs=FS, output='sos')

for type in seizure_types:
    print(type)
    drap = 10
    liste = []
    for set in raw_paths :
        for dir_file in os.listdir(os.path.join(set, type)) :
            if np.random.rand() < 0.08 and drap > 0:
                drap -= 1
                file = os.path.join(set, type, dir_file)
                eeg = pickle.load(open(file, 'rb'))
                x = eeg.data
                x = x[:,:500]
                time = np.linspace(0,0.996,250)
                plt.figure(1)
                plt.plot(time,x[3:6,:250].T)
                plt.xlabel("Time [s]")
                plt.ylabel("Tension [uV]")
                x = sig.filtfilt(b_notch, a_notch, x)
                x = sig.sosfilt(sos, x)
                x = (x-x.mean())/x.std()
                plt.figure(2)
                plt.plot(time,x[3:6,:250].T)
                plt.xlabel("Time [s]")
                plt.ylabel("Tension [uV]")
                feature = feature_extractor_time(x)
                # print(feature)
                f, PSD = sig.periodogram(x, FS)
                freq_E = concat_PSD(f, PSD)
                PSD = PSD[:,f<58]
                f = f[f<58]
                plt.figure(4)
                plt.plot(f,PSD[3:6,:].T)
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Power Spectral Density [V^2/Hz]")

                X = t.fft(x)
                tfreq = t.fftfreq(500, 1/250)
                freq_u = tfreq[(tfreq>0)*(tfreq<58)]
                X1 = np.abs(X)[:,(tfreq>0)*(tfreq<58)]
                # X1 = (X1-X1.mean())/X1.std()
                plt.figure(3)
                spect, freq, _ = plt.magnitude_spectrum(x[2], Fs=FS,scale='dB')
              
                # spect = 20*np.log10(X1[3:6,:])
                # plt.plot(freq_u, X1[3:6,:].T)
                # plt.xlabel("Frequency [Hz]")
                # plt.ylabel("Norm of the FFT")
                # plt.yscale("log")
                # liste.append(X1[:,28])
                # if len(liste) == 5:
                #     for elem in liste:
                #         plt.plot(elem, '--')
                #     plt.show()

                # extract_cwt(x)
                # dwt = extract_dwt(x)

                plt.show()
