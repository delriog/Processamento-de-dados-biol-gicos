import re
import numpy as np
from scipy import signal
import sys
import os
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler, minmax_scale
import pandas as pd
import time

# determinação das faixas de frequências para cada rítmo
delta = (0, 4)
theta = (4, 8)   # meditação, imaginação e criatividade
alpha = (8, 12)  # relaxamento e alerta, mas não focados em algo; calma, criatividade e meditação
beta = (12, 30)  # alerta e foco em atividade específica
gamma = (30, 100)

# determinação das faixas de frequências específicas da onda Beta
beta1 = (12, 15) # foco moderado, aparece em leitura, escrita, etc
beta2 = (15, 20) # foco intenso, aparece quando solucionamos problemas e tomada de decisões
beta3 = (20, 30) # estresse e ansiedade, atividades mentais excessivas e hiperatividade

y = ('delta', 'theta', 'alpha', 'beta', 'gamma')


FS=250

def carregar_arquivo(arquivo):
    arquivo_nome = "./datasets/" + str(arquivo)
    with open(arquivo_nome) as arquivo:
        linhas = arquivo.readlines()

    data = list()
    for i, linha in enumerate(linhas):
        res = re.search('^\d{1,3},((\ -?.+?,){8})', linha)
        if res:
            cols = res.group(1)
            data.append([float(d[1:]) for d in cols.split(',') if d])

    data = np.array(data[1:])
    arquivo_nome = str(arquivo_nome).split(".")
    np.save("."+arquivo_nome[1]+".npy", data)
    return data

def butter_bandpass(data, lowcut, highcut, fs=FS, order=4):
    nyq = fs * 0.5
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return signal.filtfilt(b, a, data)

def butter_lowpass(data, lowcut, fs=FS, order=4):
    nyq = fs * 0.5
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='lowpass')
    return signal.filtfilt(b, a, data)

def butter_highpass(data, highcut, fs=FS, order=4):
    nyq = fs * 0.5
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='highpass')
    return signal.filtfilt(b, a, data)

def butter_notch(data, cutoff, var=1, fs=FS, order=4):
    nyq = fs * 0.5
    low = (cutoff - var) / nyq
    high = (cutoff + var) / nyq
    b, a = signal.iirfilter(order, [low, high], btype='bandstop', ftype="butter")
    return signal.filtfilt(b, a, data)

def print_graphs(data):
    for i in range(data.shape[0]):
        plt.plot(data[i,:])
    plt.title('Domínio do tempo')
    plt.show()
    
    for i in range(data.shape[0]):
        plt.psd(data[i,:], Fs=FS)
    plt.title('Domínio da frequência')
    plt.show()
    
    for i in range(data.shape[0]):
        plt.specgram(data[i,:], Fs=FS)
    plt.title('Espectrograma')
    plt.show()

def my_plot(data):
    if len(data.shape) == 1:
        plt.plot(data[:40])
    else:
        for ch in range(data.shape[0]):
            plt.plot(data[ch,:40])
    plt.axvline(x=4, linestyle='--', color='red')
    plt.axvline(x=8, linestyle='--', color='blue')
    plt.axvline(x=12, linestyle='--', color='orange')
    plt.axvline(x=30, linestyle='--', color='purple')

def plot_bar(data):
    colors = ('blue', 'orange', 'green', 'red', 'purple')
    plt.bar(y, data, color=colors)
