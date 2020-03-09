# ECoG class contains signal and motion data and performs preprocessing procedure

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt
from scipy.interpolate import interp1d
import pywt


def read_ECoG_from_csv(signal_file_path, motion_file_path):
    signal_df = pd.read_csv(signal_file_path)
    motion_df = pd.read_csv(motion_file_path)
    left_shoulder = motion_df.loc[0,motion_df.columns.str.contains('LSH')].values
    right_shoulder = motion_df.loc[0,motion_df.columns.str.contains('RSH')].values
    body_center = (left_shoulder - right_shoulder)/2 #It is a static point in this experiment
    motion_left_hand = motion_df[motion_df.columns[motion_df.columns.str.contains('Motion')].
                                 append((motion_df.columns[motion_df.columns.str.contains('LWR')]))].values
    motion_left_hand[:,1:] -= body_center #centering motion
    return signal_df.values, motion_left_hand

def abs_morlet(M,w = 0.5,s = 0.1):
    return np.abs(signal.morlet(M,w = 0.5,s = 0.1))

class ECoG(object):
    def __init__(self,signal_data,motion_data,downsample = True):
        start = max(signal_data[0,0],motion_data[0,0])
        end = min(signal_data[-1,0],motion_data[-1,0])
        self.centers = np.array([[187,168],[190,126],[225,180],[229,141],[264,227],[264,193],[266,152],[282,215],[303,236],
              [303,198],[306,155],[325,221],[325,178],[323,136],[343,246],[349,205],[347,162],[340,120],[365,230],
             [365,185],[362,143],[386,212],[385,164],[380,122],[400,233],[405,190],[405,147],[420,214],[424,165],[424,125],
              [455,165],[460,125]],dtype = 'float')
        #cutting signal and motion, only overlapping time left
        signal_data = signal_data[:,:][(signal_data[:,0]>=start)]
        signal_data = signal_data[:,:][(signal_data[:,0]<=end)]
        motion_data = motion_data[:,:][motion_data[:,0]>= start] 
        motion_data = motion_data[:,:][motion_data[:,0]<= end]
        M = []
        #signal and motion have different time stamps, we ned to synchronise them
        #interpolating motion and calculating arm position in moments of "signal time"
        for i in range(1,motion_data.shape[1]):
            interpol = interp1d(motion_data[:,0],motion_data[:,i],kind="cubic")
            x = interpol(signal_data[:,0])
            M.append(x)
        #downsampling in 10 times to get faster calcultions
        self.downsample = downsample
        if downsample:
            self.signal = signal_data[::10,1:]
            self.motion = np.array(M).T[::10,:]
            self.time = signal_data[::10,0]
        else:
            self.signal = signal_data[:,1:]
            self.motion = np.array(M).T[:,:]
            self.time = signal_data[:,0]
            
    #signal filtering (not sure that it works correctly)
    def bandpass_filter(self, lowcut, highcut,inplace = False, fs = 100, order=7):
        nyq =  fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order,  (low, high), btype='band',analog=False,output='sos')
        filtered_signal = np.array([sosfilt(sos, self.signal[:,i]) for i in range(self.signal.shape[1])])
        if inplace:
            self.signal = filtered_signal.T
        return filtered_signal.T

    #Generating a scalogram by wavelet transformation 
    def scalo(self, window, freqs,start,end, step = 100,lib='pywt'): #window in sec,freqs in Hz, step in ms
        div = 1
        X = self.signal[start:end,:]
        if self.downsample:
            div = 10
        window_len = int(((window * 1000 // step) + 2) * step//div)
        scalo = np.empty((X.shape[0]-window_len,X.shape[1],freqs.shape[0],(window * 1000 // step) + 2))
        for i in range(X.shape[1]):
            for j in range(window_len,X.shape[0]):
                if lib == 'scipy':
                    scalo[j-window_len,i,:,:] = signal.cwt(data = X[j-window_len:j,i],
                                                       wavelet=abs_morlet,widths = freqs)[:,::step//div] **2
                if lib == 'pywt':
                    #print(type(pywt.cwt(data = X[j-window_len:j,i],wavelet='morl',scales = freqs)[0]))
                    scalo[j-window_len,i,:,:] = pywt.cwt(data = X[j-window_len:j,i],
                                                       wavelet='morl',scales = freqs)[0][:,::step//div] **2
        return scalo, self.motion[start+window_len:end,:], self.time[start+window_len:end]
    
    

