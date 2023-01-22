import numpy as np
import IPython.display as ipd
from numpy.lib import diff
import scipy
from scipy.signal import butter, lfilter
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from scipy.io import wavfile


## Global Parameters ##
INCREMENTS=5
BIN_WIDTH= 35
LIMIT=100
LOWCUT = 10.0  # 20
HIGHCUT = 500.0 # 500
FRAME_RATE = 2000



def butter_bandpass(LOWCUT, HIGHCUT, fs, order=5):
    nyq = 0.5 * fs
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, LOWCUT, HIGHCUT, fs, order=5):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_filter(buffer):
    return butter_bandpass_filter(buffer, LOWCUT, HIGHCUT, FRAME_RATE, order=6)

def preprocess(data, crop, normalize, denoise = True):
    data = data[0:crop]
    if normalize:
        #data = data/np.max(data)
        rms_level = 0
        r = 10**(rms_level / 10.0)
        a = np.sqrt( (len(data) * r**2) / np.sum(data**2) )
        # normalize
        data = data * a
        #data = data/np.max(data)
    if denoise:
        #wavelet = pywt.Wavelet('db3')
        #coeff = pywt.wavedec(data, wavelet, mode="per",level = 3)
        #sigma = (1/0.6745) * madev(coeff[-level])*10
        #uthresh = sigma * np.sqrt(2 * np.log(len(data)))
        #coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        #data_denoised = pywt.waverec(coeff, wavelet, mode='per')
        data_denoised = np.apply_along_axis(bandpass_filter, 0, data).astype('float32')
    return data, data_denoised


def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def optimize_threshold(records, labels, increments, bin_width, limit=200):
    bin_center = bin_width/2
    bin_starts =[]
    bin_ends = []
    bin_centers = []
    for i in range(int(limit/increments)):   ###number of bands
        bin_starts.append(bin_center - bin_width/2)
        bin_ends.append(bin_center + bin_width/2)
        bin_centers.append(bin_center)
        bin_center = bin_center + increments
    bin_starts = np.array(bin_starts)
    bin_ends = np.array(bin_ends)
    bin_centers = np.array(bin_centers)
    num_windows = int(len(bin_starts))
    bandpowers = np.zeros((records.shape[0],num_windows),dtype='float32')
    for i in range(bandpowers.shape[0]): ### number of recerds
        f = np.arange(num_windows)
        bp = np.zeros_like(f, dtype=np.float32)
        total_power = bandpower(records[i,:],2000,0,1000)
        for j in range(len(f)):
            bp[j] = bandpower(records[i,:],2000,bin_starts[j],bin_ends[j])
            bandpowers[i,j] = 10*np.log10(bp[j]/total_power)
    threshold = np.linspace(-35,-1,100)
    threshold = np.double(threshold)
    true_positives = np.zeros((len(threshold),bandpowers.shape[1]))  ##th , bin
    false_positives = np.zeros((len(threshold),bandpowers.shape[1]))
    false_negatives = np.zeros((len(threshold),bandpowers.shape[1]))
    true_negatives = np.zeros((len(threshold),bandpowers.shape[1]))
    true_positive_rate = np.zeros((len(threshold),bandpowers.shape[1]),dtype=np.float32)
    false_positive_rate = np.zeros((len(threshold),bandpowers.shape[1]),dtype=np.float32)
    distances_to_optimal = np.zeros((len(threshold),bandpowers.shape[1]),dtype=np.float32)
    for t in range(len(threshold)):
        for i in range(bandpowers.shape[1]): #number of bands
            for k in range(bandpowers.shape[0]): #number of recs
                true_positives[t,i] = true_positives[t,i] + (bandpowers[k,i] >= threshold[t] and labels[k]==1)
                false_positives[t,i] = false_positives[t,i] + (bandpowers[k,i] >= threshold[t] and labels[k]==0)
                false_negatives[t,i] = false_negatives[t,i] + (bandpowers[k,i] <= threshold[t] and labels[k]==1)
                true_negatives[t,i] = true_negatives[t,i] + (bandpowers[k,i] <= threshold[t] and labels[k]==0)
            true_positive_rate[t,i] = true_positives[t,i] / (true_positives[t,i] + false_negatives[t,i])
            false_positive_rate[t,i] = false_positives[t,i] / (false_positives[t,i] + true_negatives[t,i])
            distances_to_optimal[t,i] = np.sqrt((1-true_positive_rate[t,i])**2 + (false_positive_rate[t,i])**2)
    
    min_distance_indices = np.argwhere(distances_to_optimal == np.min(distances_to_optimal))
    optimal_threshold = threshold[min_distance_indices[0,0]]
    optimal_bin_start = bin_starts[min_distance_indices[0,1]]
    optimal_bin_ends = bin_ends[min_distance_indices[0,1]]
    return optimal_threshold, optimal_bin_start, optimal_bin_ends


def predict(record, bin_start, bin_end, threshold):
    total_power = bandpower(record,2000,0,1000)
    bandpower_for_given_band = bandpower(record,2000,bin_start,bin_end)
    bandpower_ratio = 10*np.log10(bandpower_for_given_band/total_power)
    prediction = bandpower_ratio > threshold
    return prediction


def find_bandpower_ratios(record, bin_start, bin_end):
    bandpower_ratios = []
    for i in range(record.shape[0]): 
        total_power = bandpower(record[i,:],2000,0,1000)
        bandpower_for_given_band = bandpower(record[i,:],2000,bin_start,bin_end)
        bandpower_ratios.append(10*np.log10(bandpower_for_given_band/total_power))
    return bandpower_ratios


def cross_validation(records, labels, increments, bin_width, limit=200):
    predictions = []
    for i in tqdm(range(records.shape[0])):
        record_to_validate = records[i]
        records_remaining = np.delete(records, i, 0)
        labels_remaining = np.delete(labels, i, 0)
        optimal_threshold, optimal_bin_start, optimal_bin_ends = optimize_threshold(records_remaining, labels_remaining, increments, bin_width, limit)
        # print("optimal bin start: " + str(optimal_bin_start))
        # print("optimal bin ends: " + str(optimal_bin_ends))
        # print("optimal_threshold: " + str(optimal_threshold))
        prediction = predict(record_to_validate, optimal_bin_start, optimal_bin_ends, optimal_threshold)
        predictions.append(prediction)
    total_true_predictions = np.sum(np.array(np.transpose(labels) == np.array(predictions, dtype=np.int), dtype=np.int))
    accuracy = (total_true_predictions / records.shape[0]) * 100
    sensivity = (total_true_predictions)
    return accuracy, predictions, total_true_predictions, sensivity, labels


def load_records(path, normalize, crop = 20000):
    records = []
    records_denoised = []
    fh = open(path)
    for line in fh:
        line_edit = line[:-1]
        record, sample_rate = librosa.load(line_edit, sr =2000)
        record_processed, record_processed_denoised = preprocess(record, normalize = normalize, crop=crop)
        if len(record_processed) != crop:
            sys.exit("Selected crop value is larger than the minimum length in the dataset")
        records.append(record_processed)
        records_denoised.append(record_processed_denoised)
    return np.array(records), np.array(records_denoised) 


def plot_feature(feature, title, save_flag=False):
    component = 0 
    for im in range(13):
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(111)
        plt.plot(np.arange(0,13),feature[0:13,component],'r.',markersize=25) 
        plt.plot(np.arange(13,33),feature[13:33,component],'r*',markersize=25)
        plt.plot(np.arange(0,17),feature[33:50,component],'b.',markersize=25)
        plt.plot(np.arange(17,29),feature[50:62,component],'b*',markersize=25)
        ax1.tick_params(axis='x', colors='red')
        ax1.set_xlim([-1,39])
        ax1.set_xlabel("Abnormals",fontsize = 15)
        ax1.xaxis.label.set_color('red')
        ax1_tick_locations = np.arange(33)
        ax1.set_xticks(ax1_tick_locations)
        ax2 = ax1.twiny()
        ax2.tick_params(axis='x', colors='blue')
        ax2_tick_locations = np.arange(29)
        ax2.set_xticks(ax2_tick_locations)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel("Normals",fontsize = 15)
        ax2.xaxis.label.set_color('blue')
        title_wIndex = str(im+1) + title
        plt.title(title_wIndex, fontsize = 15)
        component = component + 1
        if save_flag == True:
            path = "../plots/mfcc_pca_"+str(im+1)+".png"
            plt.savefig(path,dpi=100)

            
def plot_feature_wDeltas(feature, delta_feature, delta2_feature, title, save_flag=False):
    component = 0 
    for im in range(13):
        fig, (ax1, ax1_delta, ax1_delta2) = plt.subplots(1, 3, figsize=(36,6))
        
        ax1.plot(np.arange(0,13*4),feature[0*4:13*4,component],'r.',markersize=25) 
        ax1.plot(np.arange(13*4,33*4),feature[13*4:33*4,component],'r*',markersize=25)
        ax1.plot(np.arange(0*4,17*4),feature[33*4:50*4,component],'b.',markersize=25)
        ax1.plot(np.arange(17*4,29*4),feature[50*4:62*4,component],'b*',markersize=25)
        ax1.tick_params(axis='x', colors='red')
        ax1.set_xlim([-1,39*4])
        ax1.set_xlabel("Abnormals",fontsize = 15)
        ax1.xaxis.label.set_color('red')
        ax1_tick_locations = np.arange(33*4)
        ax1.set_xticks(ax1_tick_locations)
        ax2 = ax1.twiny()
        ax2.tick_params(axis='x', colors='blue')
        ax2_tick_locations = np.arange(29*4)
        ax2.set_xticks(ax2_tick_locations)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel("Normals",fontsize = 15)
        ax2.xaxis.label.set_color('blue')
        

        ax1_delta.plot(np.arange(0*4,13*4),delta_feature[0*4:13*4,component],'r.',markersize=25) 
        ax1_delta.plot(np.arange(13*4,33*4),delta_feature[13*4:33*4,component],'r*',markersize=25)
        ax1_delta.plot(np.arange(0*4,17*4),delta_feature[33*4:50*4,component],'b.',markersize=25)
        ax1_delta.plot(np.arange(17*4,29*4),delta_feature[50*4:62*4,component],'b*',markersize=25)
        ax1_delta.tick_params(axis='x', colors='red')
        ax1_delta.set_xlim([-1,39*4])
        ax1_delta.set_xlabel("Abnormals",fontsize = 15)
        ax1_delta.xaxis.label.set_color('red')
        ax1_tick_locations = np.arange(33*4)
        ax1_delta.set_xticks(ax1_tick_locations)
        ax2_delta = ax1_delta.twiny()
        ax2_delta.tick_params(axis='x', colors='blue')
        ax2_tick_locations = np.arange(29*4)
        ax2_delta.set_xticks(ax2_tick_locations)
        ax2_delta.set_xlim(ax1_delta.get_xlim())
        ax2_delta.set_xlabel("Normals",fontsize = 15)
        ax2_delta.xaxis.label.set_color('blue')
        
        ax1_delta2.plot(np.arange(0*4,13*4),delta2_feature[0*4:13*4,component],'r.',markersize=25) 
        ax1_delta2.plot(np.arange(13*4,33*4),delta2_feature[13*4:33*4,component],'r*',markersize=25)
        ax1_delta2.plot(np.arange(0*4,17*4),delta2_feature[33*4:50*4,component],'b.',markersize=25)
        ax1_delta2.plot(np.arange(17*4,29*4),delta2_feature[50*4:62*4,component],'b*',markersize=25)
        ax1_delta2.tick_params(axis='x', colors='red')
        ax1_delta2.set_xlim([-1,39*4])
        ax1_delta2.set_xlabel("Abnormals",fontsize = 15)
        ax1_delta2.xaxis.label.set_color('red')
        ax1_tick_locations = np.arange(33*4)
        ax1_delta2.set_xticks(ax1_tick_locations)
        ax2_delta2 = ax1_delta2.twiny()
        ax2_delta2.tick_params(axis='x', colors='blue')
        ax2_tick_locations = np.arange(29*4)
        ax2_delta2.set_xticks(ax2_tick_locations)
        ax2_delta2.set_xlim(ax1_delta.get_xlim())
        ax2_delta2.set_xlabel("Normals",fontsize = 15)
        ax2_delta2.xaxis.label.set_color('blue')
        
        title_wIndex = str(im+1) + title 
        ax1.title.set_text(title_wIndex)
        ax1_delta.title.set_text(title_wIndex + " (delta)")
        ax1_delta2.title.set_text(title_wIndex+ " (delta2)")
        component = component + 1
        if save_flag == True:
            path = "../plots/mfcc_pca_"+str(im+1)+".png"
            plt.savefig(path,dpi=100)
            

def plot_singleFeature(feature, title, save_flag=False):

    fig, ax1 = plt.subplots(1, 1, figsize=(14,7))
    ax1.plot(np.arange(0,13*4),feature[0*4:13*4],'r.',markersize=25) 
    ax1.plot(np.arange(13*4,33*4),feature[13*4:33*4],'r*',markersize=25)
    ax1.plot(np.arange(0*4,17*4),feature[33*4:50*4],'b.',markersize=25)
    ax1.plot(np.arange(17*4,29*4),feature[50*4:62*4],'b*',markersize=25)
    ax1.set_xticks([])
    ax1.set_xlabel("Abnormals",fontsize = 15)
    ax1.xaxis.label.set_color('red')
    ax2 = ax1.twiny()
    ax2.set_xticks([])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel("Normals",fontsize = 15)
    plt.xticks([])
    ax2.xaxis.label.set_color('blue')
    plt.title(title, fontsize=20)

    title_wIndex = title
    if save_flag == True:
        path = "../plots/mfcc_pca_"+str(im+1)+".png"
        plt.savefig(path,dpi=100)
    
        
