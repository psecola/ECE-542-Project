#%%
import os
import sys
import pandas as pd
import numpy as np 
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack
from sklearn.preprocessing import StandardScaler

from pynwb import NWBHDF5IO

#Import MelFilterBank from the mel module
sys.path.insert(0, r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Code\SingleWordProductionDutch-main')
import MelFilterBank as mel

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr, windowLength=0.05, frameshift=0.01):
    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise - Dutch 50hz Line Noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise - Dutch 50hz Line Noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Create feature space
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    # includes 200ms windows before and after current window as defualt
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() #Add 'F' if stacked the same as matlab
    return featStacked

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """
    Downsamples non-numerical data by using the mode
    
    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        # Depreciated - Pierce
        # newLabels[w]=scipy.stats.mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
        values, counts = np.unique(words[0:1000], return_counts=True)
        max_count_index = np.argmax(counts)
        newLabels[w] = values[max_count_index].encode("ascii", errors="ignore").decode()
    return newLabels

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = scipy.signal.windows.hann(int(np.floor(windowLength*sr + 1)))[:-1]
    # Depreciated - Pierce
    # win = scipy.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+windowLength*sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names
    
    Parameters
    ----------
    elecs: array of str
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names
    Returns
    ----------
    names: array of str
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  #Add 'F' if stacked the same as matlab


def windowingComp(eegs:dict, specs:dict, winL, frameshift, TW):
    
    # Contains all windows
    dataEEG = []
    dataSpec = []

    #Segment both EEG and Spectrogram data
    for (k1, eeg), (k2, spec) in zip(eegs.items(), specs.items()):
        
        #Calculate number of windows per segment
        nonadjustedWin = int((TW/frameshift))
        numWindowsPerSeg = int((TW-(winL-frameshift))/frameshift)
        numSegs = int(np.floor((eeg.shape[0]-numWindowsPerSeg)/(numWindowsPerSeg)))
        skippedSegs = nonadjustedWin - numWindowsPerSeg

        for seg in range(numSegs+1):
            if seg != 0:
                eegSeg = eeg[((seg*numWindowsPerSeg)+skippedSegs):(((seg+1)*numWindowsPerSeg)+skippedSegs)]
                specSeg = spec[((seg*numWindowsPerSeg)+skippedSegs):(((seg+1)*numWindowsPerSeg)+skippedSegs)]
                dataEEG.append(eegSeg)
                dataSpec.append(specSeg)
            else:
                eegSeg = eeg[(seg*numWindowsPerSeg):((seg+1)*numWindowsPerSeg)]
                specSeg = spec[(seg*numWindowsPerSeg):((seg+1)*numWindowsPerSeg)]
                dataEEG.append(eegSeg)
                dataSpec.append(specSeg)
            #test_df = eegs[0]
        eegSeg =eeg[(((numSegs+1)*numWindowsPerSeg)+skippedSegs):(eeg.shape[0])]
        specSeg = spec[(((numSegs+1)*numWindowsPerSeg)+skippedSegs):(spec.shape[0])]
        dataEEG.append(eegSeg)
        dataSpec.append(specSeg)


    return dataEEG, dataSpec

def add_zero_column_if_not_exists(df, column_name):
    """Adds a column with zeros to a DataFrame if it doesn't exist.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        column_name (str): The name of the column to add.
    """
    if column_name not in df.columns:
        df[column_name] = 0
    return df

#%% 
if __name__=="__main__":
    winL = 0.05
    frameshift = 0.01
    modelOrder = 4
    stepSize = 5
    
    # Change your directory to write files
    os.chdir(r"C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data\transformed_data")

    # Edit this to be the unique browser path to the nwb_files directory
    path_bids = r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data'
    path_output = r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data\transformed_data'
    participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')
    
    for p_id, participant in enumerate(participants['participant_id']):
        
        # Change your directory to write files
        os.chdir(os.path.join(path_output,f'{participant}'))
        
        #Load data
        io = NWBHDF5IO(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        #sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = 1024
        #audio
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = 48000
        #words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()

        print('Done Reading NWB File')

        #channels
        channels = pd.read_csv(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_channels.tsv'), delimiter='\t')
        channel_names = np.array(channels['description'])
        channels = np.array(channels['name'])
        

        print('Done Reading Channels File')

        #Extract HG features
        feat = extractHG(eeg,eeg_sr, windowLength=winL,frameshift=frameshift)

        print('Done Extracting Features')
        
        #Process Audio
        target_SR = 16000
        audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
        audio_sr = target_SR
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
        os.makedirs(os.path.join(path_output), exist_ok=True)
        scipy.io.wavfile.write(os.path.join(path_output,f'{participant}_orig_audio.wav'),audio_sr,scaled)   

        print('Done Processing Audio')

        #Extract spectrogram
        melSpec = extractMelSpecs(scaled,audio_sr,windowLength=winL,frameshift=frameshift)

        print('Done Extracting Spectrogram')

        #Align to EEG features
        words = downsampleLabels(words,eeg_sr,windowLength=winL,frameshift=frameshift)

        print('Done Downsampling Labels')

        print(melSpec.shape, feat.shape, words.shape)
        # words = words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]
        # melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
        #adjust length (differences might occur due to rounding in the number of windows)
        if melSpec.shape[0]!=feat.shape[0]:
            tLen = np.min([melSpec.shape[0],feat.shape[0]])
            melSpec = melSpec[:tLen,:]
            feat = feat[:tLen,:]
        
        print('Done aligning labels and and audio spectrogram')
        
        #Save everything - check if file exists first
        np.save(os.path.join(path_output,f'{participant}','EEG_feat2.npy'), feat)
        np.save(os.path.join(path_output,f'{participant}', 'procWords2.npy'), words)
        np.save(os.path.join(path_output,f'{participant}', 'Mel_spec2.npy'), melSpec)
        np.save(os.path.join(path_output,f'{participant}','Brain_Regions.npy'), channels)
        
        print(participant, 'Done')

        
# %%

path_bids = r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data'
path_output = r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data\transformed_data'
participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')

eegs = {}
specs = {}
channel_array = np.empty((0, 2), dtype=str)

for p_id, participant in enumerate(participants['participant_id']):
    channels = np.load(os.path.join(path_output,participant,'Brain_Regions.npy'), allow_pickle=True)
    eeg_file = np.load(os.path.join(path_output,participant,'EEG_feat2.npy'))
    eeg_BR = pd.DataFrame(eeg_file, columns=channels)
    eegs[p_id] = eeg_BR

    specs[p_id] = np.load(os.path.join(path_output,participant,'Mel_spec2.npy'))
    
    #channels
    channels2 = pd.read_csv(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_channels.tsv'), delimiter='\t')
    channel_names = np.array([channels2['name'], channels2['description']])
    channel_array = np.vstack((channel_array, channel_names.T))

# Convert to DataFrame and get unique rows
channelArrayDF = pd.DataFrame(channel_array, columns=['name', 'description'])
channelArrayDF = channelArrayDF.drop_duplicates()

# Convert to dictionary
channelDict = channelArrayDF.groupby(['description']).apply(lambda x: x['name'].tolist()).to_dict()


#%%

BR = []
for key in eegs:
    for col in eegs[key].columns:
        BR.append(col)

# Get unique Electode names
BR = list(set(BR))

# Add zero columns to each DataFrame if it does not exist
for key in eegs:
    for region in BR:
        eegs[key] = add_zero_column_if_not_exists(eegs[key], region).sort_index(axis=1)

#%% 

# Initialize the StandardScaler
scaler = StandardScaler()

# Normalize Each Column
for key in eegs:
    df = eegs[key]

    # Fit and transform the data. Overwrite OG dataframe with scaled data
    scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    eegs[key] = scaled_data


#%% 

# Consolidate electrode regions into brain regions using max amplitude
# Iterate through all the EEG dataframes to consolidate the electrode regions
for key in eegs:
    df = eegs[key]
    new_df = pd.DataFrame()
    for key2 in channelDict:
        cols = channelDict[key2]
        if len(cols) == 1:
            agg_cols = pd.DataFrame(df[cols], columns=[key2])
        else:
            agg_cols = pd.DataFrame(df[cols].max(axis=1), columns=[key2])
        new_df = pd.concat([new_df, agg_cols], axis=1)
        
    new_df.to_csv(os.path.join(path_output,''.join(('sub-',str(key+1).zfill(2))),'EEG_BrainRegions.csv'), index=False)
    print(new_df.shape)

    eegs[key] = new_df




#%%

# word was presented on the screen for a duration of 2 seconds. fixation cross was displayed for 1 second
# Segment data into windows corresponding to the word presentation and fixation cross
# Given the existing windowing of data some of the segments from one word will bleed into the subsequent word
# Segments are adjusted for that fact
winL = 0.05
frameshift = 0.01
TW = 3 # 2 seconds of word + 1 seconds of fixation cross

# Create windows containing EEG and Spectrogram corresponding to each displaying of a word and focus cross
dataEEG, dataSpec = windowingComp(eegs, specs, winL, frameshift, TW)


# %%
