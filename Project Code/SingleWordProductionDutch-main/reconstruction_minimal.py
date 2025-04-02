import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

#Import MelFilterBank from the mel module
sys.path.insert(0, r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Code\SingleWordProductionDutch-main')
import reconstructWave as rW
import MelFilterBank as mel


def createAudio(spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01):
    """
    Create a reconstructed audio wavefrom
    
    Parameters
    ----------
    spectrogram: array
        Spectrogram of the audio
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram was calculated
    frameshift: float
        Shift (in seconds) after which next window was extracted
    Returns
    ----------
    scaled: array
        Scaled audio waveform
    """
    
    # Initializes log mel-spec transformation function
    # Specify the spec (total time sampled), number of freq components, audio sampleing rate
    mfb = mel.MelFilterBank(int((audiosr*winLength)/2+1), spectrogram.shape[1], audiosr)
    
    # Establish how many folds you like to create out of mel spec
    nfolds = 10
    
    # Computes the time window lengths based on the number of data fold we want (row is time)
    hop = int(spectrogram.shape[0]/nfolds)
    
    # Create an 1D list array (vector) to append the audio for corresponding mel spec values for each time window
    # Number of values in vector should correspond to the number of nfolds
    rec_audio = np.array([])
    
    # Returns mel-spectrogram from log-mel spectrogram
    for_reconstruction = mfb.fromLogMels(spectrogram)
    for w in range(0,spectrogram.shape[0],hop): #each w is the starting point of a window
        print(w)
        
        # Extracts a window of the given Mel-spectrogram, the min() function is used if we need to truncate the window
        # Rows = time, columns = Mel-Freq, values = amplitude 
        spec = for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
        
        # Reconstructs the waveform audio (person's voice) from the windowed spectrogram segment using the method described by Griffin paper
        rec = rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        
        # Append the audio waveform to the collection of audio waveforms for a single spectrogram
        rec_audio = np.append(rec_audio,rec)
    
    # Scales the reconstructed audio output based on abs. max value of the amplitude of recorded
    #Normalizes between -1 and 1 then multiplies the 16-bit audio value
    scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
    return scaled


if __name__=="__main__":
    
    #Set local paths
    path_output = r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data\transformed_data'
    result_path = r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data\transformed_data'
    
    #set a list of participant ids
    pts = ['sub-%02d'%i for i in range(1,11)]
    
    # Define window length (50ms), frameshit (10ms), and  audio sampling rate
    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000
    
    # Define number of folds to break up the spectrogram
    nfolds = 10
    kf = KFold(nfolds,shuffle=False)
    
    # Sets linear regression function; ignore n_jobs for computational efficiency
    est = LinearRegression(n_jobs=5)
    
    #Set PCA algorithm as a function
    pca = PCA()
    
    #Set number of PCA components
    numComps = 50
    
    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    # Rows = participants, Cols = number of folds "windows", Depth = time window, values = PCA component value
    allRes = np.zeros((len(pts),nfolds,23))
    
    # Rows = participants, Cols = number of folds, values = total variance explained by PCA components
    explainedVariance = np.zeros((len(pts),nfolds))
    
    # number of times we randomize the order of spectrograms
    numRands = 1000
    
    # Rows = participants, Cols = number of spectro randimizations, Depth = time window, values = total variance explained by PCA components
    randomControl = np.zeros((len(pts),numRands, 23))

    for pNr, pt in enumerate(pts):
        #Load the transformed data from local directory
        spectrogram = np.load(os.path.join(path_output,f'{pt}', 'Mel_spec.npy'))
        data = np.load(os.path.join(path_output,f'{pt}','EEG_feat.npy'))
        labels = np.load(os.path.join(path_output,f'{pt}', 'procWords.npy'))
        featName = np.load(os.path.join(path_output,f'{pt}','feat_names.npy'))
        
        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        #Each fold is # of time windows in EEG features/# of folds
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        
        # Train and test are collections of consecutive indices to split data
        for k,(train, test) in enumerate(kf.split(data)):

            #Z-Normalize with mean and std from the training data
            #data[train,:] is of shape len(train)x1098 where N # of time windows from EEG_feat
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            trainData=(data[train,:]-mu)/std
            #data[test,:] is of shape len(test)x1098 where N # of time windows from EEG_feat
            testData=(data[test,:]-mu)/std

            #Fit PCA to training data (NxM where N = ith time window and M = mth electrode location)
            pca.fit(trainData)
            #Get percentage of explained variance by selected components
            explainedVariance[pNr,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
            #Tranform data into component space using first 50 PCA components
            #The transpose of PCA Matrix is 1098 x 50 where 1098 is # of EEG stacked windows and 50 is # of PCA components
            #The train and test are len(train)x1098 and len(test)x1098, respectively
            #Output is the spectrogram moved to a the PCA mapping size: len(train)x50
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)
            
            #trainData is len(train)x50 (low rank)
            #Fit the regression model using X (trainData) and Y (real spectrogram[train])
            # trainData = (25614, 50) and spectrogram[train,:] (25614, 23)
            est.fit(trainData, spectrogram[train, :])
            
            # Outputs Coefficients of size (23,50) it is transposed when applied to data
            # est.coef_.shape

            #Predict the reconstructed spectrogram for the test data
            #Shape of reconstructed spectrogram is (2846, 23)
            rec_spec[test, :] = est.predict(testData)

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Estimate random baseline
        for randRound in range(numRands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(spectrogram.shape[0]*0.1),int(spectrogram.shape[0]*0.9)))
            #Swap the dataset on the splitting point 
            shuffled = np.concatenate((spectrogram[splitPoint:,:],spectrogram[:splitPoint,:]))
            #Calculate the correlations
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                
                # Output pearson's correlation r (r) and associated p-value (p)
                r, p = pearsonr(spectrogram[:,specBin], shuffled[:,specBin])
                
                # Append r value to the particpant, randomization round, and time window
                randomControl[pNr, randRound,specBin]=r


        #Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)
        np.save(os.path.join(result_path,f'{pt}_predicted_spec.npy'), rec_spec)
        
        #Synthesize waveform from spectrogram using Griffin-Lim
        reconstructedWav = createAudio(rec_spec,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}_predicted.wav'),int(audiosr),reconstructedWav)

        #For comparison synthesize the original spectrogram with Griffin-Lim
        origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)

    #Save results in numpy arrays          
    np.save(os.path.join(result_path,'linearResults.npy'),allRes)
    np.save(os.path.join(result_path,'randomResults.npy'),randomControl)
    np.save(os.path.join(result_path,'explainedVariance.npy'),explainedVariance)
    
test = np.load(r'C:\Users\pseco\Documents\ECE 542 Repositories\ECE 542 Project\ECE-542-Project\Project Data\transformed_data\sub-01_predicted_spec.npy')
