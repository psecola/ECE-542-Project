import pynwb

# Download MatNWB (https://neurodatawithoutborders.github.io/matnwb/)

nwbfile = pynwb.read_nwb(r'https://neurodatawithoutborders.github.io/matnwb/SingleWordProductionDutch-iBIDS/sub-01/ieeg/sub-01_task-wordProduction_ieeg.nwb')
eeg = nwbfile.acquisition['iEEG'].data[:]
audio = nwbfile.acquisition['Audio'].data[:]
words = nwbfile.acquisition['Stimulus'].data[:]
