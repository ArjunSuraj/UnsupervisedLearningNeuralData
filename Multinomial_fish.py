import pandas as pd
import numpy as np
import scipy.io 
from hmmlearn import hmm
from sklearn import preprocessing
import sys





if __name__ == '__main__':
  component = int(sys.argv[1]) * 5
  #print(component)
  dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'
  retinaData = scipy.io.loadmat(dataFileBase+'.mat')
  spikeRaster = retinaData['bint']
  spikeRaster = np.reshape(np.moveaxis(spikeRaster,0,-1),(160,-1))       # neurons x timebins
  spikeRaster = np.transpose(spikeRaster)
  #print('spikeRaster',spikeRaster.shape)
  
  #Label encoding done by converting into String
  le = preprocessing.LabelEncoder()
  arr_spike_pattern = [np.array2string(spike_pattern,separator='') for spike_pattern in spikeRaster]
  le.fit(arr_spike_pattern)
  #print('no of patterns',le.classes_.shape)  #prints no of unique patterns
  le_transformed = le.transform(arr_spike_pattern)  #data is transformed
  
  #Perform HMM Multinomial Model
  #temporary code
  #for i in range(1,150,5):
  remodel = hmm.MultinomialHMM(n_components = component, n_iter = 100)
  reshaped_ = le_transformed.reshape(-1,1)
  remodel.fit(reshaped_)
  Z2 = remodel.predict(reshaped_)
  train_score = remodel.score(reshaped_)
  print(component,train_score/283041)
  #print(train_score)
  
  
  