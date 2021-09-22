import numpy as np
from sklearn import preprocessing
from hmmlearn import hmm
import sys

sys.path.append('/mnt/fastdata/acp20asl/acp20asl/TreeHMM/')


def spikeRasterToSpikeTimes(spikeRaster,binsize=1):
    # from a spikeRaster create a neurons list of lists of spike times
    nNeurons,tSteps = spikeRaster.shape
    nrnSpikeTimes = []
    # multiply by binsize, so that spike times are given in units of sampling indices
    bins = np.arange(tSteps,dtype=int)*binsize
    print('bintype', type(bins[0]));
    for nrnnum in range(nNeurons):
        # am passing a list of lists, convert numpy.ndarray to list,
        #  numpy.ndarray is just used to enable multi-indexing
        
        nrnSpikeTimes.append( (bins[spikeRaster[nrnnum,:] != 0]).tolist() )
    print('nrnspiketimes' , type(nrnSpikeTimes[0][0]))
    return nrnSpikeTimes

if __name__ == '__main__':
  a = np.array(([1,0],[0,1]))
  #a = np.array(([1,0],[0,1],[0,0],[1,1]))
  spikeRaster_transpose = np.tile(a,(1000,1))
  
  arr_spike_pattern = [np.array2string(spike_pattern,separator='') for spike_pattern in spikeRaster_transpose]
  print('converted to arr_spike_pattern')
  le = preprocessing.LabelEncoder()
  le.fit(arr_spike_pattern)
  print('data fitted')
  print('number of patterns',le.classes_.shape)
  le_transformed = le.transform(arr_spike_pattern)
  print('data transformed')
  print(spikeRaster_transpose.shape)
  print('running Multinomial')
  remodel = hmm.MultinomialHMM(n_components=2,n_iter = 100)
  reshaped_ = le_transformed.reshape(-1,1)
  print('reshape',reshaped_.shape)
    #train_set = reshaped_[0:95000]
   # test_set = reshaped_[5000:1000000]
    #print('train_set',reshaped_)
   # print('test_set',test_set.shape)
    #init_params='ste'
    #print(train_set_1[0:5])
    #for i in range(1,100,5):
  remodel.fit(reshaped_)
  Z2 = remodel.predict(reshaped_)
  train_score = remodel.score(reshaped_)
  print('train_score',train_score)
  print('transmat', remodel.transmat_ )
  
  import EMBasins as EMBasins
  EMBasins.pyInit()
  
  nModes = 2
  niter = 100
  binsize = 1
  nrnSpiketimes = spikeRasterToSpikeTimes(np.transpose(spikeRaster_transpose),binsize=binsize)
  #print(nrnSpiketimes)
  params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,stationary_prob,train_logli_this,test_logli_this = \
                    EMBasins.pyHMM(nrnSpiketimes, np.ndarray([0]), np.ndarray([0]), float(binsize), nModes, niter)
  train_logli = train_logli_this.flatten()
  test_logli = test_logli_this.flatten()
  print('train logli ',train_logli)
  print('trans ',trans)