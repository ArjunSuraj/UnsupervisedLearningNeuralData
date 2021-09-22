import numpy as np
import scipy.io
import shelve, sys, os.path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import MDS
from sklearn import discriminant_analysis as DA
from sklearn.metrics import adjusted_rand_score

from EMBasins_sbatch import loadDataSet,spikeRasterToSpikeTimes,spikeTimesToSpikeRaster


np.random.seed(100)

HMM = False
shuffled = False
treeSpatial = True
crossvalfold = 2            # usually 1 or 2 - depends on what you set when fitting
if (len(sys.argv) >1):
  nSeed = int(sys.argv[1])	
else:
  nSeed = 0
dataFileBaseName = 'Learnability_data/generated_data_1'
interactionFactorList = np.arange(0.,2.,0.1)

maxModes = 150
nModesList = range(1,maxModes+1,10)  #modify modes list
if HMM:
  EMBasinsStr = ('_shuffled' if shuffled else '') + \
                ('_HMM' + (str(crossvalfold) if crossvalfold>1 else '') \
                    if HMM else '_EMBasins_full') + \
                ('' if treeSpatial else '_notree')
else :
  EMBasinsStr = ('_shuffled' if shuffled else '') \
                + '_EMBasins_full' \
                + ('' if treeSpatial else '_notree')                
                
seedsList = [0,4,6,8]           
interactionsLen = len(interactionFactorList)
seedsLen = len(seedsList)
entropies = np.zeros(interactionsLen)
logLs = np.zeros(interactionsLen)
logLsTest = np.zeros(interactionsLen)
LDAtrain = np.zeros(interactionsLen)
LDAtest = np.zeros(interactionsLen)
WTAscores = np.zeros(interactionsLen)
bestNModesList = np.zeros(interactionsLen)
meanModeEntropyAcrossTimeList = np.zeros(interactionsLen)

#dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1) + '_' + str(nSeed)

for interactionFactorIdx in range(interactionsLen):
  if interactionFactorIdx < 20:
    dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1) + '_' + str(nSeed)

  def saveBestNMode(nModesList, dataFileBase, EMBasinsStr, crossvalfold, summaryType=''):
    logLVec = np.zeros(len(nModesList))
    logLTestVec = np.zeros(len(nModesList))
    #print('nModesList - ',nModesList)
    for idx,nModes in enumerate(nModesList):
		  # only reading logL-s here, see EMBasins_sbatch.py saveFit() for details on fitting data saved
      print(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
      dataBase = shelve.open(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve','r')
      logL = dataBase['train_logli']
      logLTest = dataBase['test_logli']
      if HMM:
        logLVec[idx] = np.mean([logL[k,-1] for k in range(crossvalfold)])
        logLTestVec[idx] = np.mean([logLTest[k,-1] for k in range(crossvalfold)])
      else:
        logLVec[idx] = logL[0,-1]
        logLTestVec[idx] = max(logLTest[0])#logLTest[0,-1]
      dataBase.close()
      
      
    
    #print(logLVec,'arjun',logLTestVec)
    #plt.plot(logLVec)
    #plt.plot(logLTestVec)
    #plt.show()
def test(nModesList, dataFileBase, EMBasinsStr, crossvalfold, summaryType=''):
  #abc = pd.DataFrame(columns=['Modes','log_likelihood','interaction_factor','log_likelihood_test'])
  abc = pd.DataFrame()
  for i in (seedsList): 
    interactionFactor = []
    log_likelihood = []
    log_likelihood_test = []
    modes = []
    
    log_likelihood_avg = []
    log_likelihood_test_avg = []
    nSeed = i
    
    for interactionFactorIdx in range(interactionsLen):
      if interactionFactorIdx < 20:
        dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1) + '_' + str(nSeed)
      print(dataFileBase)
      #dataFileBase = 'Learnability_data/generated_data_1_20_6_shuffled_EMBasins_full_modes1.shelve'
      logLVec = np.zeros(len(nModesList))
      logLTestVec = np.zeros(len(nModesList))
      for idx,nModes in enumerate(nModesList):
        print('test -',dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve')
        dataBase = shelve.open(dataFileBase+EMBasinsStr+'_modes'+str(nModes)+'.shelve','r')
        logL = dataBase['train_logli']
        logLTest = dataBase['test_logli']
        if HMM:
          logLVec[idx] = np.mean([max(logL[0]) for k in range(crossvalfold)])
          logLTestVec[idx] = np.mean([max(logLTest[0]) for k in range(crossvalfold)])
        else:
          logLVec[idx] = max(logL[0]) #logL[0,-1]  #max(logL[0])#
          logLTestVec[idx] = max(logLTest[0]) #logLTest[0,-1] #max(logLTest[0]) # #
        dataBase.close()
        #print('idx',idx,' nModes ',nModes,' ',logLVec,' gaps ',logLTestVec)
        modes.append(nModes)
        interactionFactor.append(interactionFactorIdx)
        log_likelihood.append(logLVec[idx])#logL.max())
        log_likelihood_test.append(logLTestVec[idx])#logLTest.max())
        print(len(interactionFactor),len(log_likelihood),len(log_likelihood_test))
        
        log_likelihood_avg.append(log_likelihood)
        log_likelihood_test_avg.append(log_likelihood_test)
      #tempDf = pd.DataFrame(columns=['Modes','log_likelihood','interaction_factor','log_likelihood_test'])
      tempDf = pd.DataFrame()
      tempDf['Modes'] = modes
      tempDf['log_likelihood'] = log_likelihood
      tempDf['interaction_factor'] = interactionFactor  
      tempDf['log_likelihood_test'] = log_likelihood_test
    abc = pd.concat([abc,tempDf])
        
    
        #print(interactionFactor,'\n',log_likelihood,'\n',log_likelihood_test)

  print(abc)
  print(abc.shape)
  
    #plot components vs log likelihood for maximum of all interaction factors
  max_value = pd.DataFrame(columns=['Modes','log_likelihood','interaction_factor','log_likelihood_test'])
  for i in range(0,20):
    df_idx = abc[abc['interaction_factor']==i]
    #print(df_idx.loc[df_idx['log_likelihood'].idxmax()])
    max_value = pd.concat([max_value,(df_idx.loc[df_idx['log_likelihood'].idxmax()])])
  #print(maximum_1_70_embasin)
  maximum_1_70_embasin = pd.DataFrame()
  mode_mean = []
  log_likelihood_mean = []
  log_likelihood_test_mean = []
  interaction_factor_mean = []
  for i in range(0,20):
    d = max_value[max_value['interaction_factor']==i]
    #print(d)
    #print(d['log_likelihood'].mean(axis=0))
    
    log_likelihood_mean.append(d['log_likelihood'].mean(axis=0))
    log_likelihood_test_mean.append(d['log_likelihood_test'].mean(axis=0))
    mode_mean.append(d['Modes'].mean(axis=0))
    interaction_factor_mean.append(d['interaction_factor'].mean(axis=0))
  #maximum_1_70_embasin = pd.concat([maximum_1_70_embasin,t])
  maximum_1_70_embasin['Modes'] = mode_mean
  maximum_1_70_embasin['log_likelihood'] = log_likelihood_mean
  maximum_1_70_embasin['interaction_factor'] = interaction_factor_mean  
  maximum_1_70_embasin['log_likelihood_test'] = log_likelihood_test_mean
  
  df_maximum_1_70_embasin = pd.DataFrame(maximum_1_70_embasin)
  print(df_maximum_1_70_embasin)
  plt.figure(figsize=(10,10))
  plt.ylabel('Loglikelihood', fontsize = 10) # Y label
  plt.xlabel('Interaction factor', fontsize = 10) # X label
  max_var_1 = df_maximum_1_70_embasin['interaction_factor'].max(),df_maximum_1_70_embasin['log_likelihood'].max()
  print(df_maximum_1_70_embasin['interaction_factor'].max(),df_maximum_1_70_embasin['log_likelihood'].max())

  maxi = df_maximum_1_70_embasin['log_likelihood'].max() #find maximum log_likelihood  
  xy=df_maximum_1_70_embasin.loc[df_maximum_1_70_embasin['log_likelihood']==maxi] #find the location of the maximum train_output to plot in the graph
  max_test = df_maximum_1_70_embasin['log_likelihood_test'].max() #find maximum log_likelihood of test
  xy_test=df_maximum_1_70_embasin.loc[df_maximum_1_70_embasin['log_likelihood_test']==max_test] #find the location of the maximum train_output to plot in the graph


  plt.annotate('Maximum'+str(max_var_1), xy=(df_maximum_1_70_embasin['interaction_factor'].max(),df_maximum_1_70_embasin['log_likelihood'].max()),xytext = (0,-5),arrowprops=        dict(facecolor='black',shrink=0.3))
  plt.plot(df_maximum_1_70_embasin['interaction_factor'],df_maximum_1_70_embasin['log_likelihood'],label="Loglikelihood train")
  plt.plot(df_maximum_1_70_embasin['interaction_factor'],df_maximum_1_70_embasin['log_likelihood_test'],label="Loglikelihood test")
  plt.legend()
  
  plt.scatter(xy['interaction_factor'],xy['log_likelihood'],color = '#eb3434',s=[200])
  plt.scatter(xy_test['interaction_factor'],xy_test['log_likelihood_test'],color = '#eb3434',s=[200])
  plt.savefig('figure_all/'+EMBasinsStr+'_'+str(nSeed) +'_'+'1.png')
  plt.figure(figsize=(10,10))
  plt.ylabel('Modes', fontsize = 10) # Y label
  plt.xlabel('Interaction factor', fontsize = 10) # X label
  plt.plot(df_maximum_1_70_embasin['interaction_factor'],df_maximum_1_70_embasin['Modes'])
  plt.savefig('figure_all/'+EMBasinsStr+'_'+ str(nSeed)+'_' +'2.png')
  #plt.show()
    
  #show all plots
  #d = df_m.loc[df_m['idxfactor']==1]
  maxi = []
  x_d = []
  y_d = []

  fig, axes = plt.subplots(4,5, sharex=False, sharey=False)
  fig.set_figheight(15)
  fig.set_figwidth(20)
  #plt.gcf().suptitle(r'Comparison between Loglikelihood of train and test', size=16)
  #axes.set_xlim([0,150])
  #axes.set_ylim([ymin,ymax])
  bottom = 1
  fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
  #fig.tight_layout()
  #fig.tight_layout()
  #print(enumerate(axes.flatten()))
  d = pd.DataFrame()
  for i, ax in enumerate(axes.flatten()):
    c = abc.loc[abc['interaction_factor']==i]
    m_l = []
    m_l_t = []
    m_m = []
    m_i = []
    
    for idx,nModes in enumerate(nModesList):
      t = c.loc[c['Modes']==nModes]
      m_l.append(t['log_likelihood'].mean(axis=0))
      m_l_t.append(t['log_likelihood_test'].mean(axis=0))
      m_m.append(t['Modes'].mean(axis=0))
      m_i.append(t['interaction_factor'].mean(axis=0))
    d['Modes'] = m_m
    d['log_likelihood'] = m_l
    d['interaction_factor'] = m_i  
    d['log_likelihood_test'] = m_l_t
    print(d)
      
      
      
    maxim = d['log_likelihood'].max() #find maximum train_output for the particular idxfactor
    xy=d.loc[d['log_likelihood']==maxim]
    max_test = d['log_likelihood_test'].max() #find maximum train_output for the particular idxfactor
    xy_test=d.loc[d['log_likelihood_test']==max_test]
    #print('xy',np.array(xy))
    #print(max)  
    #print(xy_test.Modes.item())
    ax.set_title('IdxFactor ='+str(i/10)+ ', nModes = '+ str(xy_test.Modes.item()),fontsize=10)#", Max = "+str(round(maxim, 5)),fontsize=10)
    ax.set_ylabel('Loglikelihood', fontsize = 8) # Y label
    ax.set_xlabel('Modes', fontsize = 8) # X label
    #ax.set(xlabel='Components', ylabel='training output')
    ax.tick_params(axis='both', which='both', labelsize=7)
    #for tick in ax.get_xticklabels():
    # tick.set_visible(True)
    ax.scatter(d['Modes'], d['log_likelihood_test'],color='tab:orange',marker=(5,2),s=50)#,s=50) #,color='tab:orange') #plt.cm.Paired(i/10.)
    ax.scatter(d['Modes'], d['log_likelihood'],color='tab:blue',marker=(5,2),s=50)#,color='tab:blue') #plt.cm.Paired(i/5.)
    s1 = ax.plot(d['Modes'], d['log_likelihood_test'],color='tab:orange',linewidth=0.5)#, label=r'Loglikelihood test') #plt.cm.Paired(i/10.)
    s2 = ax.plot(d['Modes'], d['log_likelihood'],color='tab:blue',linewidth=1)#, label=r'Loglikelihood train') #plt.cm.Paired(i/5.)
    ax.plot(xy_test['Modes'], xy_test['log_likelihood_test'],'ko')
    ax.plot(xy['Modes'], xy['log_likelihood'],'ro')
    ax.set_xlim([0,150])
    
    lowDim = MDS(n_components=2)
    df_mds = pd.DataFrame()
    df_mds['Modes'] = d['Modes']
    df_mds['log_likelihood_test'] =  d['log_likelihood_test']
    
    lowDimData = lowDim.fit_transform(df_mds)
    x,y = lowDimData.T
    x_d.append(x)
    y_d.append(y)
    print('x',x)
    print('y',y)
    
  
  
  #ax.legend([l,b],loc="upper left")
  #lgd = plt.legend((s1, s2), (r'$\lambda$D', r'$\lambda$D'), loc='upper center')
  
  #lines, labels = fig.axes[-1].get_legend_handles_labels()  
  #fig.legend(lines, labels, loc = 'upper left') 
  fig.savefig('figure_all/'+EMBasinsStr+'_'+ str(nSeed)+'_' +'3.png')
  
  
  plt.figure(figsize=(10,10))
  plt.plot(x_d[0],y_d[0])
  plt.show()

if __name__ == '__main__':
  saveBestNMode(nModesList, dataFileBase, EMBasinsStr, crossvalfold)
  test(nModesList, dataFileBase, EMBasinsStr, crossvalfold)   
