# Unsupervised Fitting of Models to Neural Data  
  
Fitting various models to neural data in an unsupervised manner, i.e. without using the stimuli that generated the neural responses.  
This project is an ongoing collaboration with Gašper Tkačik and Michael J. Berry II.  
  
Currently, we try to fit two models:  
1. Model allowing temporal (independent or Hidden Markov Model) and spatial (independendent or tree) correlations, from the paper:  
Prentice, Jason S., Olivier Marre, Mark L. Ioffe, Adrianna R. Loback, Gašper Tkačik, and Michael J. Berry II. 2016. “Error-Robust Modes of the Retinal Population Code.” PLOS Computational Biology 12 (11): e1005148. [https://doi.org/10.1371/journal.pcbi.1005148](https://doi.org/10.1371/journal.pcbi.1005148).  
We use python bindings from a companion repository: [https://github.com/adityagilra/TreeHMM-local](https://github.com/adityagilra/TreeHMM-local) which has been forked (and slightly modified) from the original C++ code with Matlab bindings [https://github.com/adriannaloback/TreeHMM-local](https://github.com/adriannaloback/TreeHMM-local).  
  
2. Winner take all neural clustering algorithm:  
Loback, Adrianna R., and Michael J. Berry. 2018. “A Biologically Plausible Mechanism to Learn Clusters of Neural Activity.” BioRxiv, August, 389155. [https://doi.org/10.1101/389155](https://doi.org/10.1101/389155).
We use the python bindings from a companion repository: [https://github.com/adityagilra/VIWTA-SNN](https://github.com/adityagilra/VIWTA-SNN) which has been forked (and slightly modified) from the original C++ code with Matlab bindings [https://github.com/adriannaloback/VIWTA-SNN](https://github.com/adriannaloback/VIWTA-SNN).  
  
On three kinds of data:  
1. Experimental retinal data from Prentice et al 2016   
[https://datadryad.org/stash/dataset/doi:10.5061/dryad.1f1rc](https://datadryad.org/stash/dataset/doi:10.5061/dryad.1f1rc).  
  
2. Experimental retinal data from  
Tkačik, Gašper, Olivier Marre, Dario Amodei, Elad Schneidman, William Bialek, and Michael J. Berry II. 2014. “Searching for Collective Behavior in a Large Network of Sensory Neurons.” PLOS Computational Biology 10 (1): e1003408. [https://doi.org/10.1371/journal.pcbi.1003408](https://doi.org/10.1371/journal.pcbi.1003408).  
We use the neural responses to 297 repeats of a 19s long movie. The full data is available here  
[https://research-explorer.app.ist.ac.at/record/5562](https://research-explorer.app.ist.ac.at/record/5562)  
  
3. We also use data generated by an algorithm that varies the correlation between neurons parametrically between 0 to 1.9 (1 corresponds to experiment), while keeping individual neural firing rates constant in the dataset 2 above.  
The code for the same is part of this repo in the folder [data_generation](https://github.com/adityagilra/UnsupervisedLearningNeuralData/tree/master/data_generation). See below for details.  
  
-------------  
  
## Fitting with TreeHMM model by Prentice et al 2016:  
Clone the github repo: [https://github.com/adityagilra/TreeHMM-local](https://github.com/adityagilra/TreeHMM-local) into a folder called `TreeHMM` (shoudn't have `-local`). Run `make` in this folder after setting the right python and boost versions and directories. Libraries boost and boostpython must be present on your system. Set the PYTHONPATH to include its parent.  
  
The file `EMBasins_sbatch.py` can be called directly on the command line with an argument that decides the dataset, the number of latent modes, and the random number generator seed. The file can also be called in batch mode on a cluster using the SLURM system, i.e. `sbatch --array=0-659 submit_EMBasins.sbatch` where the taskid is passed in as the command line argument. The taskid is decomposed by the script into the seed for dataset generation, the interactionFactorIdx which specifies the dataset, and the modeIdx which specifies the mode number in steps of 5 starting from 1.  
  
Ensure that you have downloaded and saved the datasets 1, 2 and 3 above, in folders specified by the DataFileBase variable in `EMBasins_sbatch.py`. Currently, the settings are:  
```python
if interactionFactorIdx < 20:
    #dataFileBaseName = 'Learnability_data/synthset_samps'
    #dataFileBase = dataFileBaseName + '_' + str(interactionFactorIdx+1)
    dataFileBase = 'Learnability_data/generated_data_1_'\
                                        +str(interactionFactorIdx+1)+'_'\
                                        +str(nSeed)
elif interactionFactorIdx == 20:
    dataFileBase = 'Learnability_data/IST-2017-61-v1+1_bint_fishmovie32_100'
elif interactionFactorIdx == 21:
    dataFileBase = 'Prenticeetal2016_data/unique_natural_movie/data'
```  
The taskid command line argument argument, from which interactionFactorIdx is derived, specifies which dataset is used. You don't need to have all the datasets present, only those passed in via the command line. Taskids 0-599 are for generated dataset 3, 600 to 629 are for dataset 2, 630-659 are for dataset 1.  
  
You can choose between Hidden Markov Model vs time-independent model by calling pyHMM() or pyEMBasins() respectively (no need to recompile). See details in `EMBasins_sbatch.py`.  
  
Spatial correlations / tree term can be removed by modifying this statement at the top of EMBasins.cpp in the TreeHMM-local repo (need to recompile i.e. `make` after this)  
 // Selects which basin model to use  
 typedef TreeBasin BasinType;  
 to  
 typedef IndependentBasin BasinType;  
Thus you can switch from HMM to EMBasins to remove time-domain correlations,  
 and TreeBasin to IndependentBasin to remove space-domain correlations.  
    
Another key boolean parameter is `shuffle` which shuffles time-bins in the dataset. This is usually set to `True` unless you are sure your dataset has clear temporal dependencies.  
  
The fitting script will save the fitted parameters with appropriate filename:  
```python
    if HMM:
        def saveFit(dataFileBase,nModes,params,trans,P,emiss_prob,alpha,pred_prob,hist,samples,stationary_prob,train_logli,test_logli):
            dataBase = shelve.open(dataFileBase + ('_shuffled' if shuffle else '') \
                                                + '_HMM'+(str(crossvalfold) if crossvalfold>1 else '') \
                                                + ('' if treeSpatial else '_notree') \
                                                +'_modes'+str(nModes)+'.shelve')
            ...
    else:
        def saveFit(dataFileBase,nModes,params,w,samples,state_list,state_hist,state_list_test,state_hist_test,P,P_test,prob,prob_test,train_logli,test_logli):
            dataBase = shelve.open(dataFileBase + ('_shuffled' if shuffle else '') \
                                                + '_EMBasins_full' \
                                                + ('' if treeSpatial else '_notree') \
                                                + '_modes'+str(nModes)+'.shelve')
            ...
```
based on the settings you chose in the same folder as the dataset, as a .shelve file. Be sure to set the same boolean settings as during fitting in the analyse and plot scripts as well, as it searches for the appropriate filename based on these settings.
  
-------------  
  
## Fitting with Winner Take All neural model by Loback and Berry 2014:  
Clone the github repository: [https://github.com/adityagilra/VIWTA-SNN](https://github.com/adityagilra/VIWTA-SNN) into a folder called `VIWTA_SNN` (note: hyphen to underscore). Run `make` in this folder after setting the right python and boost versions and directories. Libraries boost and boostpython must be present on your system. Set the PYTHONPATH to include its parent.  
  
The file `WTAcluster_sbatch.py` and `submit_WTAcluster.sbatch` work in the same way as `EMBasins_sbatch.py` and `submit_EMBasins.sbatch` above. `WTAcluster_sbatch.py` can be called independently on the commandline with an argument, or via the SLURM system in batch mode on a cluster. `WTAcluster_sbatch.py` depends on `EMBasins_sbatch.py` for its pre-processing, so look at both to understand usage.  
  
-------------  
  
## Code (provided by Gašper Tkačik) in the data_generation folder for generating synthetic datasets parametrized by alpha which controls spatial correlation strength.  
  
Shared folder Synthsets has the parameters to generate various synthetic data.  
These params are pre-fitted for various alphas, ensuring that individual neural mean firing rates match the experimental data from Tkačik, et al 2014.  
  
synthset_k_X_Y_Z.mat means:  
X == number of neurons (always 4, meaning 120 neuron groups)  
Y == the replicate (the subset of 120 neurons from data being studied; there is a lot of overlap of neurons within the group)  
Z == integer representing alpha factor, alpha = synthset.factor, scales all correlations up and down.  
  
In each synthset, there are parameters of the model fit (K-pairwise), which are `[synthset.hs; synthset.js]` in matlab.  
  
### Python bindings:  
Run `make` in the data_generation directory after setting the right python and boost versions and directories. You'll need the boost and boostpython libraries installed. Then see `runMMCGen.py` on how to read in the synthsets and to generate sythetic data.  
  
Generated datasets are stored in `../Learnability_data/generated_data_`... which is where EMBasins_sbatch.py will look for. You need to have this folder `Learnability_data` at the same level as this repo (one folder level higher than `data_generation` folder).
  
### Matlab bindings:  
The sampling is called by Matlab to C source, which needs to be complied using matlab mex compiler, using  
`mex mxMaxentTGen.cpp mt19937ar.cpp`  
This will produce “mxMaxentTGen” which can be executed from matlab.  
  
This routine is called from runMMCGen.m script, which you would execute as follows:  
`[i1 i2 stats ee smp_mc] = runMMCGen([synthset.hs;synthset.js], 120, 100000, 100, round(rand()*10000000),'KSpikeIsing', 0);`  
samples would be in smp_mc matrix (binary)  
This is for 120 neurons, draw 100000 samples by recording a sample, doing 100 MC steps, recording a sample etc (so “100” is the sampling frequency). The round(rand()…) stuff is the initial random seed. ‘KSpikeIsing’ is the form of the model, 0 doesn’t matter.  
  
As a basic check of correctness, one should be able to plot the true mean firing rates vs the mean firing rates sampled from the model (since the pre-fitted synthsets pin the firing rate of each of 120 neurons to its observed value):  
`figure;plot(synthset.mv0, mean(smp_mc'),'ko’)`  
This should be close to a straight line.  
  
-------------
  
## Post-fitting analyses and plots

After fitting a dataset with a model as above, the scripts `EMBasins_sbatch.py` or `WTAcluster_sbatch.py` store the fitted parameters and other data in .shelve files in the same `dataFileBase` directory as the dataset. After fitting, run `EMBasins_sbatch_plot.py` for analysis and plotting.
  
You need to set the same boolean switches in `EMBasins_sbatch_plot.py` as you had in the model fitting files, e.g.  
```python
HMM = False
shuffled = True
treeSpatial = True
```  
By default, the script will cycle through all 20 generated datasets, and the 2 experimental Marre et al 2017 and Prentice et al 2016 datasets (`interactionidx`=0-20,21,22). Choose a smaller range to limit the datasets to analyse.  
`for interactionFactorIdx in range(interactionsLen):`  

The first time you run this script on any freshly fitted datasets, be sure to set:
```python
findBestNModes = True       # loop over all the nModes data
                            #  & find best nModes for each dataset
                            # must be done at least once before plotting
                            #  to generate _summary.shelve

assignModesToData = True    # read in modes of spike patterns from fit
                            #  assign and save modes to each timebin in dataset
                            #  (need to do only once after fitting)
```
The plotting script loads the fitted data for all 30 number of modes (1 to 146 in steps of 5). It then selects the number of modes that gives the best log likelihood of fitting the dataset, which is then called the cross-validated number of modes. The script will save summary data for this mode in a ...summary.shelve file. It'll also assign and save the best-fit modes weights as determined by the model params to each time bin / neural pattern in the corresponding dataset.  
  
Thus, next time you run this script, you can set these two boolean switches to `False` and it'll just read the ...summary.shelve file.  
  
The are other boolean flags, `doMDS, doLDA, cfWTAresults, doMDSWTA` inform what to further analyze and plot. See the comments next to their definition in `EMBasins_sbatch_plot.py`.

Apart from the main analysis and plot script `EMBasins_sbatch_plot.py`, I've also written other analysis and plot files:  
  
`EMBasins_sbatch_plot_seeds.py`  
This does the same as above, but across multiple super-datasets generated with different seeds.  
  
`EMBasins_vs_HMM_cluster_compare.py`  
Compare clustering between EMBasins and HMM using Adjusted Random Index/Score.  
  
`EMBasins_vs_HMM_vs_WTA_cluster_MDS.py`  
Compare clustering between EMBasins, HMM and WTA using MDS or PCA.  
  