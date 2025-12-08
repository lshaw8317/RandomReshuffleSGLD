import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from utils import Sampler, load_data, LogReg, get_HMC
plt.rcParams.update({'text.usetex':False,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.style.use('seaborn-v0_8-paper')
figdir='figs'
resultsdir='RESULTS'
datadir='data'

def plotter(expname,K):
    with open(f"LogReg{expname}_SGLDK{K}.pkl", 'rb') as f:
        sgld_dict=pickle.load(f)
    
    ##Plotting
    strats = [i for i in sgld_dict.keys() if i in ['RR','RM','FG']]
    err={s:[] for s in strats}
    K=sgld_dict['K']
    etarange=sgld_dict['etarange']
    try:
        with open(f"LogReg_{expname}K{K}_err.pkl", 'rb') as f:
            err=pickle.load(f)
    except:

        with open(f"LogReg_{expname}HMCtruemean.pkl", 'rb') as f:
            truemean=pickle.load(f)
            import torch
            if isinstance(truemean, torch.Tensor):
                truemean=truemean.detach().numpy()
        
        n=len(etarange)
        for strat in err.keys():
            loc=sgld_dict[strat]
            for i,timestep in enumerate(loc.keys()):
                samples=loc[timestep]
                s=samples
                e=(s.mean(axis=0)-truemean).mean(axis=0) #shape (n_paths,n_features)
                e1=(s.mean(axis=1)-truemean)[-10*K:]
                e1=np.linalg.norm(e1, axis=1) / np.linalg.norm(truemean)#shape (n_iters, n_features)
                h=np.float64(timestep)
                #Plot oscillations
                if i==2 and e.shape[0] >= 100:
                    plt.figure(figsize=(3,2))
                    num=np.round(h,1)
                    stratlab=strat #'RM' if strat=='1SS' else strat
                    plt.title(f'LogReg {expname}: ' + 'SGLD-'+stratlab+', $h='+f'{num}'+'$')
                    
                    plt.semilogy(np.arange(len(e1))/K, np.abs(e1),'k',ls='-' ,base=2)
                    plt.xlabel('Iteration over dataset')
                    plt.ylabel('$\|\Delta\mu\|/\|\mu\|$')
                    if strat=='1SS':
                        plt.yticks([2**-1.4,2**-1.25],['$2^{-1.4}$','$2^{-1.25}$'])
                    else:
                        plt.yticks([2**-1.5,2**-1.4],['$2^{-1.5}$','$2^{-1.4}$'])

                    plt.savefig(os.path.join(figdir,f'LogReg{expname}K{K}_Oscillations{strat}.png'),format='png',bbox_inches='tight')
                err[strat]+=[np.linalg.norm(e) / np.linalg.norm(truemean)]

    with open(f"LogReg_{expname}K{K}_err.pkl", 'wb') as f:
        pickle.dump(err,f)
    
    plt.figure(figsize=(3,2))
    for strat in err.keys():
        plt.loglog(etarange,err[strat],base=2,label=f'SGLD-{strat}')
    plt.loglog(etarange,etarange,'k--',base=2,alpha=0.7)
    plt.loglog(etarange,(etarange)**2,'k--',base=2,alpha=0.7)
    plt.ylim([2**-6.9,2**1])
    
    plt.title(f'LogReg {expname}, $R={K}$')
    plt.xlabel('$h$')
    plt.ylabel('$\|\Delta\mu\|/\|\mu\|$')
    plt.legend()
    plt.savefig(os.path.join(figdir,f'LogReg{expname}K{K}.png'),format='png',bbox_inches='tight')

def runLRExp(expname, K, n_paths=10**4):
    Nsamples_HMC = 10**6

    x, y = load_data(expname, datadir=datadir)
    try:
        with open(f"LogReg_{expname}HMCtruemean.pkl", 'rb') as f:
            truemean=pickle.load(f)
    except:
        loss = LogReg([x, y], 1, 1, expname=expname)
        get_HMC(loss, Nsamples_HMC)
    
    loss = LogReg([x, y], K, n_paths, expname=expname)
    etarange = 2.**np.arange(-3, 1)
    Nsamples=np.int64(np.minimum(10000+(1000/(etarange)**2), 10**4))

    strats=['RR','RM','FG']
    sgld_dict={s:{} for s in strats}
    sgld_dict['K'] = K
    sgld_dict['etarange']=etarange
    for strat in strats:
        sampler = Sampler(loss, method='sgld', strat=strat)
        for i,timestep in enumerate(etarange):
            samples = sampler.run(timestep, Nsamples[i])
            sgld_dict[strat][str(timestep.item())] = samples

    with open(f"LogReg{expname}_SGLDK{K}.pkl", 'wb') as f:
        pickle.dump(sgld_dict,f)

if __name__ == '__main__':
    for ename in ['CTG']:
        runLRExp(ename, 16, 20)
        plotter(ename, 16)