# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:50:52 2024

@author: lshaw
"""
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm as normal
from sklearn.utils import shuffle as myshuffler
import sympy as sympy
from sympy import sin,cos,sqrt,pi, S,tan, I,atan,exp,tanh
A,B,i,j=sympy.symbols("A,B,i,j", real=True)
w,lambda_p,lambda_m=sympy.symbols("omega,lambda_p,lambda_m", real=False)
N,V,K,h,M,C,sigma,r,Vii,Vij,n,T=sympy.symbols("N,V,K,h,M,C,sigma,r,V_ii,V_ij,n,T", positive=True)
sympy.init_printing(forecolor="White",use_latex=True,fontsize='12pt')

plt.rc('text', usetex=True)
plt.rc('font', **{'serif': ['cm'],'size':20})
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
gaussian = normal.pdf
figdir='figs'
#%%
device='cpu'
n=bs=32 #constant batch size to ensure CLT holds roughly for a batch
K = 32 # number of batches
N=n*K
etarange = 2.**np.arange(-2,6,1)/K
etarange=np.flip(etarange)
x = torch.randn(N,dtype=torch.float64,device=device)
n_paths=1000
epochs=50+np.int32(20/(etarange*K)**3)
nbins=1000
truemean = x.mean().cpu()

class MyBatcher:
    def __init__(self,data,bs,n_paths):
        self.n_paths=n_paths
        self.data=data
        self.datasource = self.data.repeat((n_paths,1))
        self.length = len(data)
        self.bs=bs
        self.K=self.length //self.bs 

    def RRsampler(self):
        self.datasource=self.data[torch.argsort(torch.rand((self.n_paths,self.length)), dim=-1)]
        return None

mybatcher=MyBatcher(x,n,n_paths)
bs=mybatcher.bs

def EM(z,h,noisefactor,data):
    term = data.mean(axis=-1,keepdims=True)
    z_ = (1.-h)*z
    z_ += h*term
    zeta=torch.randn_like(z)
    noise = noisefactor*zeta
    z_ += noise
    return z_

mean_err = []
var_err=[]
var_errbar=[]
z=torch.zeros(size=(n_paths,1))
z=z.to(dtype=x.dtype).to(device=x.device)
fig2=plt.figure()
fig3=plt.figure()

for eta_idx,eta in enumerate(etarange):
    h = torch.tensor([eta],device=x.device)
    verr_prog = torch.zeros((epochs[eta_idx]*K)+1,device=x.device)
    var_time=torch.zeros((epochs[eta_idx]*K)+1,device=x.device)
    idx = 0
    mybatcher.RRsampler()
    if eta_idx==0: #do a burnin
        for epoch in range(25):
            mybatcher.RRsampler()
            for k in range(K): 
                k_=np.random.randint(mybatcher.length)
                inds=np.arange(k_,k_+bs)%mybatcher.length
                data=mybatcher.datasource[:,inds]
                # data=(mybatcher.datasource[:,k*bs:(k+1)*bs]).to(x.device)
                z = EM(z, h, np.sqrt(2*h/N),data)

    for epoch in range(epochs[eta_idx]):
        mybatcher.RRsampler()
        for k in range(K): 
            k_=np.random.randint(mybatcher.length)
            inds=np.arange(k_,k_+bs)%mybatcher.length
            data=mybatcher.datasource[:,inds]
            # data=(mybatcher.datasource[:,k*bs:(k+1)*bs]).to(x.device)
            z = EM(z, h, np.sqrt(2*h/N),data)

            idx+=1
            var_time[idx]=torch.mean(N*(z**2-truemean**2)-1)
            verr_prog[idx]=torch.mean((idx-1)*verr_prog[idx-1]+(N*(z**2-truemean**2)-1))/idx

    with open(f'figs/EMvartime_1SS_eta{eta}_K{K}.npz','wb') as f:
        np.save(f,var_time.cpu())
    with open(f'figs/EMverrprog_1SS_eta{eta}_K{K}.npz','wb') as f:
        np.save(f,verr_prog.cpu())
        
    num=round(np.log2(h.item()*K),1)
    var_err += [(verr_prog[-1])] #relative error in variance


hrange=etarange*K
var_err=np.array(var_err)
var_errbar=np.array(var_errbar)

var_errRR=[]
var_err1SS=[]
fig2=plt.figure(figsize=(4,3))
ls=2*['-','--',':','-.']
for idx,eta in enumerate(etarange):
    num=round(np.log2(eta*K),1)
    with open(f'figs/EMverrprog_RR_eta{eta}_K{K}.npz','rb') as f:
        verr_prog=np.load(f)
    with open(f'figs/EMvartime_RR_eta{eta}_K{K}.npz','rb') as f:
        var_time=torch.tensor(np.load(f))
    plt.figure(fig2)
    plt.semilogy(np.arange(len(var_time[-10*K:]))/K,
                  torch.abs(var_time[-10*K:]),'k',ls=ls[idx] ,
                  label='$hK=2^{'+f'{num}'+'}$',base=2)
    fig2title=f'Relative Variance Error, Asymptotic, RR $K={K}$'
    var_errRR += [verr_prog[-1]] #relative error in variance
    with open(f'figs/EMverrprog_1SS_eta{eta}_K{K}.npz','rb') as f:
        verr_prog=np.load(f)
    with open(f'figs/EMvartime_1SS_eta{eta}_K{K}.npz','rb') as f:
        var_time=torch.tensor(np.load(f))
    plt.figure(fig3)
    plt.semilogy(np.arange(len(var_time[-10*K:]))/K,
                  torch.abs(var_time[-10*K:]),'k',ls=ls[idx] ,
                  label='$hK=2^{'+f'{num}'+'}$',base=2)
    fig3title=f'Relative Variance Error, Asymptotic, 1SS $K={K}$'
    var_err1SS += [verr_prog[-1]] #relative error in variance
plt.figure(fig2)
plt.title(fig2title)
plt.xlabel('Iteration over dataset')
plt.legend()
plt.figure(fig3)
plt.title(fig3title)
plt.xlabel('Iteration over dataset')
plt.legend()

hrange=etarange*K
plt.figure(figsize=(4,3))
plt.loglog(hrange,var_errRR,'ks-',ms=8,label='RR')
plt.loglog(hrange,var_err1SS,'kX-',ms=8,label='1SS')

plt.title('EM SGLD$'+f'K={K}$, Relative Variance Error'+'; RR vs 1SS')
plt.xlabel('$hK$')
plt.loglog(hrange,np.max(var_err1SS)*(hrange/np.max(hrange)),'k--',label='ref line $(hK)^1$',base=2)
plt.loglog(hrange,np.max(var_errRR)*(hrange/np.max(hrange))**2,'k:',label='ref line $(hK)^2$',base=2,lw=3)

plt.legend()
