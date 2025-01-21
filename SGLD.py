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
def hist_laxis(data, n_bins=100, range_limits=None,bins=None,density=False):
    ##Not tranposed, gives histogram over each path
    data=np.array(data)
    # Setup bins and determine the bin location for each element for the bins
    R = np.min(data),np.max(data) if range_limits is None else range_limits
    N = data.shape[-1] #histogram along last dimension
    bins = np.linspace(R[0],R[1],n_bins+1) if bins is None else np.array(bins)
    n_bins=len(bins)-1
    data2D = data.reshape(-1,N)
    idx = np.searchsorted(bins, data2D,'right')-1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx==-1) | (idx==n_bins)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to 
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = n_bins*data2D.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    bw=(bins[1:]-bins[:-1])[None,...]
    return  counts/np.sum(counts,axis=-1,keepdims=True)/bw if density else counts

def covcalc(s):
    c=(s**2).mean(dim=0)
    musq=truemean**2
    cov_SGLD=(c-musq[None,...]).mean(dim=0) #shape (n_paths,n_features)
    err=torch.linalg.norm(cov_SGLD-truecov)/torch.linalg.norm(truecov)
    c1=(s**2).mean(dim=1)
    cov_SGLD=(c1-musq[None,...])[-25*K:] #shape (n_iters,n_paths,n_features)
    e1=torch.linalg.norm(((cov_SGLD-truecov)/truecov).flatten(start_dim=1,end_dim=-1),dim=-1)
    return err,e1

device='cpu'
n=bs=20 #constant batch size to ensure CLT holds roughly for a batch
K = 8 # number of batches
N=n*K
etarange = (2.**torch.arange(-2,4,1))/K
x = torch.randn(N,dtype=torch.float64,device=device)
n_paths=1000
epochs=1000+np.int32(20/(etarange*K)**3)
nbins=1000

truemean = x.mean().cpu()
truecov=torch.ones_like(truemean)/N
zrange = torch.linspace(normal.ppf(0.001), normal.ppf(0.999), nbins)
truedist = gaussian(zrange, loc=truemean, scale=1/np.sqrt(N))[None,...]
width=abs((zrange[0]-zrange[1])/2)
bins=zrange-width
addon=(zrange[-1]+width).reshape((1,))
bins=torch.cat((bins,addon))

class MyBatcher:
    def __init__(self,data,bs,n_paths,strat='1SS'):
        self.data=data
        self.length = len(data)
        shape=tuple([n_paths]+[1 for i in data.shape])
        self.datasource = data[None,...].repeat(shape)
        self.bs=bs
        self.K=self.length//self.bs 
        self.index=0
        self.n_paths=n_paths
        self.set_strat(strat)
    
    def set_strat(self,strat):
        if strat=='RR':
            print('RR selected')
            self.sample=self.RRsampler
        else:
            print('1SS selected')
            self.sample=self.OneSSsampler
            
    def RRsampler(self):
        if self.index==0:
            self.datasource=self.data[torch.argsort(torch.rand(size=(self.n_paths,self.length)), dim=-1)]
        k,bs=self.index,self.bs
        self.index=(self.index+1)%self.K 
        data=self.datasource[:,k*bs:(k+1)*bs]
        return data

    def OneSSsampler(self):
        if self.index==0:
            self.datasource=self.data[torch.argsort(torch.rand(size=(self.n_paths,self.length)), dim=-1)]
        k_=np.random.randint(low=0,high=self.length)
        self.index=(self.index+1)%self.K 
        inds=np.arange(k_,k_+self.bs)%self.length
        data=self.datasource[:,inds]
        return data

mybatcher=MyBatcher(x,n,n_paths)
bs=mybatcher.bs

def exact(z, expA, noisefactor, data):
    z_ = z
    term = data.mean(axis=-1,keepdims=True)
    z_ -= term  # convert to zbar
    z_ = expA*z_
    z_ += term
    zeta=torch.randn_like(z)
    noise = noisefactor*zeta
    z_ += noise
    return z_

def EM(z,h,noisefactor,data):
    term = data.mean(axis=-1,keepdims=True)
    z_ = (1.-h)*z
    z_ += h*term
    zeta=torch.randn_like(z)
    noise = noisefactor*zeta
    z_ += noise
    return z_

def LM(z,h,noisefactor,data,Rn):
    term = data.mean(axis=-1,keepdims=True)
    z_ = (1.-h)*z
    z_ += h*term
    zeta=torch.randn_like(z)
    noise = noisefactor*(zeta+Rn)/2
    z_ += noise
    return z_,zeta

def EMFullGrad(z,h,noisefactor):
    term = truemean
    z_ = (1.-h)*z
    z_ += h*term
    zeta=torch.randn_like(z)
    noise = noisefactor*zeta
    z_ += noise
    return z_

def LMFullGrad(z,h,noisefactor,Rn):
    term = truemean
    z_ = (1.-h)*z
    z_ += h*term
    zeta=torch.randn_like(z)
    noise = noisefactor*(zeta+Rn)/2
    z_ += noise
    return z_,zeta

def Wasserstein1(u, v):
    '''
    Return W1 distance between two densities u,v
    '''
    return np.sum(np.abs(np.cumsum(u,axis=-1)/np.sum(u,axis=-1,keepdims=True)-np.cumsum(
        v,axis=-1)/np.sum(v,axis=-1,keepdims=True)),axis=-1)/np.sum(u,axis=-1)


z=torch.zeros(size=(n_paths,1))
zlm=z.to(dtype=x.dtype).to(device=x.device).detach()
zem=zlm.clone().detach()


strats=['RR','1SS','FG']
sgld_dict={'EM':{s:{} for s in strats},'LM':{s:{} for s in strats}}
sgld_dict['K']=K
sgld_dict['etarange']=etarange


for eta_idx,eta in enumerate(etarange):
    h = torch.tensor([eta],device=x.device)
    expA_=torch.exp(-h)
    noisefactor = torch.sqrt((1-torch.exp(-2*h))/N)
    for strat in strats:
        mybatcher.set_strat(strat)
        Nsamples=epochs[eta_idx]*K
        i=0
        sampleslm=torch.zeros((Nsamples,*z.shape))
        samplesem=torch.zeros((Nsamples,*z.shape))

        R=torch.randn_like(zlm)
        for epoch in range(epochs[eta_idx]):
            for k in range(K): 
                #z = exact(z, expA_, noisefactor, data)
                if strat=='FG':
                    zem = EMFullGrad(zem, h, np.sqrt(2*h/N))
                    zlm,R = LMFullGrad(zlm, h, np.sqrt(2*h/N),R)
                    samplesem[i]=zem
                    sampleslm[i]=zlm
                else:
                    data=mybatcher.sample().to(x.device)
                    zlm,R = LM(zlm, h, np.sqrt(2*h/N),data,R)
                    zem = EM(zem, h, np.sqrt(2*h/N),data)
                    sampleslm[i]=zlm
                    samplesem[i]=zem

                i+=1
        sgld_dict['EM'][strat][str(eta.item())]=samplesem
        sgld_dict['LM'][strat][str(eta.item())]=sampleslm

##Plotting
err={'EM':{s:[] for s in strats},'LM':{s:[] for s in strats}}
K=sgld_dict['K']
for disc in err.keys():
    for strat in strats:
        loc=sgld_dict[disc][strat]
        es=[]
        for i,timestep in enumerate(loc.keys()):
            s=loc[timestep]
            e,e1=covcalc(s)
            es+=[e]
            h=float(timestep)
            # Plot oscillations
            if i==(len(loc.keys())-2):
                plt.figure()
                num=round(np.log2(h),1)
                stratlab='RM' if strat=='1SS' else strat
                plt.title(disc+'-'+stratlab+': Asymptotic Error, $h=2^{'+f'{num}'+'}$')
                
                plt.semilogy(np.arange(len(e1))/K,
                                      torch.abs(e1),'k',ls='-' ,base=2)
                plt.xlabel('Iteration over dataset')
                plt.ylabel('$\\frac{\|\Delta\Sigma\|}{\|\Sigma\|}$')
                # plt.savefig(os.path.join(figdir,f'ExactK{K}_Oscillations{strat}.pdf'),format='pdf',bbox_inches='tight')
        # print(es)
        err[disc][strat]=es
#FG
markerlist=['s','X','o']
plt.figure()
plt.loglog(etarange,err['EM']['RR'],'r-',base=2,label='EM-RR')
plt.loglog(etarange,err['EM']['1SS'],'g-',base=2,label='EM-RM')
plt.loglog(etarange,err['EM']['FG'],'b-',base=2,label='EM-FG')
plt.loglog(etarange,err['LM']['RR'],'r--',base=2,label='LM-RR')
plt.loglog(etarange,err['LM']['1SS'],'g--',base=2,label='LM-RM')
plt.loglog(etarange,err['LM']['FG'],'b--',base=2,label='LM-FG')

plt.title(f'Model Problem, $K={K}$')
plt.xlabel('$h$')
plt.ylabel('Relative Variance Error')
plt.legend()
# plt.savefig(os.path.join(figdir,f'ExactK{K}.pdf'),format='pdf',bbox_inches='tight')



#%% RR
eDh=exp(-h)
system=(1-eDh)**2*V
eDhi=exp(-i*h)
eDhj=exp(-j*h)
system=eDhi*system*eDhj

##i neq j
expr12=(1-exp(-h*r))**2/(1-exp(-h))**2-(1-exp(-r*h*2))/(1-exp(-2*h))
system_ij=system.subs({exp(-h*i)*exp(-h*j):expr12})

##i=j
expr11=(1-exp(-2*h*r))/(1-exp(-2*h))
iisubs={exp(-h*i)*exp(-h*j):expr11}
system_ii=system.subs(iisubs)

system_ii=system_ii.subs({V:Vii})
system_ij=system_ij.subs({V:Vij}).replace(Vij,-Vii/(K-1))

#Case V_b^K sum over n epochs
epoch_sumij=system_ij.subs({r:K})
epoch_sumii=system_ii.subs({r:K})

eDhj=exp(-j*K*h)
epoch_sumij=eDhj**2*epoch_sumij
epoch_sumii=eDhj**2*epoch_sumii

# epochsubs={exp(2*K*h*j*lambda_p):(1-exp(2*K*n*h*lambda_p))/(1-exp(2*K*h*lambda_p)),
#                           exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1-exp(n*K*h*(lambda_m+lambda_p)))/(1-exp(K*h*(lambda_m+lambda_p))),
#                           exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1-exp(n*K*h*(lambda_m+lambda_p)))/(1-exp(K*h*(lambda_m+lambda_p))),
#                           exp(2*K*h*j*lambda_m):(1-exp(2*n*K*h*lambda_m))/(1-exp(2*K*h*lambda_m))}
epochsubs={exp(-2*K*h*j):(1)/(1-exp(-2*K*h))}

exprij=epoch_sumij.subs(epochsubs)
exprii=epoch_sumii.subs(epochsubs)
eDh=exp(-r*h)
es_ij=eDh**2*exprij
es_ii=eDh**2*exprii

rem_ij=system_ij
rem_ii=system_ii

partii=(es_ii+rem_ii).expand().simplify().factor()
expr=(partii+es_ij+rem_ij).replace(Vii,K)

series_r=(sympy.series(expr,h,n=3)).coeff(h**2).simplify()
##Average over a period, accounting for correlations##
period_subs={A:(1-exp(-h*K))/(1-exp(-h))/K,
             B:(1-exp(-2*h*K))/(1-exp(-2*h))/K}

period_expr=expr.factor().expand().collect(exp(h*r))
period_expr=period_expr.replace(exp(-h*r),A).replace(exp(-2*h*r),B).subs(period_subs)
period_expr=period_expr.simplify()

Kval=16
nperiods=10
plt.figure(figsize=(4,3))
rrange=np.arange(0,nperiods*Kval,dtype=int)
hrange=2.**np.arange(-5,3,2)/Kval
varfunc=sympy.lambdify(args=(h,r), expr=expr.replace(K,Kval))
ans=varfunc(hrange[...,None],rrange[None,...]%Kval)
periodfunc=sympy.lambdify(args=(h), expr=period_expr.replace(K,Kval))
perioderrs=periodfunc(hrange)
errs=[]
ls=2*['-','--',':','-.']
for idx,row in enumerate(ans):
    # plt.semilogy(rrange/Kval,row,base=2)
    s=perioderrs[idx]
    line,=plt.semilogy(rrange/Kval,s*np.ones_like(rrange),color='k',base=2,ls=ls[idx],label='$h=2^{'+f'{round(np.log2(hrange[idx]))}'+'}$',alpha=1)
    plt.semilogy(rrange/Kval,row,base=2,ls=line.get_ls(),alpha=.5,color='k')

# plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error')
plt.xlabel('Iteration over dataset')
plt.title(f'ED-RR variance error $K={Kval}$')
plt.legend()
# plt.savefig(os.path.join(figdir,'EDRR1.pdf'),format='pdf',bbox_inches='tight')

plt.figure(figsize=(4,3))
plt.loglog(hrange,perioderrs,'kX-',base=2,ms=10,label=f'Average error over a period')
hKrange=hrange*Kval
plt.loglog(hrange,(hKrange**2)/6,'b--', label='Ref. line $(hK)^2/6$',base=2)
plt.ylabel('Relative variance error')
plt.xlabel('$h$')
plt.title(f'ED-RR variance error, $K={Kval}$')
plt.legend()
# plt.savefig(os.path.join(figdir,'EDRR2.pdf'),format='pdf',bbox_inches='tight')

##Average over a period, accounting for correlations##
period_series=(sympy.series(period_expr,h,n=3)).simplify()
#h^2*K(K+1)/6

plt.figure()
for Kval in 2**np.arange(1,10,2):
    varfunc=sympy.lambdify(h, expr=period_expr.replace(K,Kval))
    plt.loglog(hrange,varfunc(hrange/Kval),base=2,label=f'$K={Kval}$')
    # plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error')
plt.xlabel('$h$')
plt.title(f'RR variance error')
plt.legend()

#%% Shuffle Once
eDh=exp(-h)
system=(1-eDh)*(1-eDh)*V
eDhi=exp(-i*h)
eDhj=exp(-j*h)
system=eDhi*system*eDhj
r1,r2=sympy.symbols("r1:3", positive=True)

##i neq j r1<=r2
expr11=(1-exp(-h*r2))*(1-exp(-h*r1))/((1-exp(-h))**2)-(1-exp(-2*r1*h))/(1-exp(-2*h))

Vbrr_ij=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:r,r2:r})})) 
VbKK_ij=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:K,r2:K})}))
VbrK_ij=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:r,r2:K})}))
VbKr_ij=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:r,r2:K})}))


##i=j r1<r2
expr11=(1-exp(-2*h*r1))/(1-exp(-2*h))

Vbrr_ii=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:r})})) 
VbKK_ii=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:K})}))
VbrK_ii=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:r})}))
VbKr_ii=(system.subs({exp(-h*i)*exp(-h*j):expr11.subs({r1:r})}))


[Vbrr_ii,VbKK_ii,VbKr_ii,VbrK_ii]= [expr.subs({V:Vii}) for expr in [Vbrr_ii,VbKK_ii,VbKr_ii,VbrK_ii]]
[Vbrr_ij,VbKK_ij,VbKr_ij,VbrK_ij]= [expr.subs({V:-Vii/(K-1)}) for expr in [Vbrr_ij,VbKK_ij,VbKr_ij,VbrK_ij]]

#Vb_KK sum over (infinite) epochs
eDhi=exp(-i*K*h)
eDhj=exp(-j*K*h)
es_KKij=eDhi*VbKK_ij*eDhj
es_KKii=eDhi*VbKK_ii*eDhj

expr11=(1-exp(-K*h))**(-2)

expr_subs={exp(-K*h*i)*exp(-K*h*j):expr11}
es_KKii=es_KKii.subs(expr_subs)
es_KKij=es_KKij.subs(expr_subs)

eDh=exp(-r*h)
es_KKii=sympy.simplify(eDh*es_KKii*eDh) 
es_KKij=sympy.simplify(eDh*es_KKij*eDh) 

#Vb_rr
remainder_ij=Vbrr_ij
remainder_ii=Vbrr_ii

#Vb_Kr and Vb_rK sum
eDhi=exp(-i*K*h)
es_Krij=eDhi*VbKr_ij
es_Krii=eDhi*VbKr_ii
es_rKij=VbKr_ij*eDhi
es_rKii=VbKr_ii*eDhi

ex_p=1/(1-exp(-K*h))

expr_subs={exp(-K*h*i):ex_p}
es_Krii=es_Krii.subs(expr_subs)
es_Krij=es_Krij.subs(expr_subs)
es_rKii=es_rKii.subs(expr_subs)
es_rKij=es_rKij.subs(expr_subs)

eDh=exp(-r*h)
es_Krii=sympy.simplify(eDh*es_Krii) 
es_Krij=sympy.simplify(eDh*es_Krij) 
es_rKii=sympy.simplify(es_rKii*eDh) 
es_rKij=sympy.simplify(es_rKij*eDh) 

#various components
rem_ii=(remainder_ii) 
rem_ij=(remainder_ij) 

partii=(es_KKii+es_Krii+es_rKii+rem_ii)
expr=(partii+es_KKij+es_Krij+es_rKij+rem_ij).replace(Vii,K)

Kval=64
nperiods=10
plt.figure(figsize=(4,3))
rrange=np.arange(0,nperiods*Kval,dtype=int)
hrange=2.**np.arange(-8,6,2)
varfunc=sympy.lambdify(args=(h,r), expr=expr.replace(K,Kval))
ans=varfunc(hrange[...,None]/Kval,rrange[None,...]%Kval)
errs=[]
for idx,row in enumerate(ans):
    # plt.semilogy(rrange/Kval,row,base=2)
    s=np.cumsum(row)/(rrange+1)
    errs+=[s[-1]]
    line,=plt.semilogy(rrange/Kval,s,base=2,label='$hK=2^{'+f'{round(np.log2(hrange[idx]))}'+'}$',ls='-',alpha=1)
    plt.semilogy(rrange/Kval,row,base=2,ls='-',color=line.get_c(),alpha=.5)

plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('Iteration over dataset')
plt.title(f'SO variance error, $K={Kval}$')
plt.legend()

##Average over a period, accounting for correlations##
period_subs={A:(1-exp(-h*K))/(1-exp(-h))/K,
             B:(1-exp(-2*h*K))/(1-exp(-2*h))/K,
             C:(1-exp(-3*h*K))/(1-exp(-3*h))/K}

period_expr=expr.apart(exp(h*r)).expand().collect(exp(h*r))
period_expr=period_expr.replace(exp(-h*r),A).replace(exp(-2*h*r),B).replace(exp(-3*h*r),C).subs(period_subs)
period_expr=period_expr.factor()

plt.figure()
for Kval in 2**np.arange(1,10,2):
    varfunc=sympy.lambdify(h, expr=period_expr.replace(K,Kval))
    plt.loglog(hrange,varfunc(hrange/Kval),base=2,label=f'$K={Kval}$')
    # plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$hK$')
plt.title(f'SO variance error')
plt.legend()

#%% 1SS and 1SSwR
var_err=V*(1-exp(-h))**2/(1-exp(-2*h))
var_err.replace(V,K)
hrange=np.logspace(-2,6,base=2)

for val in [.01,.1,1,10]:
    varfunc=sympy.lambdify(h, expr=expr.subs({K:Kval}))
    plt.loglog(hrange,varfunc(hrange),label=f'$M={val}N$',base=2)
plt.loglog(hrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('variance error $V_q-1/N$ (units of $V_x$)')
plt.xlabel('$h$')
plt.title('1SSwR and 1SS variance error, critically damped $C=2\\sqrt{MN}$')
plt.legend()
#%% EM RR
system=h**2*V
eDhi=(1-h)**i
eDhj=(1-h)**j
system=eDhi*system*eDhj

##i neq j
expr12=(1-(1-h)**r)**2/(h)**2-(1-(1-h)**(2*r))/(1-(1-h)**2)
system_ij=system.subs({eDhi*eDhj:expr12})

##i=j
expr11=(1-(1-h)**(2*r))/(1-(1-h)**2)
iisubs={eDhi*eDhj:expr11}
system_ii=system.subs(iisubs) 

system_ii=system_ii.subs({V:Vii})
system_ij=system_ij.subs({V:Vij}).replace(Vij,-Vii/(K-1))
# system_ij=sympy.simplify(system_ij)

#Case V_b^K sum over n epochs
epoch_sumij=system_ij.subs({r:K})
epoch_sumii=system_ii.subs({r:K})

eDhj=(1-h)**(j*K)
epoch_sumij=eDhj**2*epoch_sumij
epoch_sumii=eDhj**2*epoch_sumii

# epochsubs={exp(2*K*h*j*lambda_p):(1-exp(2*K*n*h*lambda_p))/(1-exp(2*K*h*lambda_p)),
#                           exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1-exp(n*K*h*(lambda_m+lambda_p)))/(1-exp(K*h*(lambda_m+lambda_p))),
#                           exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1-exp(n*K*h*(lambda_m+lambda_p)))/(1-exp(K*h*(lambda_m+lambda_p))),
#                           exp(2*K*h*j*lambda_m):(1-exp(2*n*K*h*lambda_m))/(1-exp(2*K*h*lambda_m))}
epochsubs={eDhj**2:(1)/(1-(1-h)**(2*K))}

exprij=epoch_sumij.subs(epochsubs)
exprii=epoch_sumii.subs(epochsubs)
eDh=(1-h)**r
es_ij=eDh**2*exprij
es_ii=eDh**2*exprii

rem_ij=system_ij
rem_ii=system_ii

partii=(es_ii+rem_ii).expand().simplify().factor()
expr=(partii+es_ij+rem_ij).replace(Vii,K)

series_r=(sympy.series(expr,h,n=3)).coeff(h**2).simplify()

##Average over a period, accounting for correlations##
period_subs={A:(1-(1-h)**K)/h/K,
             B:(1-(1-h)**(2*K))/(1-(1-h)**2)/K}

period_expr=expr.replace((1-h)**r,A).replace((1-h)**(2*r),B).subs(period_subs)
period_expr=period_expr.factor()

# ##Average over a period, accounting for correlations##
# ##h coeff is 0
period_series=h**2*(sympy.series(period_expr,h,n=3)).coeff(h**2).simplify()
# #h^2*K(K+1)/4

# #add bias due to EM (normalised for relative variance error)
bias=sympy.series((1/(1-h/2)),h,n=3)-1
period_series+=h*bias.coeff(h)+h**2*bias.coeff(h**2)

expr+=h/(2-h)
locperiod_expr=period_expr+h/(2-h)
Kval=8
nperiods=10
plt.figure(figsize=(4,3))
rrange=np.arange(0,nperiods*Kval,dtype=int)
hrange=2.**np.arange(-5,3,.1)/Kval
varfunc=sympy.lambdify(args=(h,r), expr=expr.replace(K,Kval))
ans=varfunc(hrange[...,None],rrange[None,...]%Kval)
periodfunc=sympy.lambdify(args=(h), expr=locperiod_expr.replace(K,Kval))
perioderrs=periodfunc(hrange)
errs=[]
# ls=['-','--',':','-.']
# for idx,row in enumerate(ans):
#     # plt.semilogy(rrange/Kval,row,base=2)
#     s=perioderrs[idx]
#     line,=plt.semilogy(rrange/Kval,s*np.ones_like(rrange),color='k',base=2,ls=ls[idx],label='$h=2^{'+f'{round(np.log2(hrange[idx]))}'+'}$',alpha=1)
#     plt.semilogy(rrange/Kval,row,base=2,ls=line.get_ls(),alpha=.5,color='k')

# # plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
# plt.ylabel('Relative variance error')
# plt.xlabel('Iteration over dataset')
# plt.title(f'Theoretical SGLD-RR, $R={Kval}$')
# plt.legend()
# plt.savefig(os.path.join(figdir,'EMRR1.pdf'),format='pdf',bbox_inches='tight')

plt.figure(figsize=(4,3))
hKrange=hrange*Kval
rmerror=(hrange+hKrange)/(2-hrange)
plt.loglog(hrange,rmerror,'b-',base=2,ms=10,label=f'SGLD-RM')
plt.loglog(hrange,perioderrs,'r-',base=2,ms=10,label=f'SGLD-RR')
localfunc=sympy.lambdify(args=(h), expr=period_expr.replace(K,Kval))
plt.loglog(hrange,hrange/(2-hrange),'k-', label='ULA',base=2)
plt.loglog(hrange,localfunc(hrange),'r--', label='RR SG bias',base=2)
plt.loglog(hrange,hKrange/(2-hrange),'b--', label='RM SG bias',base=2)

plt.ylabel('Relative variance error')
plt.xlabel('$h$')
plt.title(f'Theoretical Variance Error, $R={Kval}$')
plt.legend()
plt.savefig(os.path.join(figdir,'EMRR2.pdf'),format='pdf',bbox_inches='tight')


#%%For EM, with RR, 1SS or Exact Gradient
plt.loglog(h*K,(K*h)**2/6+h/2,'kX-',base=2,label='RR',ms=8) #RR
plt.loglog(h*K,K*h/2+h/2,'ks-',base=2,label='1SS',ms=8) #1SS
plt.xlabel('$hK$')
plt.ylabel('Relative Variance Error')
plt.loglog(h*K,h/2,'kd-',base=2,label='Full Grad',ms=8) #Exact Gradient
plt.title('EM SGLD sampling for 1D Gaussian')
plt.legend()