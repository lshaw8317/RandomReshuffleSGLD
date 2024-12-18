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

device='cpu'
n=bs=20 #constant batch size to ensure CLT holds roughly for a batch
K = 4 # number of batches
N=n*K
etarange = 2.**np.arange(-5,1,1)/K
etarange=np.flip(etarange)
x = torch.randn(N,dtype=torch.float64,device=device)
n_paths=10000
epochs=50+np.int32(20/(etarange*K)**3)
nbins=1000
# gg1=torch.tensor((np.cos(np.pi/6)+1j*np.sin(np.pi/6))/np.sqrt(3))
# gg1=torch.ones(1)
# gammas=torch.tensor([gg1,gg1.conj()],dtype=torch.complex128)

truemean = x.mean().cpu()
zrange = torch.linspace(normal.ppf(0.001), normal.ppf(0.999), nbins)
truedist = gaussian(zrange, loc=truemean, scale=1/np.sqrt(N))[None,...]
width=abs((zrange[0]-zrange[1])/2)
bins=zrange-width
addon=(zrange[-1]+width).reshape((1,))
bins=torch.cat((bins,addon))

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

def EMExact(z,h,noisefactor):
    term = truemean
    z_ = (1.-h)*z
    z_ += h*term
    zeta=torch.randn_like(z)
    noise = noisefactor*zeta
    z_ += noise
    return z_

def Wasserstein1(u, v):
    '''
    Return W1 distance between two densities u,v
    '''
    return np.sum(np.abs(np.cumsum(u,axis=-1)/np.sum(u,axis=-1,keepdims=True)-np.cumsum(
        v,axis=-1)/np.sum(v,axis=-1,keepdims=True)),axis=-1)/np.sum(u,axis=-1)

mean_err = []
var_err=[]
var_errbar=[]
W1=[]
W1vars=[]
z=torch.zeros(size=(n_paths,1))
z=z.to(dtype=x.dtype).to(device=x.device)
fig1=plt.figure()
fig2=plt.figure()
for eta_idx,eta in enumerate(etarange):
    h = torch.tensor([eta],device=x.device)
    expA_=torch.exp(-h)
    noisefactor = torch.sqrt((1-torch.exp(-2*h))/N)
    verr_prog = torch.zeros((epochs[eta_idx]*K)+1,device=x.device)
    var_time=torch.zeros((epochs[eta_idx]*K)+1,device=x.device)
    idx = 0
    mybatcher.RRsampler()
    if eta_idx==0: #do a burnin
        for epoch in range(25):
            mybatcher.RRsampler()
            for k in range(K): 
                # k_=np.random.randint(mybatcher.length)
                # inds=np.arange(k_,k_+bs)%mybatcher.length
                # data=mybatcher.datasource[:,inds]
                data=(mybatcher.datasource[:,k*bs:(k+1)*bs]).to(x.device)
                # z = exact(z, expA_, noisefactor, data)
                z = EM(z, h, np.sqrt(2*h/N),data)

    for epoch in range(epochs[eta_idx]):
        mybatcher.RRsampler()
        for k in range(K): 
            # k_=np.random.randint(mybatcher.length)
            # inds=np.arange(k_,k_+bs)%mybatcher.length
            # data=mybatcher.datasource[:,inds]
            data=(mybatcher.datasource[:,k*bs:(k+1)*bs]).to(x.device)
            # z = exact(z, expA_, noisefactor, data)
            z = EM(z, h, np.sqrt(2*h/N),data)

            idx+=1
            var_time[idx]=torch.mean(N*(z**2-truemean**2)-1)
            verr_prog[idx]=torch.mean((idx-1)*verr_prog[idx-1]+(N*(z**2-truemean**2)-1))/idx
            # counts_fixedpath += hist_laxis(z.T, bins=bins,density=False) #pathwise sample
            # counts_fixedtime += hist_laxis(z, bins=bins,density=False) #plot W1 as function of time
            
    with open(f'figs/EMvartime_RR_eta{eta}_K{K}.npz','wb') as f:
        np.save(f,var_time.cpu())
    with open(f'figs/EMverrprog_RR_eta{eta}_K{K}.npz','wb') as f:
        np.save(f,verr_prog.cpu())
        
    num=round(np.log2(h.item()*K),1)

    plt.figure(fig1)
    time=np.arange(1,idx+1)[...,None]
    # errorbar = torch.sqrt(verr_prog.var(axis=-1)/n_paths)#torch.sqrt(3/time)*(1+h*K)/torch.sqrt(1-torch.exp(-h))
    verr_prog_plot = torch.abs(verr_prog)[1:] #relative variance error
    plt.semilogy(time.squeeze()/K,
                 verr_prog_plot, 
                 label='$hK=2^{'+f'{num}'+'}$',base=2)
    
    # plt.fill_between(time.squeeze()/K, verr_prog_plot-errorbar, verr_prog_plot+errorbar,
    #              color='gray', alpha=0.2)
    

    plt.figure(fig2)
    plt.semilogy(np.arange(len(var_time[-50*K:]))/K,
                  torch.abs(var_time[-50*K:]), 
                  label='$hK=2^{'+f'{num}'+'}$',base=2)
    fig2title=f'Relative Variance Error, Asymptotic, $K={K}$'
    plt.xlim([0,50])

    # W1_path=Wasserstein1(counts_fixedpath,truedist) 
    var_err += [(verr_prog[-1])] #relative error in variance
    # var_errbar += [np.sqrt(verr_prog[-1].var())/np.sqrt(n_paths)] #relative error in variance
    # W1 += [W1_path.mean()]
    # W1vars+=[np.sqrt(W1_path.var())/np.sqrt(samples.shape[1])]

hrange=etarange*K
# W1=np.array(W1)
# W1vars=np.array(W1vars)
var_err=np.array(var_err)
var_errbar=np.array(var_errbar)

plt.figure(fig1)
plt.title(f'Relative Variance Error, $K={K}$')
plt.xlabel('Iteration over dataset')
plt.legend()
# plt.ylim([np.min(hrange)*K*x.var()/4,np.max(hrange)*K*x.var()])

plt.figure(fig2)
plt.legend()
plt.title(fig2title)
plt.xlabel('Iteration over dataset')
# plt.ylim([2.**-10,2**-3])

var_errRR=[]
fig2=plt.figure(figsize=(4,3))
ls=['-','--',':','-.']
for idx,eta in enumerate([.125,.0625,.03125,.015625]):
    num=round(np.log2(eta*K),1)
    with open(f'figs/EMverrprog_RR_eta{eta}_K{K}.npz','rb') as f:
        verr_prog=np.load(f)
    with open(f'figs/EMvartime_RR_eta{eta}_K{K}.npz','rb') as f:
        var_time=torch.tensor(np.load(f))
    plt.figure(fig2)
    if idx<2:
        plt.semilogy(np.arange(len(var_time[-25*K:]))/K,
                      torch.abs(var_time[-25*K:]),'k',ls=ls[idx] ,
                      label='$hK=2^{'+f'{num}'+'}$',base=2)
    fig2title=f'Relative Variance Error, Asymptotic, 1SS $K={K}$'
    var_errRR += [verr_prog[-1]] #relative error in variance
plt.figure(fig2)
plt.title(fig2title)
plt.xlabel('Iteration over dataset')
plt.legend()
# plt.ylim([2**-8,2**-2])
# plt.savefig('figs/EM_SGLDExp1SS.pdf',format='pdf')

hrange=etarange*K
plt.figure(figsize=(4,3))
# errorbar=var_errbar#np.sqrt(3/len(samples))*np.array(1.+hrange*K)/np.sqrt(1-np.exp(-hrange))
# plt.loglog(hrange,var_err1SS,'kX-',ms=8,label='1SS')
plt.loglog(hrange,var_errRR,'ks-',ms=8,label='RR')
# plt.loglog(hrange,var_errTrueGrad,'kd-',ms=8,label='EM')

# plt.fill_between(hrange, var_err-errorbar, var_err+errorbar,
#                   color='gray', alpha=0.2)
plt.title('EM SGLD$'+f'K=8$, Relative Variance Error'+'; RR vs 1SS')
plt.xlabel('$hK$')
plt.loglog(hrange,np.max(var_err1SS)*(hrange/np.max(hrange)),'k--',label='ref line $(hK)^1$',base=2)
plt.loglog(hrange,np.max(var_errRR)*(hrange/np.max(hrange))**2,'k:',label='ref line $(hK)^2$',base=2,lw=3)
# plt.loglog(hrange,np.max(var_errRR)*(hrange/np.max(hrange))**(3/2),'k.-',label='ref line $(hK)^{3/2}$',base=2,lw=3)

# plt.ylim([2**-4,2**0])
plt.legend()
# plt.savefig('figs/EM_SGLDExpBias.pdf',format='pdf')

# plt.figure()
# plt.loglog(hrange,mean_err)
# plt.title('$'+f'K={K}$, Mean Error')
# plt.xlabel('$h=K\\eta$')
# plt.loglog(hrange,np.max(mean_err)*hrange**2,'k--',label='ref line $h^1$',base=2)
# # plt.ylim([2**-11,2**-3])
# plt.legend()



# plt.figure()
# plt.loglog(hrange,W1,base=2)
# plt.loglog(hrange,np.max(W1)*hrange/np.max(hrange),'r--',label='ref line $h^1$',base=2)
# plt.loglog(hrange,np.max(W1)*(hrange/np.max(hrange))**2,'g--',label='ref line $h^2$',base=2)
# plt.fill_between(hrange, W1-W1vars, W1+W1vars,
#                  color='gray', alpha=0.2)
# plt.title('$'+f'K={K}$, W1 distance')
# plt.xlabel('$hK$')
# plt.legend()
# # plt.ylim([2**-9,2**-2])


#%% RR
eDh=exp(-h)
system=(1-eDh)**2*V
eDhi=exp(-i*h)
eDhj=exp(-j*h)
system=eDhi*system*eDhj

##i neq j
expr12=(1-exp(-h*r))**2/(1-exp(-h))**2-(1-exp(-r*h*2))/(1-exp(-2*h))
system_ij=sympy.simplify(system.subs({exp(-h*i)*exp(-h*j):expr12.factor()})) 

##i=j
expr11=(1-exp(-2*h*r))/(1-exp(-2*h))
iisubs={exp(-h*i)*exp(-h*j):expr11}
system_ii=sympy.simplify(system.subs(iisubs)) 

system_ii=system_ii.subs({V:Vii})
system_ij=system_ij.subs({V:Vij}).replace(Vij,-Vii/(K-1))
system_ij=sympy.simplify(system_ij)

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
es_ij=sympy.simplify(eDh**2*exprij) 
es_ii=sympy.simplify(eDh**2*exprii) 

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
period_expr=period_expr.factor()

Kval=16
nperiods=10
plt.figure(figsize=(4,3))
rrange=np.arange(0,nperiods*Kval,dtype=int)
hrange=2.**np.arange(-5,3,2)
varfunc=sympy.lambdify(args=(h,r), expr=expr.replace(K,Kval))
ans=varfunc(hrange[...,None]/Kval,rrange[None,...]%Kval)
periodfunc=sympy.lambdify(args=(h), expr=period_expr.replace(K,Kval))
perioderrs=periodfunc(hrange/Kval)
errs=[]
ls=['-','--',':','-.']
for idx,row in enumerate(ans):
    # plt.semilogy(rrange/Kval,row,base=2)
    s=perioderrs[idx]
    line,=plt.semilogy(rrange/Kval,s*np.ones_like(rrange),color='k',base=2,ls=ls[idx],label='$hK=2^{'+f'{round(np.log2(hrange[idx]))}'+'}$',alpha=1)
    plt.semilogy(rrange/Kval,row,base=2,ls=line.get_ls(),alpha=.5,color='k')

# plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('Iteration over dataset')
plt.title(f'SGLD RR variance error $K={Kval}$')
plt.legend()
plt.savefig(os.path.join(figdir,'SGLDRR1.pdf'),format='pdf',bbox_inches='tight')

plt.figure(figsize=(4,3))
plt.loglog(hrange,perioderrs,'kX-',base=2,ms=10,label=f'Average error over a period')
scaler=np.min(perioderrs)
plt.loglog(hrange,scaler*(hrange/np.min(hrange))**2,'k--', label='Ref. line $(hK)^2$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$hK$')
plt.title(f'SGLD RR variance error, $K={Kval}$')
plt.legend()
plt.savefig(os.path.join(figdir,'SGLDRR2.pdf'),format='pdf',bbox_inches='tight')

##Average over a period, accounting for correlations##
period_series=(sympy.series(period_expr,h,n=3)).simplify()
#h^2*K(K+1)/6

plt.figure()
for Kval in 2**np.arange(1,10,2):
    varfunc=sympy.lambdify(h, expr=period_expr.replace(K,Kval))
    plt.loglog(hrange,varfunc(hrange/Kval),base=2,label=f'$K={Kval}$')
    # plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$hK$')
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
system_ij=sympy.simplify(system.subs({eDhi*eDhj:expr12.factor()})) 

##i=j
expr11=(1-(1-h)**(2*r))/(1-(1-h)**2)
iisubs={eDhi*eDhj:expr11}
system_ii=sympy.simplify(system.subs(iisubs)) 

system_ii=system_ii.subs({V:Vii})
system_ij=system_ij.subs({V:Vij}).replace(Vij,-Vii/(K-1))
system_ij=sympy.simplify(system_ij)

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
es_ij=sympy.simplify(eDh**2*exprij) 
es_ii=sympy.simplify(eDh**2*exprii) 

rem_ij=system_ij
rem_ii=system_ii

partii=(es_ii+rem_ii).expand().simplify().factor()
expr=(partii+es_ij+rem_ij).replace(Vii,K)

series_r=(sympy.series(expr,h,n=3)).coeff(h**2).simplify()

##Average over a period, accounting for correlations##
period_subs={A:(1-(1-h)**K)/h/K,
             B:(1-(1-h)**(2*K))/(1-(1-h)**2)/K}

period_expr=expr.factor().expand().collect(exp(h*r))
period_expr=period_expr.replace((1-h)**r,A).replace((1-h)**(2*r),B).subs(period_subs)
period_expr=period_expr.factor()

##Average over a period, accounting for correlations##
##h coeff is 0
period_series=h**2*(sympy.series(period_expr,h,n=3)).coeff(h**2).simplify()
#h^2*K(K+1)/6

#add bias due to EM (normalised for relative variance error)
bias=sympy.series((1/(1-h/2)),h,n=3)-1
period_series+=h*bias.coeff(h)+h**2*bias.coeff(h**2)
#%%For EM, with RR, 1SS or Exact Gradient
plt.loglog(h*K,(K*h)**2/6+h/2,'kX-',base=2,label='RR',ms=8) #RR
plt.loglog(h*K,K*h/2+h/2,'ks-',base=2,label='1SS',ms=8) #1SS
plt.xlabel('$hK$')
plt.ylabel('Relative Variance Error')
plt.loglog(h*K,h/2,'kd-',base=2,label='Full Grad',ms=8) #Exact Gradient
plt.title('EM SGLD sampling for 1D Gaussian')
plt.legend()