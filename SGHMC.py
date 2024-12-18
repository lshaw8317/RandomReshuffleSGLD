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
from sympy import sin,cos,sqrt,pi, S,tan, I,atan,exp
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

n=bs=20 #constant batch size to ensure CLT holds roughly for a batch
K = 8 # number of batches
N=n*K
etarange = 2.**np.arange(-1,1,.5)/K
etarange=np.flip(etarange)
x = torch.randn(N,dtype=torch.float64)
n_paths=10000
epochs=50+np.int32(N/(etarange*K)**2)
nbins=1000
shuffle = True
M=N
C=M
# gg1=torch.tensor((np.cos(np.pi/6)+1j*np.sin(np.pi/6))/np.sqrt(3))
# gg1=torch.ones(1)
# gammas=torch.tensor([gg1,gg1.conj()],dtype=torch.complex128)


dataloader = torch.utils.data.DataLoader(x, batch_size=n, shuffle=shuffle)

A = torch.tensor([[0, 1/M], [-N, -C/M]],dtype=torch.float64)
truemean = x.sum()/N
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

def U(theta, data):
    return .5*((data-theta)**2).sum()

def gradU(theta, data):
    return N*theta-data.sum()

def exact(z, expA, noisefactor, data):
    z_ = z
    term = data.mean(axis=-1,keepdims=True)
    z_[:,0] -= term  # convert to zbar
    z_ = expA@z_
    z_[:,0] += term
    zeta=torch.randn(z_.shape).to(z_.dtype)
    noise = noisefactor@zeta
    z_ += noise
    return z_


def Wasserstein1(u, v):
    '''
    Return W1 distance between two densities u,v
    '''
    return np.sum(np.abs(np.cumsum(u,axis=-1)/np.sum(u,axis=-1,keepdims=True)-np.cumsum(
        v,axis=-1)/np.sum(v,axis=-1,keepdims=True)),axis=-1)/np.sum(u,axis=-1)

if C**2/(4*M**2)-N/M<0:
    w=np.sqrt(N/M-C**2/(4*M**2))
    def expA(h):
        c=np.cos(w*h)
        s=np.sin(w*h)
        return torch.exp(-.5*C*h/M)*torch.tensor([[c+.5*C*s/(w*M), s/(w*M)], [-N*s/w, c-.5*C*s/(w*M)]])[None,...]
    # expA = lambda h: torch.linalg.matrix_exp(h*A)

        
elif C**2/(4*M**2)-N/M>0:
    w=np.sqrt(C**2/(4*M**2)-N/M)
    def expA(h):
        c=np.cosh(w*h)
        s=np.sinh(w*h)
        return torch.exp(-.5*C*h/M)*torch.tensor([[c+.5*C*s/(w*M), s/(w*M)], [-N*s/w, c-.5*C*s/(w*M)]])

else: #C=2sqrt(MN)
    a=np.sqrt(N/M)
    def expA(h):
        return torch.exp(-h*a)*torch.tensor([[1.+h*a, h/M], [-N*h, 1-h*a]])

def choldecomp(e):
    A_,B,C_,D=e.flatten()
    L1=torch.sqrt((1-A_**2)/N-M*B**2)
    L2=(-A_*C_/N-M*B*D)/L1
    L3=torch.sqrt((1-D**2)*M-C_**2/N-L2**2)
    return torch.tensor([[L1, 0], [L2, L3]])


# etarange = np.array([2**0,2**-.5,2**-1,2**-1.5,2**-2])/K
mean_err = []
var_err=[]
var_errbar=[]
W1=[]
W1vars=[]
z=torch.zeros(size=(n_paths,2,1))*np.sqrt(M)
z=z.to(dtype=x.dtype)
# z[:,0]=torch.randn(size=(n_paths,1))/np.sqrt(N)+x.mean()[None,...]
fig1=plt.figure()
fig2=plt.figure()
for eta_idx,eta in enumerate(etarange):
    h = torch.tensor([eta])
    # expA_=torch.tensor(np.array([expA(h*g) for g in gammas]))
    expA_=expA(h)
    # noisefactor = torch.tensor(np.array(
    #     [choldecomp(e) for e in expA_]),dtype=expA_.dtype)
    noisefactor=choldecomp(expA_)
    idx = 0
    verr_prog = torch.zeros((epochs[eta_idx]*K)+1,device=x.device)
    var_time=torch.zeros((epochs[eta_idx]*K)+1,device=x.device)
    mybatcher.RRsampler()
    if eta_idx==0: #do a burnin
        for epoch in range(25):
            # mybatcher.RRsampler()
            for k in range(K): 
                k_=np.random.randint(mybatcher.length)
                inds=np.arange(k_,k_+bs)%mybatcher.length
                data=mybatcher.datasource[:,inds]
                # data=(mybatcher.datasource[:,k*bs:(k+1)*bs]).to(x.device)
                z = exact(z, expA_, noisefactor, data)
    i=0
    for epoch in range(epochs[eta_idx]):
        # mybatcher.RRsampler()
        for k in range(K): 
            k_=np.random.randint(mybatcher.length)
            inds=np.arange(k_,k_+bs)%mybatcher.length
            data=mybatcher.datasource[:,inds]
            # data=(mybatcher.datasource[:,k*bs:(k+1)*bs]).to(x.device)
            z = exact(z, expA_, noisefactor, data)
            idx+=1
            var_time[idx]=torch.mean(N*(z[:,0]**2-truemean**2)-1)
            verr_prog[idx]=torch.mean((idx-1)*verr_prog[idx-1]+(N*(z[:,0]**2-truemean**2)-1))/idx
            # counts_fixedpath += hist_laxis(z.T, bins=bins,density=False) #pathwise sample
            # counts_fixedtime += hist_laxis(z, bins=bins,density=False) #plot W1 as function of time
            
    # samples=samples.squeeze(dim=-1)
    # counts_fixedpath = hist_laxis(samples.T, bins=bins,density=True) #pathwise sample
    # counts_fixedtime = hist_laxis(samples, bins=bins,density=True) #plot W1 as function of time

    # plt.plot(.5*(bins[1:]+bins[:-1]), np.mean(counts_fixedpath,axis=0,keepdims=False), label='Analytic')
    # plt.plot(zrange, truedist.flatten(), label='True posterior')
    # plt.legend()
    # plt.title('$'+f'K={K}, \\eta={eta}'+'$'+f', Shuffle={str(shuffle)}')
    num=round(np.log2(h.item()*K),1)
                
    with open(f'figs/SGHMCvartime_1SS_eta{eta}_K{K}.npz','wb') as f:
        np.save(f,var_time.cpu())
    with open(f'figs/SGHMCverrprog_1SS_eta{eta}_K{K}.npz','wb') as f:
        np.save(f,verr_prog.cpu())
        


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

    
    # plt.figure(fig2)
    # W1_time=Wasserstein1(counts_fixedtime,truedist) 
    # plt.semilogy(time.squeeze()/K,
    #               W1_time, 
    #               label='$h=2^{'+f'{num}'+'}$',base=2)
    # fig2title=f'Wasserstein-1 Distance, $K={K}$'

    # W1_path=Wasserstein1(counts_fixedpath,truedist) 
    var_err += [(verr_prog[-1]).mean()] #relative error in variance
    # var_errbar += [np.sqrt(verr_prog[-1].var())/np.sqrt(samples.shape[1])] #relative error in variance
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

K=8
var_errRR=[]
etas=[]
fig2=plt.figure(figsize=(4,3))
ls=[':','-.','-','--']
vprog = [filename for filename in os.listdir('figs/') if filename.startswith("SGHMCverrprog_RR")]
vtime = [filename for filename in os.listdir('figs/') if filename.startswith("SGHMCvartime_RR")]
for idx,(f1,f2) in enumerate(zip(vprog,vtime)):
    eta=float(f1.split('eta')[-1].split('_')[0])
    etas+=[eta]
    num=round(np.log2(eta*K),1)
    with open(os.path.join('figs',f1),'rb') as f:
        verr_prog=np.load(f)
    with open(os.path.join('figs',f2),'rb') as f:
        var_time=torch.tensor(np.load(f))
    print(len(verr_prog))
    plt.figure(fig2)
    if idx>1:
        plt.semilogy(np.arange(len(var_time[-25*K:]))/K,
                      torch.abs(var_time[-25*K:]),'k',ls=ls[idx] ,
                      label='$hK=2^{'+f'{num}'+'}$',base=2)
    fig2title=f'Relative Variance Error, Asymptotic, RR $K={K}$'
    var_errRR += [verr_prog[-1]] #relative error in variance
plt.figure(fig2)
plt.title(fig2title)
plt.xlabel('Iteration over dataset')
plt.legend()
# plt.ylim([2**-8,2**-2])
plt.savefig('figs/SGHMCExpRR.pdf',format='pdf')

hrange=np.array(etas)*K
plt.figure(figsize=(4,3))
# errorbar=var_errbar#np.sqrt(3/len(samples))*np.array(1.+hrange*K)/np.sqrt(1-np.exp(-hrange))
plt.loglog(hrange,var_err1SS,'kX-',ms=8,label='1SS')
plt.loglog(hrange,var_errRR,'ks-',ms=8,label='RR')

# plt.fill_between(hrange, var_err-errorbar, var_err+errorbar,
#                   color='gray', alpha=0.2)
plt.title('$'+f'K=8$, Relative Variance Error'+'; RR vs 1SS')
plt.xlabel('$hK$')
plt.loglog(hrange,np.max(var_err1SS)*(hrange/np.max(hrange)),'k--',label='ref line $(hK)^1$',base=2)
plt.loglog(hrange,np.max(var_errRR)*(hrange/np.max(hrange))**3,'k:',label='ref line $(hK)^3$',base=2,lw=3)
# plt.ylim([2**-4,2**0])
plt.legend()
plt.savefig('figs/SGHMCExpBias.pdf',format='pdf')

plt.figure()
errorbar=var_errbar#np.sqrt(3/len(samples))*np.array(1.+hrange*K)/np.sqrt(1-np.exp(-hrange))
plt.loglog(hrange,var_err,'kX-',ms=10)
# plt.fill_between(hrange, var_err-errorbar, var_err+errorbar,
#                   color='gray', alpha=0.2)
plt.title('$'+f'K={K}$, Relative Variance Error'+'; 1SS')
plt.xlabel('$hK$')
plt.loglog(hrange,np.max(var_err)*(hrange/np.max(hrange)),'k--',label='ref line $h^1$',base=2)
plt.loglog(hrange,np.max(var_err)*(hrange/np.max(hrange))**3,'k:',label='ref line $h^3$',base=2,lw=3)
# plt.ylim([2**-4,2**0])
plt.legend()
plt.savefig('1SS_SGHMC.pdf',format='pdf')

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
def mymatrixfactor(mat):
    newmat=sympy.Matrix.zeros(*mat.shape)
    for idx in range(len(mat)):
        newmat[idx]=mat[idx].factor()
    return newmat

Amat=sympy.Matrix([[0, 1/M], [-N, -C/M]])
iden=sympy.Matrix.eye(2)

#lam_p=-C/2M+sqrt(C**2-4*N*M)/2M
#lam_m=-C/2M-sqrt(C**2-4*N*M)/2M

P=sympy.Matrix([[1,1],[M*lambda_p,M*lambda_m]])
D=sympy.Matrix([[lambda_p,0],[0,lambda_m]])
Pinv=P.inv()
Vmat=sympy.Matrix([[V, 0], [0, 0]])
Vmat_trans=Pinv@Vmat@Pinv.T
eDh=(h*D).exp()
system=(iden-eDh)@Vmat_trans@(iden-eDh)
eDhi=(i*h*D).exp()
eDhj=(j*h*D).exp()
system=eDhi@system@eDhj

##i neq j
expr12=(1-exp(h*lambda_p*r))*(1-exp(h*lambda_m*r))/((1-exp(h*lambda_p))*(1-exp(h*lambda_m)))-(1-exp(r*h*(lambda_m+lambda_p)))/(1-exp(h*(lambda_m+lambda_p)))
expr11=(1-exp(h*lambda_p*r))**2/((1-exp(h*lambda_p))**2)-(1-exp(2*r*h*lambda_p))/(1-exp(2*h*lambda_p))
expr22=(1-exp(h*lambda_m*r))**2/((1-exp(h*lambda_m))**2)-(1-exp(2*r*h*lambda_m))/(1-exp(2*h*lambda_m))

system_ij=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.factor(),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12,
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr12,
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.factor()})) 

##i=j
expr11=(1-exp(2*h*lambda_p*r))/(1-exp(2*h*lambda_p))
expr12=(1-exp(h*r*(lambda_m+lambda_p)))/(1-exp(h*(lambda_m+lambda_p)))
expr22=(1-exp(2*h*lambda_m*r))/(1-exp(2*h*lambda_m))
iisubs={exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11,
                    exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12,
                    exp(i*h*lambda_m)*exp(j*h*lambda_p):expr12,
                    exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22}
system_ii=sympy.simplify(system.subs(iisubs)) 

system_ii=system_ii.subs({V:Vii})
system_ij=system_ij.subs({V:Vij}).replace(Vij,-Vii/(K-1))
system_ij=sympy.simplify(system_ij)

#Case V_b^K sum over n epochs
epoch_sumij=system_ij.subs({r:K})
epoch_sumii=system_ii.subs({r:K})

eDhj=(j*K*h*D).exp()
epoch_sumij=eDhj@epoch_sumij@eDhj
epoch_sumii=eDhj@epoch_sumii@eDhj

# epochsubs={exp(2*K*h*j*lambda_p):(1-exp(2*K*n*h*lambda_p))/(1-exp(2*K*h*lambda_p)),
#                           exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1-exp(n*K*h*(lambda_m+lambda_p)))/(1-exp(K*h*(lambda_m+lambda_p))),
#                           exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1-exp(n*K*h*(lambda_m+lambda_p)))/(1-exp(K*h*(lambda_m+lambda_p))),
#                           exp(2*K*h*j*lambda_m):(1-exp(2*n*K*h*lambda_m))/(1-exp(2*K*h*lambda_m))}
epochsubs={exp(2*K*h*j*lambda_p):(1)/(1-exp(2*K*h*lambda_p)),
                          exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1)/(1-exp(K*h*(lambda_m+lambda_p))),
                          exp(K*h*j*lambda_m)*exp(K*h*j*lambda_p):(1)/(1-exp(K*h*(lambda_m+lambda_p))),
                          exp(2*K*h*j*lambda_m):(1)/(1-exp(2*K*h*lambda_m))}

exprij=epoch_sumij.subs(epochsubs)
exprii=epoch_sumii.subs(epochsubs)
eDh=(r*h*D).exp()
epoch_sumij=sympy.simplify(eDh@exprij@eDh) 
epoch_sumii=sympy.simplify(eDh@exprii@eDh) 

remainder_ij=system_ij
remainder_ii=system_ii

#w=sqrt(C**2-4*N*M)/(2*M)
es_ii=((P@epoch_sumii@P.T)[0]) 
es_ij=((P@epoch_sumij@P.T)[0]) 
rem_ii=((P@remainder_ii@P.T)[0]) 
rem_ij=((P@remainder_ij@P.T)[0]) 

lamplus=-C/(2*M)+sqrt(C**2-4*N*M)/(2*M)
lamminus=-C/(2*M)-sqrt(C**2-4*N*M)/(2*M)
Mval=N
subbers2={C:M,M:Mval,Vii:K}
subbers={lambda_m:lamminus,lambda_p:lamplus}

partii=(es_ii+rem_ii).expand().simplify().factor()
expr=(partii+es_ij+rem_ij).subs(subbers).subs(subbers2)

series_r=(sympy.series(es_ii,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(rem_ii,h,n=4).as_coefficients_dict(h)[h**3]).subs(subbers).simplify()
series_r+=(sympy.series(es_ij,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(rem_ij,h,n=4).as_coefficients_dict(h)[h**3]).subs(subbers).simplify()
series_r=series_r.expand().simplify().factor()
# series_r = K*N**2*V_ii*(K + 1)/(24*C*M)

##Average over period
period_subs={exp(h*lambda_p*r):(1-exp(h*lambda_p*K))/(1-exp(h*lambda_p))/K,
             exp(h*lambda_m*r):(1-exp(h*lambda_m*K))/(1-exp(h*lambda_m))/K,
             exp(2*h*lambda_p*r):(1-exp(2*h*lambda_p*K))/(1-exp(2*h*lambda_p))/K,
             exp(2*h*lambda_m*r):(1-exp(2*h*lambda_m*K))/(1-exp(2*h*lambda_m))/K,
             exp(h*r*(lambda_p+lambda_m)):(1-exp(h*K*(lambda_p+lambda_m)))/(1-exp(h*(lambda_p+lambda_m)))/K}

period_es_ii=es_ii.subs(period_subs)
period_es_ij=es_ij.subs(period_subs)
period_rem_ii=rem_ii.subs(period_subs)
period_rem_ij=rem_ij.subs(period_subs)

period_var_err=(period_es_ii.subs(subbers).subs(subbers2)+
                period_es_ij.subs(subbers).subs(subbers2)+
                period_rem_ii.subs(subbers).subs(subbers2)+
                period_rem_ij.subs(subbers).subs(subbers2))
period_var_err=period_var_err.collect(N)

Kval=16
nperiods=10
plt.figure(figsize=(4,3))
rrange=np.arange(0,nperiods*Kval,dtype=int)
hrange=2.**np.arange(-2,2,1)
varfunc=sympy.lambdify(args=(h,r), expr=expr.replace(K,Kval))
ans=varfunc(hrange[...,None]/Kval,rrange[None,...]%Kval)
periodfunc=sympy.lambdify(args=(h), expr=period_var_err.replace(K,Kval))
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
plt.title(f'SGHMC RR variance error, ${sympy.latex(subbers2)}$, $K={Kval}$')
plt.legend()
plt.savefig(os.path.join(figdir,'SGHMCRR1.pdf'),format='pdf',bbox_inches='tight')

plt.figure(figsize=(4,3))
plt.loglog(hrange,perioderrs,'kX-',base=2,ms=10,label=f'Average error over a period')
scaler=np.min(perioderrs)
plt.loglog(hrange,scaler*(hrange/np.min(hrange))**3,'k--', label='Ref. line $(hK)^3$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$hK$')
plt.title(f'SGHMC RR variance error, ${sympy.latex(subbers2)}$, $K={Kval}$')
plt.legend()
plt.savefig(os.path.join(figdir,'SGHMCRR2.pdf'),format='pdf',bbox_inches='tight')


Kval=16
newexpr=expr.subs({K:Kval,r:Kval//4})
plt.figure()
nrange=np.logspace(0,5)
for hval in hrange:
    varfunc=sympy.lambdify(n, expr=newexpr.subs({h:hval/Kval}))
    plt.loglog(nrange,varfunc(nrange),base=10,label='$hK=2^{'+f'{round(np.log2(hval))}'+'}$')
    # plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$n$')
plt.title(f'RR variance error, ${sympy.latex(subbers2)}$, $K={Kval}$')
plt.legend()


##Average over a period, accounting for correlations##
period_series=(sympy.series(period_es_ii,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(period_rem_ii,h,n=4).as_coefficients_dict(h)[h**3]).subs(subbers).simplify()
period_series+=(sympy.series(period_es_ij,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(period_rem_ij,h,n=4).as_coefficients_dict(h)[h**3]).subs(subbers).simplify()
period_series=period_series.expand().simplify().factor()
## period_series= h**3*N**2*Vii*(K + 1)*(C**2*K + C**2 - 4*K*M*N)/(24*C*M*(C**2 - 4*M*N))


plt.figure()
for Kval in 2**np.arange(1,10,2):
    varfunc=sympy.lambdify(h, expr=period_var_err.replace(K,Kval))
    plt.loglog(hrange,varfunc(hrange/Kval),base=2,label=f'$K={Kval}$')
    # plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$hK$')
plt.title(f'RR variance error, ${sympy.latex(subbers2)}$')
plt.legend()

#TODO: Critically Damped Case C=2sqrt(MN)

#%% Shuffle Once
P=sympy.Matrix([[1,1],[M*lambda_p,M*lambda_m]])
D=sympy.Matrix([[lambda_p,0],[0,lambda_m]])
Pinv=P.inv()
Vmat=sympy.Matrix([[V, 0], [0, 0]])
Vmat_trans=Pinv@Vmat@Pinv.T
eDh=(h*D).exp()
system=(iden-eDh)@Vmat_trans@(iden-eDh)
eDhi=(i*h*D).exp()
eDhj=(j*h*D).exp()
system=eDhi@system@eDhj
r1,r2=sympy.symbols("r1:3", positive=True)

##i neq j r1<=r2
expr21=(1-exp(h*lambda_p*r2))*(1-exp(h*lambda_m*r1))/((1-exp(h*lambda_p))*(1-exp(h*lambda_m)))-(1-exp(r1*h*(lambda_m+lambda_p)))/(1-exp(h*(lambda_m+lambda_p)))
expr12=(1-exp(h*lambda_p*r1))*(1-exp(h*lambda_m*r2))/((1-exp(h*lambda_p))*(1-exp(h*lambda_m)))-(1-exp(r1*h*(lambda_m+lambda_p)))/(1-exp(h*(lambda_m+lambda_p)))
expr11=((1-exp(h*lambda_p*r2))*(1-exp(h*lambda_p*r1))/((1-exp(h*lambda_p))**2)-(1-exp(2*r1*h*(lambda_p)))/(1-exp(2*h*(lambda_p)))).factor()
expr22=((1-exp(h*lambda_m*r2))*(1-exp(h*lambda_m*r1))/((1-exp(h*lambda_m))**2)-(1-exp(2*r1*h*(lambda_m)))/(1-exp(2*h*(lambda_m)))).factor()

Vbrr_ij=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:r,r2:r}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:r,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr21.subs({r1:r,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:r,r2:r})})) 

VbKK_ij=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:K,r2:K}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:K,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr21.subs({r1:K,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:K,r2:K})})) 

VbrK_ij=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:r,r2:K}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:r,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr21.subs({r1:r,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:r,r2:K})})) 

VbKr_ij=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:K,r2:r}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:K,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr21.subs({r1:K,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:K,r2:r})})) 

##i=j r1<r2
expr11=(1-exp(2*h*lambda_p*r1))/(1-exp(2*h*lambda_p))
expr12=(1-exp(h*r1*(lambda_m+lambda_p)))/(1-exp(h*(lambda_m+lambda_p)))
expr22=(1-exp(2*h*lambda_m*r1))/(1-exp(2*h*lambda_m))

Vbrr_ii=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:r,r2:r}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:r,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr12.subs({r1:r,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:r,r2:r})})) 

VbKK_ii=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:K,r2:K}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:K,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr12.subs({r1:K,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:K,r2:K})})) 

VbrK_ii=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:r,r2:K}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:r,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr12.subs({r1:r,r2:K}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:r,r2:K})})) 

VbKr_ii=sympy.simplify(system.subs({exp(h*i*lambda_p)*exp(h*j*lambda_p):expr11.subs({r1:K,r2:r}),
                       exp(i*h*lambda_p)*exp(j*h*lambda_m):expr12.subs({r1:K,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_p):expr12.subs({r1:K,r2:r}),
                       exp(i*h*lambda_m)*exp(j*h*lambda_m):expr22.subs({r1:K,r2:r})})) 

[Vbrr_ii,VbKK_ii,VbKr_ii,VbrK_ii]= [expr.subs({V:Vii}) for expr in [Vbrr_ii,VbKK_ii,VbKr_ii,VbrK_ii]]
[Vbrr_ij,VbKK_ij,VbKr_ij,VbrK_ij]= [expr.subs({V:-Vii/(K-1)}) for expr in [Vbrr_ij,VbKK_ij,VbKr_ij,VbrK_ij]]

#Vb_KK sum over (infinite) epochs
eDhi=(i*K*h*D).exp()
eDhj=(j*K*h*D).exp()
es_KKij=eDhi@VbKK_ij@eDhj
es_KKii=eDhi@VbKK_ii@eDhj

expr11=(1-exp(K*h*lambda_p))**(-2)
expr12=1/(1-exp(K*h*(lambda_m)))/(1-exp(K*h*(lambda_p)))
expr22=(1-exp(K*h*lambda_m))**(-2)

expr_subs={exp(K*h*i*lambda_p)*exp(K*h*j*lambda_p):expr11,
                          exp(K*h*i*lambda_p)*exp(K*h*j*lambda_m):expr12,
                          exp(K*h*i*lambda_m)*exp(K*h*j*lambda_p):expr12,
                          exp(K*h*i*lambda_m)*exp(K*h*j*lambda_m):expr22}
es_KKii=es_KKii.subs(expr_subs)
es_KKij=es_KKij.subs(expr_subs)

eDh=(r*h*D).exp()
es_KKii=sympy.simplify(eDh@es_KKii@eDh) 
es_KKij=sympy.simplify(eDh@es_KKij@eDh) 

#Vb_rr
remainder_ij=Vbrr_ij
remainder_ii=Vbrr_ii

#Vb_Kr and Vb_rK sum
eDhi=(i*K*h*D).exp()
es_Krij=eDhi@VbKr_ij
es_Krii=eDhi@VbKr_ii
es_rKij=VbKr_ij@eDhi
es_rKii=VbKr_ii@eDhi

ex_p=1/(1-exp(K*h*lambda_p))
ex_m=1/(1-exp(K*h*lambda_m))

expr_subs={exp(K*h*i*lambda_p):ex_p,exp(K*h*i*lambda_m):ex_m}
es_Krii=es_Krii.subs(expr_subs)
es_Krij=es_Krij.subs(expr_subs)
es_rKii=es_rKii.subs(expr_subs)
es_rKij=es_rKij.subs(expr_subs)

eDh=(r*h*D).exp()
es_Krii=sympy.simplify(eDh@es_Krii) 
es_Krij=sympy.simplify(eDh@es_Krij) 
es_rKii=sympy.simplify(es_rKii@eDh) 
es_rKij=sympy.simplify(es_rKij@eDh) 

#various components
es_KKii=((P@es_KKii@P.T)[0]) 
es_KKij=((P@es_KKij@P.T)[0]) 
es_Krii=((P@es_Krii@P.T)[0]) 
es_Krij=((P@es_Krij@P.T)[0]) 
es_rKii=((P@es_rKii@P.T)[0]) 
es_rKij=((P@es_rKij@P.T)[0]) 
rem_ii=((P@remainder_ii@P.T)[0]) 
rem_ij=((P@remainder_ij@P.T)[0]) 

lamplus=-C/(2*M)+sqrt(C**2-4*N*M)/(2*M)
lamminus=-C/(2*M)-sqrt(C**2-4*N*M)/(2*M)
Mval=N
subbers2={C:M,M:Mval,Vii:K}
subbers={lambda_m:lamminus,lambda_p:lamplus}

partii=(es_KKii+es_Krii+es_rKii+rem_ii)
expr=(partii+es_KKij+es_Krij+es_rKij+rem_ij).subs(subbers).subs(subbers2)

series_r=(sympy.series(es_KKii,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(
    es_Krii,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(
        es_rKii,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(
            rem_ii,h,n=4).as_coefficients_dict(h)[h**3]).subs(subbers).simplify()
series_r+=(sympy.series(es_KKij,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(
    es_Krij,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(
        es_rKij,h,n=4).as_coefficients_dict(h)[h**3]+sympy.series(
            rem_ij,h,n=4).as_coefficients_dict(h)[h**3]).subs(subbers).simplify()
series_r=series_r.expand().simplify().factor()


series_r=(sympy.series(es_KKii,h,n=3)+sympy.series(es_Krii,h,n=3)+sympy.series(es_rKii,h,n=3)+sympy.series(rem_ii,h,n=3)).subs(subbers).collect(h)
series_r+=(sympy.series(es_KKij,h,n=3)+sympy.series(es_Krij,h,n=3)+sympy.series(es_rKij,h,n=3)+sympy.series(rem_ij,h,n=3)).subs(subbers).collect(h)
series_r=series_r.factor().collect(h)

Kval=64
nperiods=10
plt.figure(figsize=(4,3))
rrange=np.arange(0,nperiods*Kval,dtype=int)
hrange=2.**np.arange(-8,6,.25)
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
plt.title(f'SO variance error, ${sympy.latex(subbers2)}$, $K={Kval}$')
plt.legend()

Kval=64
newexpr=expr.subs({K:Kval})
plt.figure()
for rval in np.linspace(0,Kval,8,dtype=int):
    varfunc=sympy.lambdify(h, expr=newexpr.replace(r,rval))
    plt.loglog(hrange,varfunc(hrange),base=2,label=f'$r={rval}$')
    # plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$h$')
plt.title(f'SO variance error, ${sympy.latex(subbers2)}$, $K={Kval}$')
plt.legend()

#%% 1SS and 1SSwR
P=sympy.Matrix([[1,1],[M*lambda_p,M*lambda_m]])
D=sympy.Matrix([[lambda_p,0],[0,lambda_m]])
Pinv=P.inv()
Vmat=sympy.Matrix([[V, 0], [0, 0]])
Vmat_trans=Pinv@Vmat@Pinv.T
#lam_p=-C/2M+sqrt(C**2-4*N*M)/2M
#lam_m=-C/2M-sqrt(C**2-4*N*M)/2M
eDh=(h*D).exp()
system=(iden-eDh)@Vmat_trans@(iden-eDh)
eDh=(i*h*D).exp()
system=eDh@system@eDh
var_err=(P@system@P.T)[0]

# epochsubs={exp(2*i*h*lambda_p):(1-exp(2*h*K*lambda_p))/(1-exp(2*h*lambda_p))*(1-exp(2*h*K*n*lambda_p))/(1-exp(2*h*K*lambda_p)),
#                       exp(2*i*h*lambda_m):(1-exp(2*h*K*lambda_m))/(1-exp(2*h*lambda_m))*(1-exp(2*h*K*n*lambda_m))/(1-exp(2*h*K*lambda_m)),
#                       exp(i*h*lambda_p)*exp(i*h*lambda_m):(1-exp(h*K*(lambda_m+lambda_p)))/(1-exp(h*(lambda_m+lambda_p)))*(1-exp(h*K*n*(lambda_m+lambda_p)))/(1-exp(h*K*(lambda_m+lambda_p)))}
epochsubs={exp(2*i*h*lambda_p):1/(1-exp(2*h*lambda_p)),
                      exp(2*i*h*lambda_m):1/(1-exp(2*h*lambda_m)),
                      exp(i*h*lambda_p)*exp(i*h*lambda_m):1/(1-exp(h*(lambda_m+lambda_p)))}

var_err=var_err.subs(epochsubs)

var_err=var_err.subs(subbers).expand().factor()

myseries=var_err.series(h,n=2)
term1=myseries.coeff(h).simplify()

expr=var_err.replace(V,Vii).simplify().subs(subbers).subs(subbers2)

Kval=32
plt.figure(figsize=(4,3))
hrange=2.**np.arange(-8,6,.25)
varfunc=sympy.lambdify(args=(h), expr=expr.replace(K,Kval))
ans=varfunc(hrange/Kval)
plt.semilogy(hrange,ans,base=2,ls='-')
plt.loglog(hrange,ans[-1]*hrange/np.min(hrange),'b--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error (units of $V_x$)')
plt.xlabel('$hK$')
plt.title(f'1SS variance error, ${sympy.latex(subbers2)}$, $K={Kval}$')
plt.legend()

##Critically damped case C=2sqrt(MN)
lam=-sqrt(N/M)
Amat=sympy.Matrix([[0, 1/M], [-N, 2*lam]]) #sigma**2=1 for simplicity
iden=sympy.Matrix.eye(2)
J=sympy.Matrix([[lam,1],[0,lam]])

P=sympy.Matrix([[1,M*lam],[M*lam,M*(N+1)]])
Pinv=P.inv()
Vmat=sympy.Matrix([[V, 0], [0, 0]])
Vmat_trans=Pinv@Vmat@Pinv.T
eDh=(h*J).exp()
system=(iden-eDh)@Vmat_trans@(iden-eDh.T)
eDh=(i*h*J).exp()
system=eDh@system@eDh.T
var_err=(P@system@P.T)[0].simplify()

var_err=var_err.expand()
var_err=var_err.subs({i**2*exp(2*i*h*lam):(exp(-h*lam)+exp(h*lam))/(exp(-h*lam)-exp(h*lam))**3})
var_err=var_err.subs({i*exp(2*i*h*lam):1/(exp(h*lam)-exp(-h*lam))**2})
var_err=var_err.subs({exp(2*i*h*lam):1/(1-exp(2*h*lam))}).simplify()

var_err=var_err.simplify()
myseries=var_err.series(h,n=4)
term1=myseries.coeff(h).simplify()

expr=var_err.subs({V:1}).subs({M:A*N}).simplify()

hrange=np.logspace(-2,6,base=2)
for val in [.01,.1,1,10]:
    varfunc=sympy.lambdify(h, expr=expr.subs({A:val}))
    plt.loglog(hrange,varfunc(hrange),label=f'$M={val}N$',base=2)
plt.loglog(hrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('variance error $V_q-1/N$ (units of $V_x$)')
plt.xlabel('$h$')
plt.title('1SSwR and 1SS variance error, critically damped $C=2\\sqrt{MN}$')
plt.legend()


# %%
mat = torch.randn(4, 4)**2
mat /= mat.sum(dim=0)

hrange=np.logspace(start=-3.,stop=-5.,num=10)
T=1
case1=[]
case2=[]
case3=[]
for h in hrange:
    n = int(T//h)
    nrange=torch.arange(1,n+1,1)
    ans = torch.linalg.matrix_exp(h*nrange.reshape(-1,1,1)*mat)

    # case 1
    p = np.random.uniform(low=0., high=1.)
    A = torch.linalg.matrix_exp(h*p*mat)
    B = torch.linalg.matrix_exp(h*(1.-p)*mat)
    err1=torch.zeros(n)
    for i in range(n):
        err1[i] = torch.linalg.norm(torch.matrix_power(A@B,i+1)-ans)
    case1+=[err1.max()]
    
    # case 2
    p = np.random.uniform(low=0., high=1.)
    A = torch.linalg.matrix_exp(h*p*mat)
    B = torch.linalg.matrix_exp(h*(1-p)*mat)
    r = np.random.uniform(low=0., high=1.)
    if r > .5:
        m1, m2 = A, B
    else:
        m1, m2 = B, A
    result = m1@m2
    err2=torch.zeros(n)
    err2[0]=torch.linalg.norm(result-ans)
    for i in range(1,n):
        r = np.random.uniform(low=0., high=1.)
        if r > .5:
            m1, m2 = A, B
        else:
            m1, m2 = B, A
        result = (m1@m2)@result
        err2[i] = torch.linalg.norm(result-ans)
    case2+=[err2.max()]

    
    # case 3
    p = np.random.uniform(low=0., high=1.)
    A = torch.linalg.matrix_exp(h*p*mat)
    B = torch.linalg.matrix_exp(h*(1-p)*mat)
    result=A@B
    err3=torch.zeros(n)
    err3[0]=torch.linalg.norm(result-ans)
    for i in range(n-1):
        p = np.random.uniform(low=0., high=1.)
        A = torch.linalg.matrix_exp(h*p*mat)
        B = torch.linalg.matrix_exp(h*(1-p)*mat)
        result = (A@B)@result
        err3[i] = torch.linalg.norm(result-ans)
    case3+=[err3.max()]

hrange=np.flip(hrange)
plt.figure()
plt.loglog(hrange,case1,label='case1')
plt.loglog(hrange,case2,label='case2')
plt.loglog(hrange,case3,label='case3')
plt.legend()
