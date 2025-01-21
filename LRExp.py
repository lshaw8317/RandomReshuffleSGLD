import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from torch import cholesky_solve as cho_solve
from torch.linalg import solve_triangular, cholesky
import os
from scipy.optimize import fsolve,minimize
from torch import sigmoid as expit
from scipy.special import expit as npexpit
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import pickle
plt.rcParams.update({'text.usetex':True,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.rc('text', usetex=True)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-v0_8-paper')
import time as time
figdir='figs'

#%%
class MyBatcher:
    def __init__(self,data,bs,n_paths,strat):
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

class HMCIntegrators:
    def __init__(self, J, Jchol, MAP,mybatcher):
        self.J=J
        self.Jchol=Jchol
        self.MAP=MAP
        self.mybatcher=mybatcher
    
    def U(self,q):
        return 
    
    def grad(self,q):
        return 
    
    def stochgrad(self,q,data):
        return 

    def partialgrad(self,q):
        U_dash=self.grad(q)
        return U_dash-self.J@(q-self.MAP)
    
    def set_strat(self,strat):
        if strat=='RR':
            self.mybatcher.set_strat('RR')
        else:
            self.mybatcher.set_strat('1SS')

    
    def HMCsample(self,q0,hmed,T,Nsamples):
        q=q0.detach()
        acc=0
        ham=lambda q,v:.5*torch.sum(v*(self.J[None,...]@v))+self.U(q)
        samples=torch.zeros((Nsamples,*q.shape))
        Nsteps=int(torch.floor(T/hmed))
        for n in range(0,Nsamples):
            h=(1.-0.2*torch.rand(1))*hmed
            v=solve_triangular(self.Jchol[None,...], torch.randn_like(q),upper=True) #Draw v ~ N(0,Jinv)
            qp=q.detach()
            H0=ham(q,v)
            
            #Do a leg of T//h steps of Strang
            qp,v=self.PrecondVerlet(qp,v,Nsteps,h)

            accept=H0-ham(qp,v) #acceptance probability
            #Accept/reject
            if (accept>torch.log(torch.rand(1))):
                acc+=1
                q=qp 
            else:
                pass
            samples[n]=q
        return acc/Nsamples,samples
    
    def LDsample(self,q0,hmed,Nsamples,pcond=False,stoch=True):
        q=q0.detach()
        samples=torch.zeros((Nsamples,*q.shape))
        # log_posts = np.empty(Nsamples)
        if stoch:
            integ=self.PSGLD if pcond else self.SGLD
        else:
            integ=self.PLD if pcond else self.LD
        #Burnin
        for n in range(0,1000):
            q=integ(q,hmed)
            
        for n in range(0,Nsamples):
            q=integ(q,hmed)
            # log_posts[n] = self.log_posterior(q)
            samples[n]=q
        return samples,None #log_posts

    ##PrecondIntegrators##
    def PrecondVerlet(self,qp,v,Nsteps,h):
        #Do a leg of T//h steps of Strang
        #(b1) Kick
        theta1=h/2
        v-=theta1*cho_solve(self.grad(qp),self.Jchol[None,...],upper=True)
        for t in torch.arange(Nsteps):
            qp+=h*v #Drift
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            v-=theta*cho_solve(self.grad(qp),self.Jchol[None,...],upper=True)
        return qp,v

    
    ##Stochastic Gradient##
    def PSGLD(self,qp,h):
        data=self.mybatcher.sample()
        eta = torch.sqrt(2*h)*solve_triangular(self.Jchol[None,...], torch.randn_like(qp),upper=True)
        grad = self.stochgrad(qp,data)
        update=cho_solve(grad,self.Jchol[None,...],upper=True)
        qp = qp - h*update + eta
        return qp

    def SGLD(self,qp,h):
        data=self.mybatcher.sample()
        eta = torch.sqrt(2*h)*torch.randn_like(qp)
        grad = self.stochgrad(qp,data)
        update=grad
        qp = qp - h*update + eta
        return qp
    
    ##Full Gradient##
    def LD(self,qp,h):
        eta = torch.sqrt(2*h)*torch.randn_like(qp)
        grad = self.grad(qp)
        update=grad
        qp = qp - h*update + eta
        return qp
    
    def PLD(self,qp,h):
        eta = torch.sqrt(2*h)*solve_triangular(self.Jchol[None,...], torch.randn_like(qp),upper=True)
        grad = self.grad(qp)
        update=cho_solve(grad,self.Jchol[None,...],upper=True)
        qp = qp - h*update + eta
        return qp
    
class LogRegExp(HMCIntegrators):
    def __init__(self,data,bs,n_paths,strat='1SS'):
        self.x,self.y=data
        self.n=self.x.shape[0] 
        #Add dummy for bias
        self.xnew=torch.cat((self.x,torch.ones((self.n,1))),dim=1)
        I=torch.eye(self.xnew.shape[1]).to(self.xnew.dtype)
        self.C=I*25.
        self.Cinv=I/25.

        MAP=torch.tensor(self.calc_MAP())
        #calculate hessian at MAP
        arg=self.xnew@MAP
        J=self.xnew.T*(expit(arg)*expit(-arg))@self.xnew
        J+=self.Cinv
        Jchol=cholesky(J, upper=True)
        data_comb=torch.cat((self.xnew,self.y[...,None]),dim=-1)
        mybatcher=MyBatcher(data=data_comb,bs=bs,n_paths=n_paths,strat=strat)
        super().__init__(J, Jchol, MAP,mybatcher)

    def U(self,q):
        arg=self.xnew@q
        ans=-torch.sum(self.y[None,...,None]*arg)
        ans+=torch.sum(torch.logaddexp(torch.zeros_like(arg),arg))
        term=q*torch.matmul(self.Cinv[None,...],q)
        return .5*torch.sum(term)+ans
    
    def grad(self,q):
        term=torch.matmul(self.Cinv[None,...],q)
        arg=torch.matmul(self.xnew,q)
        temp=self.y[None,...,None]-expit(arg) #has shape (n_paths,n,1)
        return term-torch.matmul(self.xnew.T[None,...],temp)
    
    def _U(self,q): ##np version
        arg=self.xnew@q
        ans=-np.dot(self.y,arg)
        ans+=(np.logaddexp(0.0,arg)).sum()
        return .5*(np.dot(q,self.Cinv@q))+ans

    def _grad(self,q): ##np version
        term=self.Cinv@q
        arg=self.xnew@q
        temp=(self.y-npexpit(arg))
        return term-self.xnew.T@temp
    
    def stochgrad(self,q,data):
       x,y=data[...,:-1],data[...,-1] #x has shape (n_paths,n,n_features)
       term=torch.matmul(self.Cinv[None,...],q) #q has shape (n_paths,n_features,1)
       arg=torch.matmul(x,q) #has shape (n_paths,n,1)
       temp=y[...,None]-expit(arg) #has shape (n_paths,n,1)
       return term-self.mybatcher.K*torch.matmul(x.permute(0,2,1),temp)
    
    def calc_MAP(self):
        x0=np.random.randn(*self.xnew.shape[1:])*.2
        guess=fsolve(self._grad,x0=x0)
        return minimize(self._U,x0=guess).x


class GaussianExp(HMCIntegrators):
    def __init__(self,x,bs,n_paths,strat='1SS'):
        self.x=x
        self.n=x.shape[0] # # of data points, which should always be > # of params
        MAP=self.x.mean(dim=0)
        self.truemean=MAP.unsqueeze(-1)
        self.cov=torch.cov(x.T).reshape(x.shape[1],x.shape[1]) #x has shape (n,features)
        self.truecov=self.cov/self.n

        mybatcher=MyBatcher(data=self.x,bs=bs,n_paths=n_paths,strat=strat)
        Jchol=cholesky(torch.linalg.inv(self.cov), upper=True)*torch.sqrt(torch.tensor(self.n))
        super().__init__(Jchol.T@Jchol, Jchol, MAP,mybatcher)
        
        
    def U(self,q):
        arg=torch.matmul(self.Jchol[None,...], (q-self.truemean[None,...]))
        ans=torch.sum(arg*arg)
        return .5*ans
    
    def grad(self,q):
        return torch.matmul(self.J[None,...], (q-self.truemean[None,...]))

    def stochgrad(self,q,data):
        return torch.matmul(self.J[None,...],q-data.mean(dim=1,keepdims=False)[...,None])


def getHMC(Exp1,h,Tp,Nsamples):
    #Preconditioned
    start=time.time()
    q0=Exp1.MAP[None,...].unsqueeze(-1)
    acc,s=Exp1.HMCsample(q0, h, Tp,Nsamples)
    end=time.time()
    samples={'acc':acc,'samples':s,'h':h,'T':Tp,'Exp':Exp1,'type':'pHMC'}
    samples['exec_time']=end-start
    print('HMC'+f', acc={acc}')
    return samples

def getLD(Exp1,h, Nsamples,strat='RR',pcond=True,stoch=True):
    start=time.time()
    if stoch:
        Exp1.set_strat(strat)
    else:
        strat='FULLGRAD'
    shape=tuple([Exp1.mybatcher.n_paths]+[1 for i in Exp1.MAP.shape])
    q0=Exp1.MAP[None,...].repeat(shape).unsqueeze(-1)
    s,lp=Exp1.LDsample(q0, h, pcond=pcond,Nsamples=Nsamples,stoch=stoch)
    end=time.time()
    samples={'samples':s,'h':h,'logpost':lp,'RandPol':strat,'stoch':stoch}
    samples['exec_time']=end-start
    return samples



#%% Gaussian 1D: Sim Data
torch.manual_seed(2024)
K=2**3
N=20*K
bs=N//K
d=1
n_paths=10000
x=torch.randn((N,d,1)).squeeze(-1) #number of examples, number of features, 1
Exp1=GaussianExp(x, bs,n_paths=n_paths)

with open("Gaussian1D_truemeancov.pkl", 'wb') as f:
    pickle.dump({'truemean':Exp1.truemean,'truecov':Exp1.truecov},f)

with open("Gaussian1D_truemeancov.pkl", 'rb') as f:
    d=pickle.load(f)
    truecov=d['truecov']
    truemean=d['truemean']

etarange = (2.**torch.arange(-3,1,1)*1.1)/K
Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)
truecov,truemean=Exp1.truecov,Exp1.truemean
strats=['RR','1SS']
sgld_dict={s:{} for s in strats}
sgld_dict['K']=K
sgld_dict['etarange']=etarange
for strat in strats:
    for i,timestep in enumerate(etarange):
        stoch =False if strat=='FULLGRAD' else True
        samples=getLD(Exp1,timestep,Nsamples[i],strat=strat,pcond=True,stoch=stoch) 
        sgld_dict[strat][str(timestep.item())]=samples

Exp1=GaussianExp(x, bs,n_paths=1000)
strat='FULLGRAD'
sgld_dict[strat]={}
for i,timestep in enumerate(etarange):
    stoch =False if strat=='FULLGRAD' else True
    samples=getLD(Exp1,timestep,Nsamples[i],strat=strat,pcond=True,stoch=stoch) 
    sgld_dict[strat][str(timestep.item())]=samples

exp1_file = open(f"Gaussian1DK{K}.pkl", 'wb')
pickle.dump(sgld_dict,exp1_file)
exp1_file.close()

with open(f"Gaussian1DK{K}.pkl", 'rb') as f:
    sgld_dict=pickle.load(f)

##Plotting
strats=['RR','1SS','FULLGRAD']
err={s:[] for s in strats}
RRplot=plt.figure()
K=sgld_dict['K']
for strat in strats:
    loc=sgld_dict[strat]
    for i,timestep in enumerate(loc.keys()):
        stoch =False if strat=='FULLGRAD' else True
        samples=loc[timestep]
        s=samples['samples']
        c=torch.matmul(s,s.permute(0,1,3,2)).mean(dim=0)
        musq=truemean@truemean.T
        cov_SGLD=(c-musq[None,...]).mean(dim=0) #shape (n_paths,n_features)
        err[strat]+=[torch.linalg.norm(cov_SGLD-truecov)/torch.linalg.norm(truecov)]
        c1=torch.matmul(s,s.permute(0,1,3,2)).mean(dim=1)
        cov_SGLD=(c1-musq[None,...])[-10*K:] #shape (n_iters,n_paths,n_features)
        e1=torch.linalg.norm(((cov_SGLD-truecov)/truecov).flatten(start_dim=1,end_dim=-1),dim=-1)
        h=samples['h']
        # Plot oscillations
        if i==(len(loc.keys())-1):
            plt.figure(figsize=(3,2))
            num=round(h.item(),1)
            stratlab='RM' if strat=='1SS' else strat
            plt.title('Experimental Variance Error: ' + 'SGLD-'+stratlab+', $h='+f'{num}'+'$')
            
            plt.semilogy(np.arange(len(e1))/K,
                                  torch.abs(e1),'k',ls='-' ,base=2)
            plt.xlabel('Iteration over dataset')
            # if strat=='RR':
                # plt.ylabel('Relative variance error')
            plt.savefig(os.path.join(figdir,f'Gaussian1DK{K}_Oscillations{strat}.pdf'),format='pdf',bbox_inches='tight')

# markerlist=['s','X','o']
markerlist=['','','']
etarange=sgld_dict['etarange']
plt.figure(figsize=(3,2))
plt.loglog(etarange,err['1SS'],'b-',base=2,label='SGLD-RM',marker=markerlist[1])
plt.loglog(etarange,err['RR'],'r-',base=2,label='SGLD-RR',marker=markerlist[0])
plt.loglog(etarange,err['FULLGRAD'],'k-',base=2,label='ULA',marker=markerlist[2])

plt.title(f'Experimental Variance Error, $R={K}$')
plt.xlabel('$h$')
# plt.ylabel('$\|\Delta\Sigma\|/\|\Sigma\|$')
plt.legend()
plt.savefig(os.path.join(figdir,f'Gaussian1DK{K}.pdf'),format='pdf',bbox_inches='tight')

# with open("Gaussian2D_truemeancov.pkl", 'wb') as f:
#     pickle.dump({'truemean':Exp1.truemean,'truecov':Exp1.truecov},f)
#%% Gaussian 2D: Sim Data
torch.manual_seed(2024)
N=2**10
C=torch.tensor([[1.,.5],[.5,1.]])
d=len(C)
K=2**5
bs=N//K

n_paths=1000
x=torch.matmul(cholesky(C)[None,...],torch.randn((N,d,1))).squeeze(-1) #number of examples, number of features, 1
Exp1=GaussianExp(x, bs,n_paths=n_paths)


with open("Gaussian2D_truemeancov.pkl", 'wb') as f:
    pickle.dump({'truemean':Exp1.truemean,'truecov':Exp1.truecov},f)

with open("Gaussian2D_truemeancov.pkl", 'rb') as f:
    d=pickle.load(f)
    truecov=d['truecov']
    truemean=d['truemean']

etarange = 2.**torch.arange(-2,6,1)/K
Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)
truecov,truemean=Exp1.truecov,Exp1.truemean
strats=['RR','1SS','FULLGRAD']
err={s:[] for s in strats}
sgld_dict={s:{} for s in strats}
sgld_dict['K']=K
sgld_dict['etarange']=etarange
for strat in strats:
    for i,timestep in enumerate(etarange):
        stoch =False if strat=='FULLGRAD' else True
        samples=getLD(Exp1,timestep,Nsamples[i],strat=strat,pcond=True,stoch=stoch) 
        sgld_dict[strat][str(timestep.item())]=samples
        
exp1_file = open(f"Gaussian2DK{K}.pkl", 'wb')
pickle.dump(sgld_dict,exp1_file)
exp1_file.close()

with open(f"Gaussian2DK{K}.pkl", 'rb') as f:
    sgld_dict=pickle.load(f)

##Plotting
err={s:[] for s in strats}
RRplot=plt.figure()
K=sgld_dict['K']
for strat in strats:
    loc=sgld_dict[strat]
    for i,timestep in enumerate(loc.keys()):
        stoch =False if strat=='FULLGRAD' else True
        samples=loc[timestep]
        s=samples['samples']
        c=torch.matmul(s,s.permute(0,1,3,2)).mean(dim=0)
        musq=truemean@truemean.T
        cov_SGLD=(c-musq[None,...]).mean(dim=0) #shape (n_paths,n_features)
        err[strat]+=[torch.linalg.norm(cov_SGLD-truecov)/torch.linalg.norm(truecov)]
        c1=torch.matmul(s,s.permute(0,1,3,2)).mean(dim=1)
        cov_SGLD=(c1-musq[None,...])[-10*K:] #shape (n_iters,n_paths,n_features)
        e1=torch.linalg.norm(((cov_SGLD-truecov)/truecov).flatten(start_dim=1,end_dim=-1),dim=-1)
        h=samples['h']
        #Plot oscillations
        plt.figure()
        num=round(np.log2(h.item()*K),1)
        
        plt.semilogy(np.arange(len(e1))/K,
                              torch.abs(e1),'k',ls='--' ,
                              label='$hK=2^{'+f'{num}'+'}$',base=2)
   
plt.figure(RRplot)
fig2title=f'Relative Variance Error, Asymptotic, RR $K={K}$'
plt.title(fig2title)
plt.xlabel('Iteration over dataset')
plt.legend()
plt.show()

markerlist=['s','X','o']
plt.loglog(etarange,err['RR'],'k-',base=2,label='RR',marker=markerlist[0])
plt.loglog(etarange,err['1SS'],'k-',base=2,label='1SS',marker=markerlist[1])
plt.loglog(etarange,err['FULLGRAD'],'k-',base=2,label='FULLGRAD',marker=markerlist[2])
plt.loglog(etarange,etarange,'k--',base=2,alpha=0.7)
plt.loglog(etarange,(etarange)**2,'k--',base=2,alpha=0.7)
# plt.loglog(etarange,(etarange)**1.5,'k--',base=2,alpha=0.7)

l=len(etarange)
c=(etarange[1]+etarange[2])/2
plt.text(c,1.3*c,'$h$')
plt.text(c,1.3*c**2,'$h^2$')
# plt.text(c,1.3*c**1.5,'$h^{3/2}$')
plt.title(f'Gaussian 2D, $K={2**5}$')
plt.xlabel('$h$')
plt.ylabel('$\\frac{\|\Delta\Sigma\|}{\|\Sigma\|}$')
plt.legend()
# plt.savefig(os.path.join(figdir,f'Gaussian2DK{K}.pdf',format='pdf',bbox_inches='tight')

# with open("Gaussian2D_truemeancov.pkl", 'wb') as f:
#     pickle.dump({'truemean':Exp1.truemean,'truecov':Exp1.truecov},f)
#%% LogReg Experiment 1: Sim Data
Nsamples_HMC=10**6
# np.random.seed(2024)
# d=25
# p=d+1
# N=2**10
# scaler=np.hstack((5*np.ones(shape=(1,5)),np.ones(shape=(1,5)),.2*np.ones(shape=(1,d-10))))
# params=np.random.normal(size=(p,))
# x=np.random.normal(size=(N,d),scale=scaler) #input data
# xnew=np.hstack((np.ones(shape=(N,1)),x))
# p_i=expit(torch.tensor(xnew@params))
# y=np.random.binomial(1, p_i).flatten() # output data
# sig_sq=25 #priors
# C=sig_sq

# exp1_file = open("SimDataSmall.pkl", 'wb')
# pickle.dump({'x':x,'y':y,'params':params},exp1_file)
# exp1_file.close()

with open("SimDataSmall.pkl", 'rb') as f:
    d=pickle.load(f)
    x=torch.tensor(d['x'])
    y=torch.tensor(d['y'])

N=len(x)
K=2**3
bs=N//K

Exp1=LogRegExp([x,y],bs,n_paths=20)

#HMC to get mean
# Tp=torch.tensor(torch.pi/2)
# hpV=Tp/3
# samples=getHMC(Exp1,hpV,Tp,Nsamples_HMC) 
# truemean=samples['samples'].mean(axis=0)

# exp1_file = open("LogReg_SimDataSmallHMC.pkl", 'wb')
# pickle.dump(samples,exp1_file)
# exp1_file.close()

# with open("LogReg_SimDataSmallHMCtruemean.pkl", 'wb') as f:
#     pickle.dump(truemean,f)
    
with open("LogReg_SimDataSmallHMCtruemean.pkl", 'rb') as f:
    truemean=pickle.load(f).detach()

etarange = 2.**torch.arange(2,9)*1.1/K
Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)
# Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)

strats=['RR','1SS','FULLGRAD']
sgld_dict={s:{} for s in strats}
sgld_dict['K']=K
sgld_dict['etarange']=etarange
for strat in strats:
    stoch=True
    if strat=='FULLGRAD':
        stoch=False
        Exp1=LogRegExp([x,y],bs,n_paths=20)
    for i,timestep in enumerate(etarange):
        samples=getLD(Exp1,timestep,Nsamples[i],strat=strat,pcond=True,stoch=stoch) 
        s=samples['samples']
        sgld_dict[strat][str(timestep.item())]=samples

# exp1_file = open(f"LogRegSimSmall_SGLDK{K}.pkl", 'wb')
# pickle.dump(sgld_dict,exp1_file)
# exp1_file.close()

with open(f"LogRegSimSmall_SGLDK{K}.pkl", 'rb') as f:
    sgld_dict=pickle.load(f)

##Plotting
err={s:[] for s in strats}
# K=sgld_dict['K']
for strat in strats:
    loc=sgld_dict[strat]
    for i,timestep in enumerate(loc.keys()):
        samples=loc[timestep]
        s=samples['samples']
        e=(s.mean(dim=0)-truemean).mean(dim=0) #shape (n_paths,n_features)
        e1=(s.mean(dim=1)-truemean)[-10*K:]
        e1=torch.linalg.norm(e1/truemean,dim=1) #shape (n_iters,n_features)
        h=samples['h']
        #Plot oscillations
        if i==(len(loc.keys())-2):
            plt.figure(figsize=(3,2))
            num=round(h.item(),1)
            stratlab='RM' if strat=='1SS' else strat
            plt.title('LogReg SimData: ' + 'SGLD-'+stratlab+', $h='+f'{num}'+'$')
            
            plt.semilogy(np.arange(len(e1))/K,
                                  torch.abs(e1),'k',ls='-' ,base=2)
            plt.xlabel('Iteration over dataset')
            plt.ylabel('$\|\Delta\mu/\mu\|$')
            plt.savefig(os.path.join(figdir,f'LRSimDataK{K}_Oscillations{strat}.pdf'),format='pdf',bbox_inches='tight')
            if strat=='1SS':
                plt.yticks([2**.5,2**.75],['$2^{0.5}$','$2^{0.75}$'])
            else:
                plt.yticks([2**.25,2**.5],['$2^{0.25}$','$2^{0.5}$'])
        err[strat]+=[torch.linalg.norm(e)/torch.linalg.norm(truemean)]

# markerlist=['s','X','o']
markerlist=['','','']

plt.figure(figsize=(3,2))
plt.loglog(etarange,err['1SS'],'b-',base=2,label='SGLD-RM',marker=markerlist[1])
plt.loglog(etarange,err['RR'],'r-',base=2,label='SGLD-RR',marker=markerlist[0])
plt.loglog(etarange,err['FULLGRAD'],'k-',base=2,label='ULA',marker=markerlist[2])
# plt.loglog(etarange,etarange,'k--',base=2,alpha=0.7)
# plt.loglog(etarange,(etarange)**2,'k--',base=2,alpha=0.7)
# # plt.loglog(etarange,(etarange)**1.5,'k--',base=2,alpha=0.7)

plt.title(f'LogReg SimData, $R={K}$')
plt.xlabel('$h$')
plt.ylabel('$\|\Delta\mu/\mu\|$')
plt.legend()
plt.savefig(os.path.join(figdir,f'LogRegSimDataSmallK{K}.pdf'),format='pdf',bbox_inches='tight')
#%% LogReg Experiment 2: Chess
Nsamples_HMC=10**8
np.random.seed(2022)
data = pd.read_table('chess.txt', sep=",", header=None)
y = np.array(data.iloc[:,-1]=='won',dtype=np.float64)
X = data.iloc[:,:-1]
x = np.zeros_like(X,dtype=np.float64)
for i in range(x.shape[-1]): 
    x[:,i] = pd.factorize(X.iloc[:,i],sort=True)[0]
x,y=torch.tensor(x),torch.tensor(y)
K=16
N=len(x)
bs=N//K

Exp1=LogRegExp([x,y],bs,n_paths=20)

# #HMC to get mean
# Tp=torch.tensor(torch.pi/2)
# hpV=Tp/2
# samples=getHMC(Exp1,hpV,Tp,Nsamples_HMC) 
# truemean=samples['samples'].mean(axis=0)

# with open("LogReg_ChessHMC.pkl", 'wb') as f:
#     pickle.dump(samples,f)

# with open("LogReg_ChessHMCtruemean.pkl", 'wb') as f:
#     pickle.dump(truemean,f)

# with open("LogReg_ChessHMC.pkl", 'rb') as f:
#     samples=pickle.load(f)


strats=['RR','1SS','FULLGRAD']
err={s:[] for s in strats}
sgld_dict={s:{} for s in strats}
with open("LogReg_ChessHMCtruemean.pkl", 'rb') as f:
    truemean=pickle.load(f)
    
# with open(f"LogRegChess_SGLDK{K}.pkl", 'rb') as f:
#     sgld_dict=pickle.load(f)

etarange = 2.**torch.arange(0,6,1)/K
Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)

for strat in strats:
    for i,timestep in enumerate(etarange):
        stoch =False if strat=='FULLGRAD' else True
        samples=getLD(Exp1,timestep,Nsamples[i],strat=strat,pcond=True,stoch=stoch) 
        # samples=sgld_dict[strat][str(timestep)]
        s=samples['samples']
        e=(s.mean(dim=0)-truemean).mean(dim=0) #shape (n_paths,n_features)
        sgld_dict[strat][str(timestep)]=samples
        err[strat]+=[torch.linalg.norm(e)/torch.linalg.norm(truemean)]

# exp1_file = open(f"LogRegChess_SGLDK{K}.pkl", 'wb')
# pickle.dump(sgld_dict,exp1_file)
# exp1_file.close()

markerlist=['s','X','o']
plt.loglog(etarange,err['RR'],'k-',base=2,label='RR',marker=markerlist[0])
plt.loglog(etarange,err['1SS'],'k-',base=2,label='1SS',marker=markerlist[1])
plt.loglog(etarange,err['FULLGRAD'],'k-',base=2,label='FULLGRAD',marker=markerlist[2])
plt.loglog(etarange,etarange,'k--',base=2,alpha=0.7)
plt.loglog(etarange,(etarange)**2,'k--',base=2,alpha=0.7)
# plt.loglog(etarange,(etarange)**1.5,'k--',base=2,alpha=0.7)

l=len(etarange)
c=(etarange[1]+etarange[2])/2
plt.text(c,1.3*c,'$h$')
plt.text(c,1.3*c**2,'$h^2$')
# plt.text(c,1.3*c**1.5,'$h^{3/2}$')
plt.title(f'LogReg_Chess, $K={K}$')
plt.xlabel('$h$')
plt.ylabel('$\\frac{\|\Delta\mu\|}{\|\mu\|}$')
plt.legend()
plt.savefig(f'LogRegChessK{K}.pdf',format='pdf',bbox_inches='tight')
plt.show()

#%% LogReg Experiment 3: CTG
np.random.seed(2022)
Nsamples_HMC=10**6
ctg = pd.read_table('CTG.txt',header=0)
X = np.array(ctg.iloc[:,:21])
x = StandardScaler().fit_transform(X)
n=x.shape[0]
y = np.array(ctg.iloc[:,-1])
y=np.where(y>2,1,0)
x,y=torch.tensor(x),torch.tensor(y)
K=16
N=len(x)
bs=N//K

Exp1=LogRegExp([x,y],bs,n_paths=1000)

# #HMC to get mean
# Tp=torch.tensor(torch.pi/2)
# hpV=Tp/2
# samples=getHMC(Exp1,hpV,Tp,Nsamples_HMC) 
# truemean=samples['samples'].mean(axis=0)

# with open("LogReg_CTGHMC.pkl", 'wb') as f:
#     pickle.dump(samples,f)

# with open("LogReg_CTGHMCtruemean.pkl", 'wb') as f:
#     pickle.dump(truemean,f)

# with open("LogReg_CTGHMC.pkl", 'rb') as f:
#     samples=pickle.load(f)


strats=['RR','1SS','FULLGRAD']
sgld_dict={s:{} for s in strats}
sgld_dict['K']=K
etarange = 2.**torch.arange(-1,6,1)/K
Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)

# for strat in strats:
#     for i,timestep in enumerate(etarange):
#         stoch =False if strat=='FULLGRAD' else True
#         samples=getLD(Exp1,timestep,Nsamples[i],strat=strat,pcond=True,stoch=stoch) 
#         s=samples['samples']
#         sgld_dict[strat][str(timestep)]=samples

# exp1_file = open(f"LogRegCTG_SGLDK{K}.pkl", 'wb')
# pickle.dump(sgld_dict,exp1_file)
# exp1_file.close()


with open("LogReg_CTGHMCtruemean.pkl", 'rb') as f:
    truemean=pickle.load(f)
    
with open(f"LogRegCTG_SGLDK{K}.pkl", 'rb') as f:
    sgld_dict=pickle.load(f)

##Plotting
err={s:[] for s in strats}
RRplot=plt.figure()
# K=sgld_dict['K']
for strat in sgld_dict.keys():
    loc=sgld_dict[strat]
    tsteps=list(loc.keys())
    for i,timestep in enumerate(tsteps):
        stoch =False if strat=='FULLGRAD' else True
        samples=loc[timestep]
        s=samples['samples']
        e=(s.mean(dim=0)-truemean).mean(dim=0) #shape (n_paths,n_features)
        e1=(s.mean(dim=1)-truemean)
        e1=torch.linalg.norm(e1,dim=1)/torch.linalg.norm(truemean) #shape (n_iters,n_features)
        h=samples['h']
        #Plot oscillations
        plt.figure()
        num=round(np.log2(h.item()*K),1)
        plotter=e1[-10*K:,0]
        plt.semilogy(np.arange(len(plotter))/K,
                              torch.abs(plotter),'k',ls='--' ,
                              label='$hK=2^{'+f'{num}'+'}$',base=2)
   
        err[strat]+=[torch.linalg.norm(e)/torch.linalg.norm(truemean)]
plt.figure(RRplot)
fig2title=f'Relative Variance Error, Asymptotic, RR $K={K}$'
plt.title(fig2title)
plt.xlabel('Iteration over dataset')
plt.legend()
plt.show()

markerlist=['s','X','o']
plt.loglog(etarange,err['RR'],'k-',base=2,label='RR',marker=markerlist[0])
plt.loglog(etarange,err['1SS'],'k-',base=2,label='1SS',marker=markerlist[1])
plt.loglog(etarange,err['FULLGRAD'],'k-',base=2,label='FULLGRAD',marker=markerlist[2])
plt.loglog(etarange,etarange,'k--',base=2,alpha=0.7)
plt.loglog(etarange,(etarange)**2,'k--',base=2,alpha=0.7)
# plt.loglog(etarange,(etarange)**1.5,'k--',base=2,alpha=0.7)

l=len(etarange)
c=(etarange[1]+etarange[2])/2
plt.text(c,1.3*c,'$h$')
plt.text(c,1.3*c**2,'$h^2$')
# plt.text(c,1.3*c**1.5,'$h^{3/2}$')
plt.title(f'LogReg_CTG, $K={K}$')
plt.xlabel('$h$')
plt.ylabel('$\\frac{\|\Delta\mu\|}{\|\mu\|}$')
plt.legend()
# plt.savefig(f'LogRegCTGK{K}.pdf',format='pdf',bbox_inches='tight')
plt.show()


#%% LogReg Experiment 4: StatLog
np.random.seed(2022)
Nsamples_HMC=10**6
data = pd.read_table('satTrn.txt', header=None, sep=' ')
X = np.array(data.iloc[:,:-1])
x = StandardScaler().fit_transform(X)
n=x.shape[0]
y = np.array(data.iloc[:,-1])
y=np.where(y==2,1,0)
x,y=torch.tensor(x),torch.tensor(y)
K=16
N=len(x)
bs=N//K
Exp1=LogRegExp([x,y],bs,n_paths=1000)

# #HMC to get mean
# Tp=torch.tensor(torch.pi/2)
# hpV=Tp/3
# samples=getHMC(Exp1,hpV,Tp,Nsamples_HMC) 
# truemean=samples['samples'].mean(axis=0)

# with open("LogReg_StatLogHMC.pkl", 'wb') as f:
#     pickle.dump(samples,f)

# with open("LogReg_StatLogHMCtruemean.pkl", 'wb') as f:
#     pickle.dump(truemean,f)

# with open("LogReg_StatLogHMC.pkl", 'rb') as f:
#     samples=pickle.load(f)


strats=['RR','1SS','FULLGRAD']
err={s:[] for s in strats}
sgld_dict={s:{} for s in strats}

with open("LogReg_StatLogHMCtruemean.pkl", 'rb') as f:
    truemean=pickle.load(f)
    
# with open(f"LogRegStatLog_SGLDK{K}.pkl", 'rb') as f:
#     sgld_dict=pickle.load(f)

etarange = 2.**torch.arange(0,6,1)/K
Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)

for strat in strats:
    for i,timestep in enumerate(etarange):
        stoch =False if strat=='FULLGRAD' else True
        samples=getLD(Exp1,timestep,Nsamples[i],strat=strat,pcond=True,stoch=stoch) 
        # samples=sgld_dict[strat][str(timestep)]
        s=samples['samples']
        e=(s.mean(dim=0)-truemean).mean(dim=0) #shape (n_paths,n_features)
        sgld_dict[strat][str(timestep)]=samples
        err[strat]+=[torch.linalg.norm(e)/torch.linalg.norm(truemean)]

exp1_file = open(f"LogRegStatLog_SGLDK{K}.pkl", 'wb')
pickle.dump(sgld_dict,exp1_file)
exp1_file.close()

markerlist=['s','X','o']
plt.loglog(etarange,err['RR'],'k-',base=2,label='RR',marker=markerlist[0])
plt.loglog(etarange,err['1SS'],'k-',base=2,label='1SS',marker=markerlist[1])
plt.loglog(etarange,err['FULLGRAD'],'k-',base=2,label='FULLGRAD',marker=markerlist[2])
plt.loglog(etarange,etarange,'k--',base=2,alpha=0.7)
plt.loglog(etarange,(etarange)**2,'k--',base=2,alpha=0.7)
# plt.loglog(etarange,(etarange)**1.5,'k--',base=2,alpha=0.7)

l=len(etarange)
c=(etarange[1]+etarange[2])/2
plt.text(c,1.3*c,'$h$')
plt.text(c,1.3*c**2,'$h^2$')
# plt.text(c,1.3*c**1.5,'$h^{3/2}$')
plt.title(f'LogReg_StatLog, $K={K}$')
plt.xlabel('$h$')
plt.ylabel('$\\frac{\|\Delta\mu\|}{\|\mu\|}$')
plt.legend()
plt.savefig(f'LogRegStatLogK{K}.pdf',format='pdf',bbox_inches='tight')
plt.show()