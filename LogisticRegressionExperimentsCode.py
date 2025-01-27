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
            print('RM selected')
            self.sample=self.RMsampler
            
    def RRsampler(self):
        if self.index==0:
            self.datasource=self.data[torch.argsort(torch.rand(size=(self.n_paths,self.length)), dim=-1)]
        k,bs=self.index,self.bs
        self.index=(self.index+1)%self.K 
        data=self.datasource[:,k*bs:(k+1)*bs]
        return data

    def RMsampler(self):
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
            self.mybatcher.set_strat('RM')

    
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
    def __init__(self,data,bs,n_paths,strat='RM'):
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
    def __init__(self,x,bs,n_paths,strat='RM'):
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


def plotter(expname,K):
    with open(f"LogReg{expname}_SGLDK{K}.pkl", 'rb') as f:
        sgld_dict=pickle.load(f)
    
    ##Plotting
    err={s:[] for s in ['RR','RM','FULLGRAD']}
    K=sgld_dict['K']
    etarange=sgld_dict['etarange']
    # try:
    #     with open(f"LogReg_{expname}K{K}_err.pkl", 'rb') as f:
    #         err=pickle.load(f)
    # except:

    with open(f"LogReg_{expname}HMCtruemean.pkl", 'rb') as f:
        truemean=pickle.load(f).detach()
    
    n=len(etarange)
    for strat in err.keys():
        loc=sgld_dict[strat]
        for i,timestep in enumerate(loc.keys()):
            samples=loc[timestep]
            s=samples['samples']
            e=(s.mean(dim=0)-truemean).mean(dim=0) #shape (n_paths,n_features)
            e1=(s.mean(dim=1)-truemean)[-10*K:]
            e1=torch.linalg.norm(e1,dim=1)/torch.linalg.norm(truemean)#shape (n_iters,n_features)
            h=samples['h']
            #Plot oscillations
            if i==2:
                plt.figure(figsize=(3,2))
                num=round(h.item(),1)
                stratlab=strat #'RM' if strat=='1SS' else strat
                plt.title(f'LogReg {expname}: ' + 'SGLD-'+stratlab+', $h='+f'{num}'+'$')
                
                plt.semilogy(np.arange(len(e1))/K,
                                      torch.abs(e1),'k',ls='-' ,base=2)
                plt.xlabel('Iteration over dataset')
                plt.ylabel('$\|\Delta\mu\|/\|\mu\|$')
                if strat=='1SS':
                    plt.yticks([2**-1.4,2**-1.25],['$2^{-1.4}$','$2^{-1.25}$'])
                else:
                    plt.yticks([2**-1.5,2**-1.4],['$2^{-1.5}$','$2^{-1.4}$'])

                plt.savefig(os.path.join(figdir,f'LogReg{expname}K{K}_Oscillations{strat}.pdf'),format='pdf',bbox_inches='tight')
            err[strat]+=[torch.linalg.norm(e)/torch.linalg.norm(truemean)]

    with open(f"LogReg_{expname}K{K}_err.pkl", 'wb') as f:
        pickle.dump(err,f)
            
    # markerlist=['s','X','o']
    markerlist=['','','']
    
    plt.figure(figsize=(3,2))
    plt.loglog(etarange,err['RM'],'b-',base=2,label='SGLD-RM',marker=markerlist[1])
    plt.loglog(etarange,err['RR'],'r-',base=2,label='SGLD-RR',marker=markerlist[0])
    plt.loglog(etarange[-4:],err['FULLGRAD'][-4:],'k-',base=2,label='ULA',marker=markerlist[2])
    plt.loglog(etarange,etarange,'k--',base=2,alpha=0.7)
    plt.loglog(etarange,(etarange)**2,'k--',base=2,alpha=0.7)
    plt.ylim([2**-6.9,2**1])
    
    plt.title(f'LogReg {expname}, $R={K}$')
    plt.xlabel('$h$')
    plt.ylabel('$\|\Delta\mu\|/\|\mu\|$')
    plt.legend()
    plt.savefig(os.path.join(figdir,f'LogReg{expname}K{K}.pdf'),format='pdf',bbox_inches='tight')

def runLRExp(expname,K,n_paths=10**4):
    Nsamples_HMC=10**7
    
    if expname=='Chess':
        data = pd.read_table('chess.txt', sep=",", header=None)
        y = np.array(data.iloc[:,-1]=='won',dtype=np.float64)
        X = data.iloc[:,:-1]
        x = np.zeros_like(X,dtype=np.float64)
        for i in range(x.shape[-1]): 
            x[:,i] = pd.factorize(X.iloc[:,i],sort=True)[0]
        x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='StatLog':
        data = pd.read_table('satTrn.txt', header=None, sep=' ')
        X = np.array(data.iloc[:,:-1])
        x = StandardScaler().fit_transform(X)
        y = np.array(data.iloc[:,-1])
        y=np.where(y==2,1,0)
        x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='CTG':
        ctg = pd.read_table('CTG.txt',header=0)
        X = np.array(ctg.iloc[:,:21])
        x = StandardScaler().fit_transform(X)
        y = np.array(ctg.iloc[:,-1])
        y=np.where(y>2,1,0)
        x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='SimData':
        try:
            with open("SimData.pkl", 'rb') as f:
                d=pickle.load(f)
                x=torch.tensor(d['x'])
                y=torch.tensor(d['y'])
        except:
            print('Generating simulated data for log reg experiment.')
            np.random.seed(2024)
            d=25
            p=d+1
            N=2**10
            scaler=np.hstack((5*np.ones(shape=(1,5)),np.ones(shape=(1,5)),.2*np.ones(shape=(1,d-10))))
            params=np.random.normal(size=(p,))
            x=np.random.normal(size=(N,d),scale=scaler) #input data
            xnew=np.hstack((np.ones(shape=(N,1)),x))
            p_i=expit(torch.tensor(xnew@params))
            y=np.random.binomial(1, p_i).flatten() # output data
            with open("SimData.pkl", 'wb') as f:
                pickle.dump({'x':x,'y':y,'params':params},f)
    else:
        raise ValueError('expname not valid: choose one of StatLog,Chess,CTG,SimData.')
    
    N=len(x)
    bs=N//K
    Exp1=LogRegExp([x,y],bs,n_paths=n_paths)
    try:
        with open(f"LogReg_{expname}HMCtruemean.pkl", 'rb') as f:
            truemean=pickle.load(f).detach()
    except:
        print(f'Running HMC sampler to get true mean with {Nsamples_HMC} samples.')
        #HMC to get mean
        Tp=torch.tensor(torch.pi/2)
        hpV=Tp/3
        samples=getHMC(Exp1,hpV,Tp,Nsamples_HMC) 
        truemean=samples['samples'].mean(axis=0)

        with open(f"LogReg_{expname}HMCtruemean.pkl", 'wb') as f:
            pickle.dump(truemean,f)

    etarange = 2.**torch.arange(-5,1)
    Nsamples=torch.tensor(np.minimum(10000+(1000/(etarange.numpy())**2),10**6)).to(torch.int32)

    strats=['RR','RM','FULLGRAD']
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
            sgld_dict[strat][str(timestep.item())]=samples

    with open(f"LogReg{expname}_SGLDK{K}.pkl", 'wb') as f:
        pickle.dump(sgld_dict,f)
