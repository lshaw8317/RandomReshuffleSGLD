import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.linalg import solve_triangular, cholesky, cho_solve
import os
from scipy.special import expit as npexpit
import matplotlib.pyplot as plt
import pickle
plt.rcParams.update({'text.usetex':False,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.style.use('seaborn-v0_8-paper')
import time as time
figdir='figs'
resultsdir='RESULTS'
datadir='data'
#%%
class MyBatcher:
    def __init__(self,data,K,n_paths,strat=None):
        self.data=data
        self.length = len(data)
        # shape=tuple([n_paths]+[1 for i in data.shape])
        # self.datasource = data[None,...].repeat(shape)

        self.datasource = data[None,...].repeat(repeats=n_paths,axis=0)
        self.K=min(K,self.length)
        print(f'Set K to {self.K}')
        self.bs=int(self.length/K) + 1*(self.length%K!=0)
        self.index=0
        self.n_paths=n_paths
        self.strat=None
        self.sample = self.NoSampler
    
    def redraw(self):
        d=self.data[np.argsort(np.random.rand(*(self.n_paths,self.length)), axis=-1)]
        if self.strat=='SMS':
            self.datasource=np.concatenate((d,np.flip(d,axis=(1,))),axis=1)
        else:
            self.datasource=d

    def set_strat(self,strat):
        self.index=0
        if strat=='RR':
            print('RR selected')
            self.strat='RR'
            self.sample=self.RRsampler
        elif strat=='SMS':
            print('SMS selected')
            self.strat='SMS'
            self.sample=self.SMSsampler
        elif strat=='SO':
            print('SO selected')
            self.strat='SO'
            self.redraw()
            self.sample=self.SOsampler
        else:
            print('RM selected')
            self.strat='RM'
            self.sample=self.RMsampler
            
    def RRsampler(self):
        if self.index==0:
            self.redraw()
        data=self.datasource[:,self.index*self.bs:(self.index+1)*self.bs]
        self.index=(self.index+1)%self.K 
        return data
    
    def SOsampler(self):
        data=self.datasource[:,self.index*self.bs:(self.index+1)*self.bs]
        self.index=(self.index+1)%self.K 
        return data
    
    def SMSsampler(self):
        if self.index==0:
            self.redraw()
            idx=self.bs if self.length%self.bs==0 else self.length%self.bs
            data=self.datasource[:,:idx]
            self.datasource=self.datasource[:,idx:]
        else:
            data=self.datasource[:,(self.index-1)*self.bs:self.index*self.bs]
        self.index=(self.index+1)%(2*self.K)
        return data

    def RMsampler(self):
        if self.index==0:
            self.redraw()
        k_=np.random.randint(low=0,high=self.length)
        self.index=(self.index+1)%self.K
        inds=np.arange(k_,k_+self.bs)%self.length
        data=self.datasource[:,inds]
        return data
    
    def NoSampler(self):
        raise Exception('Sampling strategy not been defined!')

class Loss:
    def __init__(self,data,K,n_paths,strat='RM', Cinv=None):
        self.x, self.y = data
        self.n = int(self.x.shape[0])
        # Add dummy for bias
        self.xnew = np.concatenate((self.x, np.ones((self.n, 1))), axis=1)
        if not Cinv:
            self.Cinv = 0
            L = self.smoothness()
            self.Cinv = L / np.sqrt(self.n)
        else:
            self.Cinv = np.asarray(Cinv)
        if self.Cinv.shape != ():
            raise ValueError('Cinv can only be scalar!')
        self.MAP = self.calc_MAP()
        data_comb = np.concatenate((self.xnew, self.y[..., None]), axis=-1)
        self.mybatcher = MyBatcher(data=data_comb, K=K, n_paths=n_paths, strat=strat)
    
    def set_strat(self,strat):
        if strat=='RR':
            self.mybatcher.set_strat('RR')
        elif strat=='SMS':
            self.mybatcher.set_strat('SMS')
        else:
            self.mybatcher.set_strat('RM')

    def NLogLoss(self,q):
        return
    
    def grad(self,q,data):
        return
    
    def fullgradient(self,q):
        return
    
    def stochgrad(self,q):
        data=self.mybatcher.sample()
        scaler=data.shape[1]/self.mybatcher.bs
        return self.grad(q,data)*scaler
    
    def calc_MAP(self,epochs):
        return
    
    def smoothness(self):
        return
        
class Sampler:
    def __init__(self, loss, method='SGLD',strat='RM'):
        self.loss = loss
        self.loss.set_strat(strat)
        self.strat=strat
        self.method=method.lower()
        if self.method=='hmc':
            self.stepper=self.HMC
            self.HMCsteps = 2
            self.ham = lambda q,v:.5*np.sum(v*(self.loss.J[None,...]@v))+self.loss.U(q)
        elif self.method=='sgld':
            self.stepper=self.SGLD
        else:
            raise ValueError('method arg to Sampler class not recognised: sgld and hmc are only available methods.')
            
    def run(self, q0, h0, Niters):
        global acc
        acc = 0
        q=np.float64(q0.copy())
        v=np.zeros_like(q)
        epochs=Niters//self.loss.mybatcher.K
        epochs+=1*epochs%2 #need number of epochs to be even for SMS
        Niters=epochs*self.loss.mybatcher.K
        samples=np.zeros((Niters,*q.shape))
        h=h0
        for n in range(0,Niters):
            q,v=self.stepper(q,v,h)
            samples[n]=q
        print(f'Acceptance rate = {round(100 * acc / Niters, 2)}')
        return samples

    def SGLD(self, qp, v, h): 
        eta = np.sqrt(2*h)*solve_triangular(self.loss.Jchol[None,...], np.random.randn(*qp.shape), lower=False)
        grad = self.loss.stochgrad(qp)
        update=cho_solve((self.loss.Jchol[None,...], False), grad)
        qp = qp - h*update + eta
        return qp, v
    
    def HMC(self, q, v, h): 
        global acc
        h=(1.-0.2*np.random.rand(1))*h
        v=solve_triangular(self.loss.Jchol[None,...], np.random.randn(*q.shape), lower=False)#Draw v ~ N(0,Jinv)
        qp = q
        H0=self.ham(qp,v)
        Nsteps = self.HMCsteps
        
        #Do a leg of T//h steps of Strang
        #(b1) Kick
        theta1=h/2
        v-=theta1*cho_solve((self.loss.Jchol[None, ...], False), self.loss.fullgradient(qp))
        for t in range(Nsteps):
            qp+=h*v #Drift
            theta = 2*theta1 if (t!=Nsteps-1) else theta1
            v-=theta*cho_solve((self.loss.Jchol[None,...], False), self.loss.fullgradient(qp))

        accept=H0-self.ham(qp,v) #acceptance probability
        #Accept/reject
        if (accept>np.log(np.random.rand(1))):
            q=qp 
            acc += 1
        return q, v
    
class LogReg(Loss):
    def __init__(self,data,K,n_paths):
        super().__init__(data, K, n_paths, Cinv=1./25.)
        arg=self.xnew@self.MAP
        J=self.xnew.T*(npexpit(arg)*npexpit(-arg))@self.xnew
        J += self.Cinv
        Jchol = cholesky(J, lower=False)
        self.J = J
        self.Jchol = Jchol
    
    def smoothness(self):
        covariance = self.xnew.T@self.xnew/self.n
        return 0.25*np.max(np.linalg.eigvalsh(covariance)) + self.Cinv
    
    def U(self,q):
        arg=self.xnew@q
        ans=-np.sum(self.y[None,...,None]*arg)
        ans+=np.sum(np.logaddexp(np.zeros_like(arg),arg))
        term=q*self.Cinv*q 
        return .5*np.sum(term)+ans/self.n
    
    def calc_MAP(self):
        x=np.random.randn(1, *self.xnew.shape[1:], 1)*.2 # (n_paths, n_features, 1)
        lr=1/self.smoothness()
        kappa = (1/lr)/self.Cinv
        momentum = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)
        x_nest = x.copy()
        history=[x]
        nepochs=600
        for i in range(nepochs):
            x_nest_old = x_nest.copy()
            g = self.fullgradient(x)
            x_nest = x - lr*g
            x = x_nest + momentum*(x_nest-x_nest_old)
            history+=[x]
        history=np.array(history).squeeze()
        err=np.linalg.norm(history[:-1]-history[-1],axis=-1)
        plt.semilogy(np.arange(len(err)),err)
        plt.xlabel('Iterations')
        plt.ylabel('$\|x-x_*\|$')
        plt.title('Correctly found minimum with fullgrad Nesterov')
        plt.savefig('temp.png', format='png')
        return x.squeeze()
    
    def fullgradient(self,q): ## np version
        term=q*self.Cinv
        arg=np.matmul(self.xnew[None,...],q)
        temp=(self.y[...,None]-npexpit(arg))
        return term-np.matmul(self.xnew[None,...].transpose(0,2,1),temp)/self.n
    
    def grad(self, q, data):
       x,y=data[...,:-1],data[...,-1] #x has shape (n_paths, n, n_features)
       term=q*self.Cinv #q has shape (n_paths,n_features,1)
       arg=np.matmul(x,q) #has shape (n_paths,n,1)
       temp=y[...,None]-npexpit(arg) #has shape (n_paths,n,1)
       bs=x.shape[1] #self.n divide term/self.mybatcher.K for true splitting scheme
       return term-np.matmul(x.transpose(0,2,1),temp)/bs

class GaussianExp(Loss):
    def __init__(self,x,K,n_paths,strat='RM'):
        super().__init__((x, None), K, n_paths, Cinv=0.)
        self.truemean=self.MAP
        self.truecov=np.cov(self.x.T).reshape(self.x.shape[1],self.x.shape[1]) #x has shape (n,features)
        self.J = np.linalg.inv(self.truecov)
        self.Jchol = cholesky(self.J, lower=False)
        
    def calc_MAP(self, epochs):
        return self.x.mean(dim=0)
    
    def U(self,q):
        arg=np.matmul(self.Jchol[None,...], (q-self.truemean[None,...]))
        ans=np.sum(arg*arg)
        return .5*ans
    
    def grad(self, q, data):
        x = data[...,:-1] #x has shape (n_paths,n,n_features)
        return np.matmul(self.J[None,...], (q-x.mean(axis=1, keepdims=False)[None,...]))
    
    def fullgradient(self, q):
        return np.matmul(self.J[None,...], (q-self.truemean[None,...]))

def getprogress(opt, h, Niters):
    q0 = opt.loss.MAP[None,...,None].repeat(repeats=opt.loss.mybatcher.n_paths,axis=0)
    s=opt.run(q0, h, Niters=Niters)
    return s


def get_loss(expname,K,n_paths=100,exp='LogReg'):

    if expname=='Chess':
        data = pd.read_table(datadir+'/chess.txt', sep=",", header=None)
        y = np.array(data.iloc[:,-1]=='won',dtype=np.float64)
        X = data.iloc[:,:-1]
        x = np.zeros_like(X,dtype=np.float64)
        for i in range(x.shape[-1]): 
            x[:,i] = pd.factorize(X.iloc[:,i],sort=True)[0]
    elif expname=='StatLog':
        data = pd.read_table(datadir+'/satTrn.txt', header=None, sep=' ')
        X = np.array(data.iloc[:,:-1])
        x = StandardScaler().fit_transform(X)
        y = np.array(data.iloc[:,-1])
        y=np.where(y==2,1,0)
    elif expname=='CTG':
        ctg = pd.read_table(datadir+'/CTG.txt',header=0)
        X = np.array(ctg.iloc[:,:21])
        x = StandardScaler().fit_transform(X)
        y = np.array(ctg.iloc[:,-1])
        y=np.where(y>2,1,0)
    elif expname=='SimData':
        try:
            with open(datadir+"/SimData.pkl", 'rb') as f:
                d=pickle.load(f)
                x=d['x']
                y=d['y']
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
            p_i=npexpit((xnew@params))
            y=np.random.binomial(1, p_i).flatten() # output data
            with open("SimData.pkl", 'wb') as f:
                pickle.dump({'x':x,'y':y,'params':params},f)
    elif expname=='SimpleData':
        try:
            with open(datadir+"/SimpleData.pkl", 'rb') as f:
                d=pickle.load(f)
                x=d['x']
                y=d['y']
        except:
            print('Generating simulated data for lin reg experiment.')
            np.random.seed(2024)
            # True parameters
            w_true = 2.0
            b_true = 0.1
            
            # Generate noisy dataset
            x = np.array([1, 2, 3, 4, 5], dtype=np.float64)[...,None]
            y = (w_true * x + b_true).flatten() + 0.2*np.random.randn(len(x))
    
            with open(datadir+"/SimpleData.pkl", 'wb') as f:
                pickle.dump({'x':x,'y':y,'params':[b_true,w_true]},f)

    else:
        raise ValueError('expname not valid: choose one of StatLog,Chess,CTG,SimData.')
    
    N=len(x)
    if exp=='LogReg':
        loss=LogReg([x,y],K,n_paths=n_paths)
    else:
        raise ValueError('Exp type not recognised.')
    return loss


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
            e1=np.linalg.norm(e1, axis=1) / np.linalg.norm(truemean)#shape (n_iters,n_features)
            h=np.float64(timestep)
            #Plot oscillations
            if i==2:
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

                plt.savefig(os.path.join(figdir,f'LogReg{expname}K{K}_Oscillations{strat}.pdf'),format='pdf',bbox_inches='tight')
            err[strat]+=[np.linalg.norm(e) / np.linalg.norm(truemean)]

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
    plt.savefig(os.path.join(figdir,f'LogReg{expname}K{K}.png'),format='png',bbox_inches='tight')

def runLRExp(expname,K,n_paths=10**4):
    Nsamples_HMC = 10**4
    loss = get_loss(expname, K, n_paths=n_paths, exp='LogReg')
    try:
        with open(f"LogReg_{expname}HMCtruemean.pkl", 'rb') as f:
            truemean=pickle.load(f).detach()
    except:
        print(f'Running HMC sampler to get true mean with {Nsamples_HMC} samples.')
        #HMC to get mean
        hpV = np.pi/4
        sampler = Sampler(loss, method='hmc',strat='RM')
        n_paths_orig = loss.mybatcher.n_paths
        sampler.loss.mybatcher.n_paths = 1
        samples = getprogress(sampler, hpV, Nsamples_HMC)
        sampler.loss.mybatcher.n_paths = n_paths_orig
        truemean = samples.mean(axis=0)

        with open(f"LogReg_{expname}HMCtruemean.pkl", 'wb') as f:
            pickle.dump(truemean,f)

    etarange = 2.**np.arange(-5,1)
    Nsamples=np.int64(np.minimum(10000+(1000/(etarange)**2), 10**3))

    strats=['RR','RM','FULLGRAD']
    sgld_dict={s:{} for s in strats}
    sgld_dict['K'] = K
    sgld_dict['etarange']=etarange
    for strat in strats:
        if strat=='FULLGRAD':
            new_loss = get_loss(expname, 1, n_paths=20, exp='LogReg') # set K = 1
            sampler = Sampler(new_loss, method='sgld')
        for i,timestep in enumerate(etarange):
            sampler = Sampler(loss, method='sgld',strat=strat)
            samples = getprogress(sampler, timestep, Nsamples[i])
            sgld_dict[strat][str(timestep.item())] = samples

    with open(f"LogReg{expname}_SGLDK{K}.pkl", 'wb') as f:
        pickle.dump(sgld_dict,f)

#%%
runLRExp('CTG', 16, 20)
# runLRExp('StatLog', 16, 20)
# runLRExp('Chess', 16, 20)
# runLRExp('SimData', 16, 20)

plotter('CTG', 16)
# plotter('StatLog', 16)
# plotter('Chess', 16)
# plotter('SimData', 16)