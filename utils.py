import numpy as np
from scipy.linalg import solve_triangular, cholesky, cho_solve
from scipy.special import expit as npexpit
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MyBatcher:
    def __init__(self,data,K,n_paths,strat=None):
        self.data=data
        self.length = len(data)

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

    def set_strat(self, strat):
        self.index=0
        if strat=='RR':
            print('RR selected')
            self.strat='RR'
            self.sample=self.RRsampler
        elif strat=='SMS':
            print('SMS selected')
            self.strat='SMS'
            self.sample=self.SMSsampler
        elif strat=='FG':
            print('FG selected')
            self.strat='FG'
            self.sample=self.FGsampler
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
    
    def FGsampler(self):
        return self.datasource
    
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
    def __init__(self, data, K, n_paths, expname, strat='RM', Cinv=None):
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
        data_comb = np.concatenate((self.xnew, self.y[..., None]), axis=-1)
        self.mybatcher = MyBatcher(data=data_comb, K=K, n_paths=n_paths, strat=strat)
        self.expname = expname
    
    def set_strat(self, strat):
        if strat=='RR':
            self.mybatcher.set_strat('RR')
        elif strat=='SMS':
            self.mybatcher.set_strat('SMS')
        elif strat=='FG':
            self.mybatcher.set_strat('FG')
        else:
            self.mybatcher.set_strat('RM')
    
    def U(self, q):
        return 
    
    def grad(self,q,data):
        return
    
    def smoothness(self):
        return
        
    def calc_MAP(self):
        x0=np.random.randn(self.xnew.shape[1])*.2 # shape (n_features,)
        guess = fsolve(self.fullgradient, x0=x0)
        return minimize(self.U, x0=guess).x.squeeze()
    
    def fullgradient(self,q): ## np version
        data = self.mybatcher.data[None]
        return self.grad(q, data)
    
    def stochgrad(self,q):
        data=self.mybatcher.sample()
        scaler=data.shape[1]/self.mybatcher.bs
        return self.grad(q, data)*scaler
        
class Sampler:
    def __init__(self, loss, method='SGLD', strat='RM'):
        self.loss = loss
        self.loss.set_strat(strat)
        self.strat=strat
        self.method=method.lower()
        self.ham = lambda q,v:.5*np.sum(v*(self.loss.J[None,...]@v))+self.loss.U(q)
        self.HMCsteps = 2

        if self.method=='hmc':
            self.stepper=self.HMC
        elif self.method=='sgld':
            self.stepper=self.SGLD
        else:
            raise ValueError('method arg to Sampler class not recognised: sgld and hmc are only available methods.')
        
    def run(self, h, Niters):
        global acc
        acc = 0
        q = np.float64(self.loss.MAP[None].repeat(repeats=self.loss.mybatcher.n_paths,axis=0))
        epochs=Niters//self.loss.mybatcher.K
        epochs+=1*epochs%2 #need number of epochs to be even for SMS
        Niters=epochs*self.loss.mybatcher.K
        samples=np.zeros((Niters,*q.shape))
        self.loss.mybatcher.index = 0 # reset index
        for n in range(0,Niters):
            q = self.stepper(q, h)
            samples[n] = q
        if acc !=0:
            print(f'Acceptance rate = {round(100 * acc / Niters, 2)}')
        return samples

    def SGLD(self, q, h): 
        eta = np.random.randn(*q.shape)
        scaled_eta = np.sqrt(2*h)*solve_triangular(self.loss.Jchol[None,...], eta, lower=False)
        grad = self.loss.fullgradient(q) if self.strat == 'FG' else self.loss.stochgrad(q)
        update=cho_solve((self.loss.Jchol[None,...], False), grad)
        m_q = q - h*update 
        qp = m_q + scaled_eta

        # Acceptance probability
        # global acc
        # U0=self.loss.U(q)
        # Jinv_g_qp = cho_solve((self.loss.Jchol[None,...], False), self.loss.stochgrad(qp))
        # u1 = (self.loss.Jchol[None,...] @ (q - qp + h * Jinv_g_qp))
        # u2 = (self.loss.Jchol[None,...] @ scaled_eta)

        # quad1 = np.sum(u1*u1)
        # quad2 = np.sum(u2*u2)

        # accept = -(quad1 - quad2) / (4.0 * h) - self.loss.U(qp) + U0

        # #Accept/reject
        # if (accept>np.log(np.random.rand(1))):
        #     acc += 1
        return qp

    def HMC(self, q, h): 
        global acc
        h=(1.-0.2*np.random.rand(1))*h
        v=solve_triangular(self.loss.Jchol[None,...], np.random.randn(*q.shape), lower=False) # Draw v ~ N(0,Jinv)
        qp = q.copy()
        H0=self.ham(q,v)
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
        return q
    
class LogReg(Loss):
    def __init__(self, data, K, n_paths, expname):
        n = int(data[0].shape[0])
        super().__init__(data, K, n_paths, expname, Cinv=1./25./n)
        self.MAP = self.calc_MAP()
        arg=self.xnew@self.MAP
        J=self.xnew.T*(npexpit(arg)*npexpit(-arg))@self.xnew / self.n
        J += self.Cinv * np.eye(J.shape[0])
        Jchol = cholesky(J, lower=False)
        self.J = J
        self.Jchol = Jchol
    
    def smoothness(self):
        covariance = self.xnew.T@self.xnew / self.n
        return 0.25*np.max(np.linalg.eigvalsh(covariance)) + self.Cinv
    
    def U(self,q):
        arg=(self.xnew@q[..., None]).squeeze() # (n, n_features) @ (n_paths, n_features)
        ans=-np.sum(self.y*arg) # (n, 1) * (n)
        ans+=np.sum(np.logaddexp(np.zeros_like(arg), arg))
        term=q*self.Cinv*q 
        return .5*np.sum(term) + ans / self.n
    
    def grad(self, q, data):
       x,y=data[...,:-1],data[...,-1] #x has shape (n_paths, n, n_features)
       term=q*self.Cinv #q has shape (n_paths, n_features)
       arg=np.matmul(x,q) #has shape (n_paths, n)
       temp=y-npexpit(arg) #has shape (n_paths, n)
       bs=x.shape[1] #self.n divide term/self.mybatcher.K for true splitting scheme
       temp = np.matmul(x.transpose(0,2,1),temp[...,None]).squeeze() / bs
       return term - temp

    
class Gaussian(Loss):
    def __init__(self,x,K,n_paths,strat='RM'):
        super().__init__((x, None), K, n_paths, Cinv=0., expname='Gaussian')
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


def load_data(expname, datadir='data'):
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
    
    return x, y

def get_HMC(loss, Nsamples_HMC, N_steps=3):
    print(f'Running HMC sampler to get true mean with {Nsamples_HMC} samples.')
    #HMC to get mean
    hpV = np.pi/2/N_steps
    sampler = Sampler(loss, method='hmc', strat='FG')
    Sampler.HMCsteps = N_steps
    samples = sampler.run(hpV, Nsamples_HMC)
    truemean = samples.mean(axis=0)

    with open(f"LogReg_{loss.expname}HMCtruemean.pkl", 'wb') as f:
        pickle.dump(truemean,f)