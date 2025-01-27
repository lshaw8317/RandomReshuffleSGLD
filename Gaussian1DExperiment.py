from LogisticRegressionExperimentsCode import *

#%% Gaussian 1D Model Problem
torch.manual_seed(2024)
K=2**3
N=20*K
bs=N//K
d=1
n_paths=20
x=torch.randn((N,d,1)).squeeze(-1) #number of examples, number of features, 1
Exp1=GaussianExp(x, bs,n_paths=n_paths)


with open("Gaussian1D_Data.pkl", 'wb') as f:
    pickle.dump({'x':x},f)

with open("Gaussian1D_truemeancov.pkl", 'wb') as f:
    pickle.dump({'truemean':Exp1.truemean,'truecov':Exp1.truecov},f)

with open("Gaussian1D_truemeancov.pkl", 'rb') as f:
    d=pickle.load(f)
    truecov=d['truecov']
    truemean=d['truemean']

etarange = (2.**torch.arange(-3,1,1)*1.1)/K
Nsamples=((100 +(20/(etarange*K)**3))*K).to(torch.int32)
truecov,truemean=Exp1.truecov,Exp1.truemean
strats=['RR','RM']
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
