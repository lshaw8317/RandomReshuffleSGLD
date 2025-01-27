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

#Case V_b^K sum over n epochs
epoch_sumij=system_ij.subs({r:K})
epoch_sumii=system_ii.subs({r:K})

eDhj=(1-h)**(j*K)
epoch_sumij=eDhj**2*epoch_sumij
epoch_sumii=eDhj**2*epoch_sumii

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

#Plotting
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
ls=['-','--',':','-.']
for idx,row in enumerate(ans):
    # plt.semilogy(rrange/Kval,row,base=2)
    s=perioderrs[idx]
    line,=plt.semilogy(rrange/Kval,s*np.ones_like(rrange),color='k',base=2,ls=ls[idx],label='$h=2^{'+f'{round(np.log2(hrange[idx]))}'+'}$',alpha=1)
    plt.semilogy(rrange/Kval,row,base=2,ls=line.get_ls(),alpha=.5,color='k')

# plt.loglog(rrange,varfunc(hrange)[-1]*hrange/2,'k--',label='ref line $h^1$',base=2)
plt.ylabel('Relative variance error')
plt.xlabel('Iteration over dataset')
plt.title(f'Theoretical SGLD-RR, $R={Kval}$')
plt.legend()
plt.savefig(os.path.join(figdir,'EMRR1.pdf'),format='pdf',bbox_inches='tight')

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
