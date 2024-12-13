'Author Riccardo Lamia, UT35TK'
'SABR formulas implementations are taken form pysabr gihthub repository as of 11/05/2022'
#from multiprocessing import Value
#from operator import index
import pandas as pd
import numpy as np
from pysabr import Hagan2002LognormalSABR
from scipy.stats import norm
from scipy.optimize import minimize
import random



strikes_near = pd.read_excel('Data_for_python_near.xlsx', sheet_name='strikes',index_col=0)
imp_vol_near = pd.read_excel('Data_for_python_near.xlsx', sheet_name='imp_vol',index_col=0)
t_near = pd.read_excel('Data_for_python_near.xlsx', sheet_name='expiry',index_col=0)
f_near = pd.read_excel('Data_for_python_near.xlsx', sheet_name='futures',index_col=0)
w = pd.read_excel('w.xlsx', sheet_name='w',index_col=0)

imp_vol_near.iloc[1]

for j in range(len(imp_vol_near)):
    imp_vol_near.replace(0,imp_vol_near.mean(axis=0),inplace=True)


lst = []

for i in range(len(strikes_near)):
    sabr = Hagan2002LognormalSABR(f=float(f_near.iloc[i])/100, shift=0, t=float(t_near.iloc[i])/252, beta=1)
    k_near = np.array(strikes_near.iloc[i])/100
    vol_near = np.array(imp_vol_near.iloc[i])*100

    
    #returned as [alpha,rho, volvol (nu)]
    par_near =sabr.fit(k_near,vol_near)
    lst.append(par_near)



SABR_params_near= pd.DataFrame(lst)
#SABR_params.to_excel('SABR_par_near.xlsx')

arr_parr_near = np.array(lst)

width =0.10
#width parameter will be used also later when computing PDF it is the distance between the strikes we want to compute


#Creating strikes for interpolation
new_k_near= []
for i in range(len(strikes_near)):
    j=0
    lst2 = []
    while j<=20:
        lst2.append(strikes_near.iloc[i,0]+j)

        j += width
    new_k_near.append(lst2)

#print(new_k)
def lognormal_vol(k, f, t, alpha, beta, rho, volvol):
    """
    Hagan's 2002 SABR lognormal vol expansion.
    The strike k can be a scalar or an array, the function will return an array
    of lognormal vols.
    """
    # Negative strikes or forwards
    if k <= 0 or f <= 0:
        return 0.
    eps = 1e-07
    logfk = np.log(f / k)
    fkbeta = (f*k)**(1 - beta)
    a = (1 - beta)**2 * alpha**2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / fkbeta**0.5
    c = (2 - 3*rho**2) * volvol**2 / 24
    d = fkbeta**0.5
    v = (1 - beta)**2 * logfk**2 / 24
    w = (1 - beta)**4 * logfk**4 / 1920
    z = volvol * fkbeta**0.5 * logfk / alpha
    # if |z| > eps
    if abs(z) > eps:
        vz = alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
        return vz
    # if |z| <= eps
    else:
        v0 = alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))
        return v0

def _x(rho, z):
    """Return function x used in Hagan's 2002 SABR lognormal vol expansion."""
    a = (1 - 2*rho*z + z**2)**.5 + z - rho
    b = 1 - rho
    return np.log(a / b)

#interpolating volatility with SABR parameters optimized using market prices
int_vol_near=[]
for i in range(len(lst)):
    vol_row=[]
    for l in range(len(new_k_near[i])):
        int_sabr = lognormal_vol(k=np.array(new_k_near[i][l])/100 ,f =np.array(f_near.iloc[i]/100), t=float(t_near.iloc[i])/252,alpha=arr_parr_near[i][0], beta=1, rho= arr_parr_near[i][1],volvol=arr_parr_near[i][2])
        vol_row.append(float(int_sabr))
    int_vol_near.append(vol_row)


#Implementing the same routine for next expiry option

strikes_next = pd.read_excel('Data_for_python_next.xlsx', sheet_name='strikes',index_col=0)
imp_vol_next = pd.read_excel('Data_for_python_next.xlsx', sheet_name='imp_vol',index_col=0)
t_next = pd.read_excel('Data_for_python_next.xlsx', sheet_name='expiry',index_col=0)
f_next = pd.read_excel('Data_for_python_next.xlsx', sheet_name='futures',index_col=0)

for j in range(len(imp_vol_next)):
    imp_vol_next.replace(0,imp_vol_next.mean(axis=0),inplace=True)


lst = []

for i in range(len(strikes_next)):
    sabr = Hagan2002LognormalSABR(f=float(f_next.iloc[i])/100, shift=0, t=float(t_next.iloc[i])/252, beta=1)
    k_next = np.array(strikes_next.iloc[i])/100
    vol_next = np.array(imp_vol_next.iloc[i])*100

    
    #returned as [alpha,rho, volvol (nu)]
    par_next =sabr.fit(k_next,vol_next)
    lst.append(par_next)

SABR_params_next= pd.DataFrame(lst)
#SABR_params.to_excel('SABR_par_near.xlsx')

arr_parr_next = np.array(lst)

#width parameter will be used also later when computing PDF it is the distance between the strikes we want to compute


#Creating strikes for interpolation
new_k_next= []
for i in range(len(strikes_next)):
    j=0
    lst2 = []
    while j<=20:
        lst2.append(strikes_next.iloc[i,0]+j)

        j += width
    new_k_next.append(lst2)



#interpolating volatility with SABR parameters optimized using market prices
int_vol_next=[]
for i in range(len(lst)):
    vol_row=[]
    for l in range(len(new_k_next[i])):
        int_sabr = lognormal_vol(k=np.array(new_k_next[i][l])/100 ,f =np.array(f_next.iloc[i]/100), t=float(t_next.iloc[i])/252,alpha=arr_parr_next[i][0], beta=1, rho= arr_parr_next[i][1],volvol=arr_parr_next[i][2])
        vol_row.append(float(int_sabr))
    int_vol_next.append(vol_row)






int_vol_df_next = pd.DataFrame(int_vol_next)




def lognormal_call(k, f, t, v, r, cp='call'):
    """Compute an option premium using a lognormal vol."""
    if k <= 0 or f <= 0 or t <= 0 or v <= 0:
        return 0.
    d1 = (np.log(f/k) + v**2 * t/2) / (v * t**0.5)
    d2 = d1 - v * t**0.5
    if cp == 'call':
        pv = np.exp(-r*t) * (f * norm.cdf(d1) - k * norm.cdf(d2))
    elif cp == 'put':
        pv = np.exp(-r*t) * (-f * norm.cdf(-d1) + k * norm.cdf(-d2))
    else:
        pv = 0
    return pv

#interpolating futures for constant maturity outlook


int_futures=[]
for i in range(len(w)):
    int_fut = float(w.iloc[i])*float(f_near.iloc[i])+(1-float(w.iloc[i]))*float(f_next.iloc[i])
    int_futures.append(int_fut)


atm_stk = pd.DataFrame(int_futures)

#interpolating vols for constant maturity outlook
int_vols = []
for i in range(len(int_vol_near)):
    int_vols_row = []
    for j in range(len(int_vol_next[i])):
        int_vol = float(w.iloc[i])*int_vol_near[i][j]+(1-float(w.iloc[i]))*int_vol_next[i][j]
        int_vols_row.append(int_vol)
    int_vols.append(int_vols_row)

new_k_int= []
for i in range(len(atm_stk)):
    j=-10
    lst2 = []
    while j<=10:
        lst2.append(atm_stk.iloc[i,0]+j)

        j += width
    new_k_int.append(lst2)

#print(new_k_int[1])
#print(atm_stk.iloc[1,0])
pd.DataFrame(int_vol_next[1000]).plot()

#Pricing calls with interpolated implied volatilities

int_calls = []
for i in range(len(int_vols)):
    calls_row = []
    for j in range(len(int_vols[i])):
        cp = lognormal_call(k=new_k_int[i][j],f =int_futures[i], t= 20/252, v = int_vols[i][j], r=0, cp='call')
        calls_row.append(float(cp))
    int_calls.append(calls_row)




#computing pdf from calls
pdfs = []
for i in range(len(int_calls)):
    pdfs_row=[]
    for j in range(1,len(int_vols[i])-1):
        pdf = (int_calls[i][j-1]-2*int_calls[i][j]+int_calls[i][j+1])/(width)**2
        pdfs_row.append(pdf)
    pdfs.append([elem/sum(pdfs_row) for elem in pdfs_row])
    #pdfs.append(pdfs_row)



'''
for i in range(1,200):
    pd.DataFrame(pdfs[-i]).plot()
'''
'''
Creating the index as Left/right tail, P(x€[-6,-3])/P(x€[3,6]), where x is what is the price change in non relative terms.
If Index >>1 Options are pricing high probability of Yield increase
If Index <<1 Options are pricing high probability of Yield decrease
Taking this intervals because those are the most extremes which are still calibrated through market prices, hence in there there should be the most extreme values for which are not replicated, but taken from market prices and interpolated. 
For reference, assuming an approx 10 Duratio/Mod Duration a -6 move in the Futures imply a 60bps move in the CTD bond and a -3 move about 30 bps move in the CTD bond.

'''
Index =[]
for i in range(len(pdfs)):
    value = pd.DataFrame(pdfs[i])[40:70].sum()/pd.DataFrame(pdfs[i])[130:160].sum()
    Index.append(value)

index = t_near.index
print(index)
dfIndex=pd.DataFrame(Index,index=index)
#dfIndex.plot()

btp_yield = pd.read_excel('yield.xlsx', sheet_name='yield',index_col=0)

dfIndex["Yield"] = btp_yield

dfIndex.plot()
