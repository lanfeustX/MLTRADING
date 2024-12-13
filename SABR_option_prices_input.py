'Author Riccardo Lamia, UT35TK'
'SABR formulas implementations are taken form pysabr gihthub repository as of 11/05/2022'

from cmath import log, sqrt, exp
import math
from statistics import variance
from turtle import right
import pandas as pd
import numpy as np
from pysabr import Hagan2002LognormalSABR
from scipy.stats import norm, skew
from scipy.optimize import minimize
import random
import copy
import scipy

def BlackD1D2(forward,strike,variance):
    d1 = (log(forward/strike)+0.5*variance) / sqrt(variance)
    d2 = d1 - sqrt(variance)
    return [d1, d2]

def Black(forward, strike, variance,cp):
    d1d2= BlackD1D2(forward,strike,variance)
    return cp*(forward*Normal_CDF_Wilmot(cp*d1d2[0])-strike*Normal_CDF_Wilmot(cp*d1d2[1]))

def BlackVega(forward,strike,variance, expiry):
    d1d2 = BlackD1D2(forward,strike,variance)

    return forward*sqrt(expiry/3.1415926535 /2)*exp(-0.5*d1d2[0]*d1d2[0])

def BlackImpVol(mktPrice,forward,strike,expiry,cp,error):
    volatility =0.2

    dv = error+1
    while abs(dv)> error:
        variance = volatility*volatility*expiry
        priceError = Black(forward,strike,variance,cp)-mktPrice
        vega =BlackVega(forward,strike,variance,expiry)
        dv = priceError/vega
        volatility = volatility - dv
    return volatility


def Normal_CDF_Wilmot(x):
    #http://iman.sn/bibliotek/livres/filieres/banque-finance-assurance/pdfs/paul-wilmott-on-quantitative-finance.pdf page 131
    
    d = 1 / (1 + 0.2316419 * abs(x))
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    temp = a5
    temp = a4 + d * temp
    temp = a3 + d * temp
    temp = a2 + d * temp
    temp = a1 + d * temp
    temp = d*temp
    cdf = 1-1/(sqrt(2*3.1415926))*exp(-0.5*x*x)*temp
    if x >=0:
        return cdf
    else:
        return 1-cdf


strikes_near = pd.read_excel('SABR_data_for_black_near.xlsx', sheet_name='strikes_near',index_col=0)
prices_near = pd.read_excel('SABR_data_for_black_near.xlsx', sheet_name='prices_near',index_col=0)
t_near = pd.read_excel('SABR_data_for_black_near.xlsx', sheet_name='t_near',index_col=0)
f_near = pd.read_excel('SABR_data_for_black_near.xlsx', sheet_name='f_near',index_col=0)
w = pd.read_excel('SABR_data_for_black_near.xlsx', sheet_name='w',index_col=0)


a_prices_near=np.array(prices_near)
a_t_near=np.array(t_near)
a_f_near=np.array(f_near)
a_w = np.array(w)
a_strikes_near = np.array(strikes_near)

'''
for j in range(len(imp_vol_near)):
    imp_vol_near.replace(0,imp_vol_near.mean(axis=0),inplace=True)
'''
#print(a_prices_near)
#Finding implied volatility of puts
imp_vol_puts_near = []
for i in range(len(a_prices_near)):
    imp_vol_puts_row = []
    for j in range(0,12):
        imp_vol= BlackImpVol(a_prices_near[i][j],a_f_near[i],strike=a_strikes_near[i][j],expiry=a_t_near[i]/252,cp=-1,error=10e-8)
        imp_vol_puts_row.append(float(imp_vol.real))
    imp_vol_puts_near.append(imp_vol_puts_row)
#print(imp_vol_puts_near)
df_p_near = pd.DataFrame(imp_vol_puts_near)

imp_vol_calls_near = []
for i in range(len(a_prices_near)):
    imp_vol_calls_row = []
    for j in range(12,len(a_prices_near[i])):
        imp_vol= BlackImpVol(a_prices_near[i][j],a_f_near[i],strike=a_strikes_near[i][j],expiry=a_t_near[i]/252,cp=1,error=10e-8)
        imp_vol_calls_row.append(float(imp_vol.real))
    imp_vol_calls_near.append(imp_vol_calls_row)
#print(imp_vol_calls_near)
df_c_near = pd.DataFrame(imp_vol_calls_near)

imp_vol_near = pd.merge(df_p_near,df_c_near,left_index=True,right_index=True, how="inner")

#Applying the same routine to next expiry options
strikes_next = pd.read_excel('SABR_data_for_black_next.xlsx', sheet_name='strikes_next',index_col=0)
prices_next = pd.read_excel('SABR_data_for_black_next.xlsx', sheet_name='prices_next',index_col=0)
t_next = pd.read_excel('SABR_data_for_black_next.xlsx', sheet_name='t_next',index_col=0)
f_next = pd.read_excel('SABR_data_for_black_next.xlsx', sheet_name='f_next',index_col=0)


a_prices_next=np.array(prices_next)
a_t_next=np.array(t_next)
a_f_next=np.array(f_next)
a_strikes_next = np.array(strikes_next)

'''
for j in range(len(imp_vol_near)):
    imp_vol_near.replace(0,imp_vol_near.mean(axis=0),inplace=True)
'''
#print(a_prices_near)
#Finding implied volatility of puts
imp_vol_puts_next = []
for i in range(len(a_prices_next)):
    imp_vol_puts_row = []
    for j in range(0,12):
        imp_vol= BlackImpVol(a_prices_next[i][j],a_f_next[i],strike=a_strikes_next[i][j],expiry=a_t_next[i]/252,cp=-1,error=10e-8)
        imp_vol_puts_row.append(float(imp_vol.real))
    imp_vol_puts_next.append(imp_vol_puts_row)
#print(imp_vol_puts_near)
df_p_next = pd.DataFrame(imp_vol_puts_next)

imp_vol_calls_next = []
for i in range(len(a_prices_next)):
    imp_vol_calls_row = []
    for j in range(12,len(a_prices_next[i])):
        imp_vol= BlackImpVol(a_prices_next[i][j],a_f_next[i],strike=a_strikes_next[i][j],expiry=a_t_next[i]/252,cp=1,error=10e-8)
        imp_vol_calls_row.append(float(imp_vol.real))
    imp_vol_calls_next.append(imp_vol_calls_row)
#print(imp_vol_calls_near)
df_c_next = pd.DataFrame(imp_vol_calls_next)

imp_vol_next = pd.merge(df_p_next,df_c_next,left_index=True,right_index=True, how="inner")

#Ideally create a function for checking if the lenght is different and returns a matched dataframe
len(imp_vol_next)<len(imp_vol_near)

#Setting date index into the imp_vol dfs

near_t_idx = t_near.index
imp_vol_near.set_index(near_t_idx,inplace=True)
next_t_idx = t_next.index
imp_vol_next.set_index(next_t_idx,inplace=True)

imp_vol_near = copy.deepcopy(imp_vol_near.loc[next_t_idx])

strikes_near = copy.deepcopy(strikes_near.loc[next_t_idx])

prices_near = copy.deepcopy(prices_near.loc[next_t_idx])

t_near = copy.deepcopy(t_near.loc[next_t_idx])

f_near = copy.deepcopy(f_near.loc[next_t_idx])

w = copy.deepcopy(w.loc[next_t_idx])



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
'''
for j in range(len(imp_vol_next)):
    imp_vol_next.replace(0,imp_vol_next.mean(axis=0),inplace=True)
'''

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

skewness = []
for i in range(len(pdfs)):
    elem = scipy.stats.skew(pdfs[i])
    skewness.append(elem)

kurtosis = []
for i in range(len(pdfs)):
    elem = scipy.stats.kurtosis(pdfs[i])
    kurtosis.append(elem)




index = t_next.index
print(index)
dfIndex=pd.DataFrame(Index,index=index)
#dfIndex.plot()

oat_yield = pd.read_excel('yield_OAT.xlsx', sheet_name='yield',index_col=0)

dfIndex["Yield"] = oat_yield


dfIndex["smooth"] = dfIndex[0].rolling(20).mean()

dfIndex["skewness"] = skewness

dfIndex["kurtosis"] = kurtosis



dfIndex[["smooth","Yield","skewness","kurtosis"]].plot()