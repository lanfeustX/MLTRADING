import numpy as np

import math

from math import *

import cmath

from scipy.stats import norm




N_prime = norm.pdf

N = norm.cdf



def cn_put( T, sigma, kT):

    if np.any(kT == 0.0): return 0.0

    s    = sigma*np.sqrt(T)

    d    = ( np.log(kT) + .5*s*s)/s

    mask=np.logical_and((s<1.e-08),(kT > 1.0))

    temp=np.where(s<1.e-08,0,norm.cdf(d))

    temp=np.where(mask,1,temp)



    #if np.any(s < 1.e-08):

    #    if np.any(kT > 1.0): return 1.

    #    else       : return 0.

    #d    = ( log(kT) + .5*s*s)/s

    return temp

# ------------------------------------



def an_put( T, sigma, kT):

    if np.any(kT == 0.0): return 0.0

    s    = sigma*np.sqrt(T)

    d    = ( np.log(kT) + .5*s*s)/s

    mask=np.logical_and((s<1.e-08),(kT > 1.0))

    temp=np.where(s<1.e-08,0,norm.cdf(d-s))

    temp=np.where(mask,1,temp)



    #if np.any(s < 1.e-08):

    #    if np.any(kT > 1.0): return 1.

    #    else       : return 0.

    #d    = ( log(kT) + .5*s*s)/s

    return temp

# ------------------------------------



def FwEuroPut(T, sigma, kT):

    return ( kT* cn_put( T, sigma, kT) - an_put( T, sigma, kT) )



def FwEuroCall(T, sigma, kT):

    return FwEuroPut(T, sigma, kT) + 1. - kT



def euro_put(So, r, q, T, sigma, k):

    kT   = np.exp((q-r)*T)*k/So

    return So*np.exp(-q*T) * FwEuroPut( T, sigma, kT)

# -----------------------



def euro_call(So, r, q, T, sigma, k):

    kT   = np.exp((q-r)*T)*k/So

    return So*np.exp(-q*T) * FwEuroCall( T, sigma, kT)

# -----------------------



def recursive_impVol( price, sL, sH, T, kT, accepted_vol):

    '''

    pact: price(sL) < price < price(sH)

    '''

    sH=np.ones(len(price))*sH

    sL=np.ones(len(price))*sL

    sM = .5*(sH+sL)



    accepted_vol=np.where(sH - sL < 1.e-8,sM,accepted_vol)



    pM = FwEuroPut(T, sM, kT)

    #if np.absolute( pM - price)< 1.e-10: return sM

    accepted_vol=np.where(np.absolute(pM - price)< 1.e-10,sM,accepted_vol)

    if not np.any(accepted_vol==0):

        return accepted_vol

    sL = np.where(pM<price,sM,sL)

    sH= np.where(pM>=price,sM,sH)

    #if pM < price: return recursive_impVol( price, sM, sH, T, kT)

    return recursive_impVol( price, sL, sH, T, kT,accepted_vol)

# --------------------------------------------





def impVolFromFwPut(price, T, kT):

    '''

    if np.any(price <= np.maximum(kT-1.0,np.zeros(len(price)))):

        raise Exception("\n\n One Price is too low")

    if np.any(price >= kT):

        raise Exception("\n\n One Price is too high")

    '''

    price=np.where(price <= np.maximum(kT-1.0,np.zeros(len(price))),np.maximum(kT-1.0,np.zeros(len(price))),price)

    price=np.where(price >= kT,kT,price)





    sL = 0.0

    #pL = FwEuroPut(T, sL, kT)



    sH = 1.0



    while True:

        pH = FwEuroPut(T, sH, kT)

        accepted_vol=np.where(np.absolute(pH - price) < 1.e-10, sH, 0)

        #if fabs( pH - price) < 1.e-10: return sH

        if np.all(pH > price): break

        sH = sH *2



    if not np.any(accepted_vol==0):

        return accepted_vol





    return recursive_impVol( price, sL, sH, T, kT,accepted_vol)





def newImpVolFromFwPut(price, T, kT):

    price=np.where(price <= np.maximum(kT-1.0,np.zeros(len(price))),1.e-6+np.maximum(kT-1.0,np.zeros(len(price))),price)

    price=np.where(price >= kT,kT,price)





    sL = 0.0





    sH = 1.0



    while True:

        pH = FwEuroPut(T, sH, kT)

        accepted_vol=np.where(np.absolute(pH - price) < 1.e-10, sH, 0)

        if np.all(pH > price): break

        sH = sH *2



    if not np.any(accepted_vol==0):

        return accepted_vol



    # make sure why this better to implement

    tol=1.e-12

    max_iter=log((sH-sL)/tol)/log(2)

    max_iter=int(max_iter)



    for i in range(max_iter):

        '''

        pact: price(sL) < price < price(sH)

        '''

        sH=np.ones(len(price))*sH

        sL=np.ones(len(price))*sL

        sM = .5*(sH+sL)







        pM = FwEuroPut(T, sM, kT)



        accepted_vol=np.where(np.absolute(pM - price)< 1.e-12,sM,accepted_vol)

        if not np.any(accepted_vol==0):

            return accepted_vol

        sL = np.where(pM<price,sM,sL)

        sH= np.where(pM>=price,sM,sH)

        max_iter -=1





    sM = .5 * (sH + sL)

    accepted_vol=np.where(accepted_vol==0,sM,accepted_vol)

    return accepted_vol







def black_scholes_call(S, K, T, r, sigma):

    '''



    :param S: Asset price

    :param K: Strike price

    :param T: Time to maturity

    :param r: risk-free rate (treasury bills)

    :param sigma: volatility

    :return: call price

    '''



    ###standard black-scholes formula

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)



    call = S * N(d1) -  N(d2)* K * np.exp(-r * T)

    return call



def vega(S, K, T, r, sigma):

    '''



    :param S: Asset price

    :param K: Strike price

    :param T: Time to Maturity

    :param r: risk-free rate (treasury bills)

    :param sigma: volatility

    :return: partial derivative w.r.t volatility

    '''



    ### calculating d1 from black scholes

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)



    #see hull derivatives chapter on greeks for reference

    vega = S * N_prime(d1) * np.sqrt(T)

    return vega







def implied_volatility_call(C, S, K, T, r, tol=1.e-10,

                            max_iterations=500):

    '''



    :param C: Observed call price

    :param S: Asset price

    :param K: Strike Price

    :param T: Time to Maturity

    :param r: riskfree rate

    :param tol: error tolerance in result

    :param max_iterations: max iterations to update vol

    :return: implied volatility in percent

    '''



    C=np.where(C <= np.maximum(S-K,0,np.zeros(len(C))),1.e-6+np.maximum(S-K,np.zeros(len(C))),C)

    C=np.where(C >= S,S,C)





    ### assigning initial volatility estimate for input in Newton_rap procedure

    sigma = 0.3*np.ones(len(S))



    for i in range(max_iterations):



        ### calculate difference between blackscholes price and market price with

        ### iteratively updated volality estimate

        diff = black_scholes_call(S, K, T, r, sigma) - C



        ###break if difference is less than specified tolerance level

        if np.all(np.absolute(diff) < tol):

            print(f'found on {i}th iteration')

            print(f'difference is equal to {diff}')

            break



        ### use newton rapshon to update the estimate

        sigma = sigma - diff / vega(S, K, T, r, sigma)



    return sigma