import numpy as np
from numpy.fft import fft, ifft
import pandas as pd

from HestonModel_FRFT import *
from DataProcessing import *
from scipy.optimize import minimize

import timeit
import time

import csv


def linearInterpolate(X, Y, Xi):
    N = len(X)
    M = len(Xi)
    Yi = []
    #print("X is: ", X)
    #print("Xi is: ", Xi)
    
    
    for j in range(M):
        k = np.where(Xi[j] <= X)
        k = k[0][0]-1
        Yi.append(Y[k+1]*(Xi[j]-X[k])/(X[k+1]-X[k])+Y[k]*(X[k+1]-Xi[j])/(X[k+1]-X[k]))
            
    return np.array(Yi)

def hestonObjFunFRFT(param, *args):
    """input:
        S
        K0
        rf
        q
        MktPrice
        K
        T
        N
        eta
        alpha
    """
    S = args[0]
    K0 = args[1]
    rf = args[2]
    q = args[3]
    MktPrice = args[4]
    K = args[5]
    T = args[6]
    N = args[7]
    eta = args[8]
    alpha = args[9]
    
    kappa = param[0]
    theta = param[1]
    sigma = param[2]
    v0 = param[3]
    rho = param[4]
    laambda = 0
    
    nk, nt = MktPrice.shape
    lambdainc = 2/N*np.log(S/K0)+0.001
    #init
    ModelPrice = np.zeros((nk, nt))
    error = 0
    
    for t in range(nt):
        CallFRFT, KK, lambdainc, eta = hestonCallPriceFRFT(N, S, rf[t], q, T[t], kappa, theta, 
                                                           laambda, rho, sigma, v0, alpha, eta, lambdainc)
        CallPrice = linearInterpolate(KK, CallFRFT, K)
        for k in range(nk):
            ModelPrice[k, t] = CallPrice[k]
        mask = ~np.isnan(MktPrice[:, t])
        error += sum(np.absolute(MktPrice[:,t][mask] - ModelPrice[:,t][mask]) / np.absolute(MktPrice[:,t][mask])) #sum((MktPrice[:,t][mask] - ModelPrice[:,t][mask])**2)

    return error / len(MktPrice[~np.isnan(MktPrice)])

def generalizedHestonObjFun(param, *args):
    error = 0
        
    for i in args:
        error += hestonObjFunFRFT(param, i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9])
    return error / len(args)
        

def estimateParameters(data):
    #estimation bounds
    b = 1e-5
    kappa_bounds = (b, 20)
    theta_bounds = (b, 2)
    sigma_bounds = (b, 2)
    v0_bounds = (b,2)
    rho_bounds = (-.999, .999)
    
    bnds = (kappa_bounds, theta_bounds, sigma_bounds, v0_bounds, rho_bounds)
    
    start = np.array([9.0, 0.05, 0.3, 0.05, -0.8])
    
    N = 2**11
    eta = 0.25
    alpha = 1.75
    
    args = []
    for i in data:
        args.append((i[0], i[4][0]-1, i[1], i[2], i[3], i[4], i[5], N, eta, alpha))
        
    args = tuple(args)
     
    timer = timeit.default_timer()
    
    result = minimize(generalizedHestonObjFun, start, 
                      args, 
                      bounds = bnds, method="SLSQP")
    
    stop = timeit.default_timer()
    print("Runtime: ", stop - timer)
    print("Parameters: ", result["x"])
    print("MSE: ", result["fun"])
    
    return result

def testParameters(data, param):
    S, rf, q, MktPrice, K, T = data
    N = 2**10
    eta = 0.25
    alpha = 1.75
    K0 = K[0]-1
    
    MSE = hestonObjFunFRFT(param, S, K0, rf, q, MktPrice, K, T, N, eta, alpha)
    
    return MSE

def generalizedParameterTest(data, param):
    N = 2**10
    eta = 0.25
    alpha = 1.75
    args = []
    for i in data:
        args.append((i[0], i[4][0]-1, i[1], i[2], i[3], i[4], i[5], N, eta, alpha))
        
    args = tuple(args)
        
    MSE = generalizedHestonObjFun(param, args[0], args[1], args[2], args[3], args[4])
    
    return MSE

def getFormattedData(listofdf):
    data = []
    
    for i in listofdf:
        data.append(formatData(i))
        
    return data

def getTrainAndTestSet(data, train_size):
    intermediate_train_set = []
    test_set = []
    
    for i in range(len(data)):
        if (i+1) % (train_size+1) == 0:
            test_set.append(data[i])
        else:
            intermediate_train_set.append(data[i])      
    
    train_set = []
    count = 0
    for j in range(len(test_set)):
        intermediate_list = []
        for k in range(train_size):
            intermediate_list.append(intermediate_train_set[count])
            count += 1
        train_set.append(tuple(intermediate_list))
            
    return tuple(train_set), tuple(test_set)

def getAllParameters(training_data):
    param = []
    count = 1
    
    for i in training_data:
        print("Parameter set nr. ", count)
        param.append(estimateParameters(i))
        count += 1
        if count % 50 == 0:
            with open('parameters.csv', 'w') as csvfile:
                wr = csv.writer(csvfile)
                wr.writerow(param)
                print("Paramaters were saved to 'parameters.csv'")
        
    return tuple(param)

def estimationOnEntireSet(df, train_size):
    seperate_days = getTermStructure(df)
    
    data = tuple(getFormattedData(seperate_days))
    
    training_data, test_data = getTrainAndTestSet(data, train_size)
    
    parameters = getAllParameters(training_data)
    
    return parameters

def testOnEntireSet(test_set, parameters):
    mse = []
    outputs = []
    for i in range(len(test_set)):
        try:    
            mse.append(testParameters(test_set[i], parameters[i]))
        except:
            print('Error on set: ', i)
        
        
    return np.array(mse)

def generalizedTestOnEntireSet(test_set, parameters):
    mse = []
    outputs = []
    for i in range(len(test_set)):
        mse.append(generalizedParameterTest(test_set[i], parameters[i]))
        
    return np.array(mse)

def getCallPrices(param, data):
    
    S, rf, q, MktPrice, K, T = data
    N = 2**10
    eta = 0.25
    alpha = 1.75
    K0 = K[0]-1
    
    kappa = param[0]
    theta = param[1]
    sigma = param[2]
    v0 = param[3]
    rho = param[4]
    laambda = 0
    
    nk, nt = MktPrice.shape
    lambdainc = 2/N*np.log(S/K0)+0.001
    #init
    ModelPrice = np.zeros((nk, nt))
    error = 0
    prices = []
    
    for t in range(nt):
        CallFRFT, KK, lambdainc, eta = hestonCallPriceFRFT(N, S, rf[t], q, T[t], kappa, theta, 
                                                           laambda, rho, sigma, v0, alpha, eta, lambdainc)
        CallPrice = linearInterpolate(KK, CallFRFT, K)
        for k in range(nk):
            ModelPrice[k, t] = CallPrice[k]
        
        mask = ~np.isnan(MktPrice[:, t])
        prices.append(ModelPrice[:,t][mask])

    
    return np.concatenate(prices)

def formatPrices(ModelPrice, data):
    S, rf, q, MktPrice, K, T = data
    strikes = []
    maturities = []
    prices = []
    r = []

    for k in range(len(K)):
        for t in range(len(T)):
            strikes.append(K[k])
            maturities.append(T[t])
            prices.append(ModelPrice[k, t])
            r.append(rf[t]*100)
            
    d = {'strike_price' : strikes, 'ttm' : maturities, 'call_price' : prices, 'r' : r}
    df = pd.DataFrame(data = d)
    
    df['close'] = np.full(len(df['strike_price']), S)
    df['moneyness'] = df['strike_price'] / df['close']
    df['q'] = np.full(len(df), q)*100
    
    
    return df

def getPricesForMatch(day, data, param, index):
    prices = getCallPrices(param, data)
    temp = []
    df = day.sort_values(['ttm', 'strike_price'])
    df = df.reset_index()
    
    count = 0
    for i in range(len(prices)):
        temp.append(prices[i])
        count = count + 1
        
        try:
            if df['strike_price'][count] == df['strike_price'][count-1]:
                temp.append(prices[i])
                count = count + 1
        except:
            if ((index + 1) % 100 == 0) or (index + 1 == 5520):
                print('Day nr. ', index + 1)
    df['prices'] = temp
    df = df.sort_values('index')
    
    return np.array(df['prices'])
    

def matchHestonToDF(seperate_days, data, param):
    heston = []
    
    count = 0
    for i in range(len(seperate_days)):
        try:
            heston.append(getPricesForMatch(seperate_days[i], data[i], param[count], i))
        except:
            heston.append(np.full(len(seperate_days[i]), 'nan'))
            print('error')
        if (i+1) % 6 == 0:
            count = count + 1
            
    
    return np.concatenate(heston)
