from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
from scipy.interpolate import interp1d
from HestonParameterEstimation import *
import time
#mpl.rcParams['agg.path.chunksize'] = 10000

def flatten(inputList):
    """Flattens list of lists to single, longer list"""
    flatList = []
    for sublist in inputList:
        for item in sublist:
            flatList.append(item)
    
    return flatList

def readData(query):
    if query == "options":
        df = pd.read_csv("C:/Users/Magnus/OneDrive - Nordic Biotech Advisors ApS/SPXOptions.csv")
    elif query == "optionsOTM":
        df = pd.read_csv("C:/Users/Magnus/OneDrive - Nordic Biotech Advisors ApS/SPXOptionsITMRemoved.csv")
    elif query == "filteredoptions":
        df = pd.read_csv("C:/Users/Magnus/OneDrive - Nordic Biotech Advisors ApS/SPXOptionsFiltered.csv")
    elif query == "dividends":
        df = pd.read_csv("C:/Users/Magnus/OneDrive - Nordic Biotech Advisors ApS/Dividends.csv")
    elif query == "price":
        df = pd.read_csv("C:/Users/Magnus/OneDrive - Nordic Biotech Advisors ApS/Price.csv")
    elif query == "yieldcurve":
        df = pd.read_csv("C:/Users/Magnus/OneDrive - Nordic Biotech Advisors ApS/YieldCurve.csv")
    
    return df

def splitPutCall(df):
    cp = {k: v for k, v in df.groupby('cp_flag')}
    
    return cp

def BSMCallPriceContinuousDividend(S0, K, r, q, T, sigma):
    """Calculates the Black Scholes Merton call price with continuous dividend"""
    s_ex = S0*np.exp(-q*T)
    d1 = (np.log(S0/K) + (r-q+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    c = s_ex*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
    
    return np.array(c)

def putToCall(put):
    """returns call price by the put-call parity"""
    c = put["bidaskmid"] + put["close"]*np.exp(-(put["q"]/100)*put["ttm"]) - put["strike_price"]*np.exp(-(put["r"]/100)*put["ttm"])
    
    return np.array(c)

def getBSIVFromCall(call):
    """Approximates the Black Scholes implied volatility from call price
                        !!!NOT IMPLEMENTED YET!!!"""
    return IV
    

def getTTM(date, exdate):
    """returns time-to-maturity in days"""
    date_format = "%Y%m%d"
    
    date = datetime.strptime(str(date), date_format)
    exdate = datetime.strptime(str(exdate), date_format)
    
    delta = exdate - date
    
    return delta.days

def getBidAskMid(bid, ask):
    mid = (bid+ask)/2
    
    return mid

def getUniqueDifference(array1, array2):
    """returns difference in unique values of two arrays"""
    unique1 = np.unique(array1)
    unique2 = np.unique(array2)
    
    difference = list(set(unique1) - set(unique2))
    
    return difference

def removeExcess(dataframe, excess):
    """removes missing dates"""
    for i in excess:
        dataframe = dataframe.drop(dataframe[dataframe.date == i].index)
        
    return dataframe

def matchWithDF(array, df):
    """matches array with length of main dataframe by date"""
    unique = np.unique(df["date"], return_counts=True)
    matchedList = []
    
    for i in range(len(unique[1])):
        for j in range(unique[1][i]):
            matchedList.append(array[i])
    
    return np.array(matchedList)

def getTermStructure(df):
    unique = np.unique(df["date"])
    term_structure = []
    
    for i in unique:
        term_structure.append(df[df.date == i])
        
    return term_structure
    
def getLinearSplineOnTermStructure(term_structure):
    spline = []
    
    for i in term_structure:
        spline.append(interp1d(i["years"], i["rate"], kind="linear", bounds_error=False, fill_value=i["rate"][0]))
    
    return spline

def getInterpolation(spline, df):
    count = 0
    unique = np.unique(df["date"], return_counts = True)
    interpolation = []
    
    for i in range(len(unique[1])):
        for j in range(unique[1][i]):
            interpolation.append(spline[i](df["ttm"][count]))
            count = count + 1
            
    return np.array(interpolation)

def getIndividualCrossSection(df):
    unique = np.unique(df["ttm"])
    cross_section = []
    
    for i in unique:
        cross_section.append(df[df.ttm == i])
        
    return cross_section


def getCrossSections(listOfDataframes):
    cross_sections = []
    for i in listOfDataframes:
        cross_sections.append(getIndividualCrossSection(i))
    
    cross_sections = flatten(cross_sections)
    
    return cross_sections

def DleftFormula(c, K, i):
    left = (float(c[i+1]) - float(c[i]))/(float(K[i+1]) - float(K[i]))
    
    return left

def DrightFormula(c, K, i):
    right = (float(c[i]) - float(c[i-1]))/(float(K[i]) - float(K[i-1]))
    
    return right

def removeDoubles(cross_section):
    spline = interp1d(cross_section["strike_price"], cross_section["call_price"], kind="linear")
    K = np.unique(cross_section["strike_price"])
    c = spline(K)
    
    return c, K

def getTheD(cross_section):
    cross_section = cross_section.sort_values("strike_price")
    cross_section = cross_section.reset_index(drop=True)
    c, K = removeDoubles(cross_section)
    D = []
    
    for i in range(1, len(K)-1):
        D.append(DleftFormula(c, K, i) - DrightFormula(c, K, i))
        
    return np.array(D)

def getDs(cross_sections):
    """returns list of list of Ds as seen in Andersen et al. (2014)"""
    Ds = []
    count = 0
    
    for i in cross_sections:
        Ds.append(getTheD(i))
        print(count)
        count = count + 1
        
    return Ds

def getNonConvexityMeasure(Ds):
    """gets non-convexity measure as seen in Andersen et al. (2014)"""
    NC = []
    
    for i in Ds:
        NC.append(-np.minimum(i, 0))     
    
    return NC

def getMean(listofarrays):
    mean = []
    
    for i in listofarrays:
        mean.append(i.mean())
        
    return np.array(mean)

def getFilterIndices(aggregate_nc):
    index = []
    count = 0
    
    for i in aggregate_nc:
        if i > 0.1:
            index.append(count)
        elif np.logical_not(~np.isnan(i)):
            index.append(count)
        count = count + 1
    return index
    
def nonConvexityFilter(indices, cross_sections):
    for index in sorted(indices, reverse = True):
        del cross_sections[index]
        
    return cross_sections

def matchrAndT(df, T):
    dic = {}
    for i in range(len(df)):
        dic[df["ttm"][i]] = df["r"][i]
        
    r = []
    for i in T:
        r.append(dic[i])
    
    return np.array(r)

def getMktPrice(df, K, T):
    nk = len(K)
    nt = len(T)
    count = 0
    #init matrix
    MktPrice = np.zeros((nk, nt))
    for k in range(nk):
        for t in range(nt):
            if (K[k] == df["strike_price"][count] and T[t] == df["ttm"][count]):
                MktPrice[k, t] = df["call_price"][count]
                if count < len(df)-1:
                    count = count + 1
            else:
                MktPrice[k, t] = np.nan
            
    
    return MktPrice
            

def formatData(df):
    df = df.sort_values(["strike_price", "ttm"])
    df = df.reset_index(drop=True)
    K = np.unique(df["strike_price"])
    T = np.unique(df["ttm"])
    q = df["q"][0]/100
    S = df["close"][0]
    rf = matchrAndT(df, T)/100
    MktPrice = getMktPrice(df, K, T)
    
    return S, rf, q, MktPrice, K, T

def getFormattedData():
    df = pd.read_csv("C:/Users/Magnus/OneDrive - Nordic Biotech Advisors ApS/SPXOptionsFiltered.csv")
    bydate = getTermStructure(df)
    data = []
    for i in bydate:
        data.append(formatData(i))
        
    return tuple(data)

def getTrainAndTest4MLP(data, training_size):
    test = []
    train = []

    count = 0
    
    intermediate_train = []
    
    for i in data:
        count += 1
        if count % (training_size + 1) == 0:
            test.append(i)
            train.append(np.concatenate(intermediate_train))
            intermediate_train = []

        else:
            intermediate_train.append(i)
            
    return tuple(train), tuple(test)

def getDataForMLP(df, training_size):
    intermediate_data = getTermStructure(df)
    data = []
    
    for i in intermediate_data:
        intermediate_list = []
        i = i.reset_index(drop=True)
        for j in range(len(i)):
            intermediate_list.append(np.array([i["strike_price"][j], i["ttm"][j], i["q"][j], i["close"][j], i["r"][j], i["call_price"][j]]))
        data.append(np.array(intermediate_list))
        
    training_data, test_data = getTrainAndTest4MLP(tuple(data), training_size)
    
    return training_data, test_data

def getRBDData(training_size):
    print("--- Getting training data ---")
    df = readData("filteredoptions")
    training_data, test_data = getDataForMLP(df, training_size)

    return (training_data, test_data)

def addC5Data(data):
    def round_down(num, divisor):
        return num - (num%divisor)
    divisor = 200
    virtual = []
    unique = np.unique(data[:, 3])
    for i in unique:
        array = np.zeros((int(round_down(i, divisor)/divisor), 6))
        array[:, 0] = np.linspace(divisor, round_down(i,divisor), int(round_down(i, divisor)/divisor))
        array[:, 3] = np.full(int(round_down(i, divisor)/divisor), i)
        array[:, 5] = array[:, 3] - array[:, 0]
        virtual.append(array)

    virtual = np.concatenate(virtual, 0)
    return np.concatenate((data, virtual), 0)

def getTtmToRDic(data, ttm):
    dic = {}
    for i in range(len(data)):
        dic[data[:, 1][i]] = data[:, 4][i]
    
    return dic

def addC6Data(data):
    unique_ttm = np.unique(data[:, 1])
    unique_S = np.unique(data[:, 3])
    r_dic = getTtmToRDic(data, unique_ttm)
    virtual = []
    count = 0
    for i in unique_ttm:
        count += 1
        if count % 5 == 0:
            array = np.zeros(6)
            array[1] = i
            array[3] = np.random.choice(unique_S)
            array[4] = r_dic[i]
            array[5] = array[3]
            virtual.append(array)
        
    virtual = np.array(virtual)
    return np.concatenate((data, virtual), 0)
        

def addVirtualOptions(data):
    condition_5 = addC5Data(data)
    condition_6 = addC6Data(condition_5)
    
    return condition_6