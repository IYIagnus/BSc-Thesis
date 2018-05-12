import numpy as np
from numpy.fft import fft, ifft

def hestonCF(phi, kappa, theta, laambda, rho, sigma, tau, S, r, q, v0):
    """Little trap formulation of the Heston characteristic function, f2 (Rouah, 2013, pp. 31-33)"""
    x = np.log(S)
    a = kappa*theta
    u = -0.5
    z = 1j
    b = kappa+laambda
    d = np.sqrt((rho*sigma*z*phi - b)**2 - sigma**2 * (2*u*z*phi - phi**2))
    g = (b - rho*sigma*z*phi + d) / (b - rho*sigma*z*phi - d)
    
    c = 1/g
    G = (1 - c*np.exp(-d*tau)) / (1 - c)
    C = (r - q)*z*phi*tau + a/sigma**2*((b - rho*sigma*z*phi - d)*tau - 2*np.log(G))
    D = (b - rho*sigma*z*phi - d)/sigma**2*((1 - np.exp(-d*tau))/(1 - c*np.exp(-d*tau)))
    
    f2 = np.exp(C + D*v0 + z*phi*x)
    
    return f2

def FRFT(x, beta):
    """Fractional fast Fourier transform (Rouah, 2013, pp. 141-145)
    Input:
        x is a vector of length N
        beta is the increment parameter"""
    N = len(x)
    i = 1j
    
    #Contruct y and z vectors
    y = np.concatenate((np.exp(-i*np.pi*np.arange(N)**2*beta)*x , np.zeros(N)))
    z = np.concatenate((np.exp( i*np.pi*np.arange(N)**2*beta), np.exp(i*np.pi*np.arange(N, 0, -1)**2*beta)))
    
    #FFT on y and z
    Dy = fft(y)
    Dz = fft(z)
    
    #h vectors
    hhat = Dy*Dz
    h = ifft(hhat)
    
    #e vector
    e = np.concatenate((np.array(np.exp(-i*np.pi*np.arange(N)**2*beta)) , np.zeros(N)))
    
    xhat = e*h
    return xhat[:N]


def getOptimalAlpha(S0, r, K, tau, rho, v0, theta, kappa, sigma, laambda):
    """NOT IMPLEMENTED YET"""
    return optimalAlpha


def hestonCallPriceFRFT(N, S0, r, q, tau, kappa, theta, laambda, rho, sigma, v0, alpha, eta, lambdainc):
    """Calculates Heston call prices by fractional fast Fourier transform (Rouah, 2013, pp. 141-145) based on the 
    Carr and Madan representation (Rouah, 2013, pp. 73-89) using Simpson's rule (Rouah, 119-120) and the little trap formulation
    of the Heston characteristic function, f2 (Rouah, 2013, pp. 31-33)
    Input:
        N = number of discretization points
        S0 = spot price
        r = risk-free rate
        q = continuously compounded dividend yield
        tau = time to maturity
        sigma = volatility
        alpha = dampening factor
        eta = integration range increment
        lambdainc = strike range increment"""
    
    #Log spot price
    s0 = np.log(S0)
    
    #Init and specify weights
    w = np.zeros(N)
    w[0] = 1/3
    w[N-1] = 1/3
    for z in range(1, N-1):
        if (z+1)%2==0:
            w[z] = 4/3
        else:
            w[z] = 2/3
                        
    #specify the b parameter
    b = N*lambdainc/2
    
    #create the grid for the integration
    v = eta*np.arange(N)
    
    #create the grid for the log-strikes
    k = lambdainc*np.arange(N) + s0 - b
    
    #create vector of strikes
    K = np.exp(k)
    
    #Specify beta for FRFT
    beta = lambdainc*eta/2/np.pi
    
    ##Find optimal alpha (WIP)
    #alpha = getOptimalAlpha()
    
    #Implement FRFT
    i = 1j
    f2 = hestonCF(v-(alpha+1)*i, kappa, theta, laambda, rho, sigma, tau, S0, r, q, v0)
    psi = (np.exp(-r*tau)*f2)/(alpha**2 + alpha - v**2 + i*v*(2*alpha+1))
    x = np.exp(i*(b-s0)*v)*psi*w
    xhat = FRFT(x, beta)
    y = np.real(xhat)
    CallPrices = eta*np.exp(-alpha*k)*y/np.pi
    
    return CallPrices, K, lambdainc, eta