import numpy as np
import matplotlib.pyplot as plt

def GeometricBrownianMotion(S0, mu, sigma, T, dt=0.01):
    N = round(T/dt)
    X = np.random.normal(mu * dt, sigma*np.sqrt(dt), N)
    X = np.cumsum(X)
    S = S0*np.exp(X)
    t = np.linspace(0, T, N)
    
    return S, t
    

for i in range(1000):
    S, t = GeometricBrownianMotion(100, 0.01, 0.1, 2)
    dSdt = []
    for i in range(1,200):
        dSdt.append(S[i]-S[i-1])
        
    plt.plot(t, S)
    plt.show()
    print(np.mean(dSdt))