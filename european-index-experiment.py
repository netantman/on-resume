import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import *

def drop_n_rescale(W, to_drop):
    for c in to_drop:
        W=W.drop(c,axis=1)
        W=W.drop(c,axis=0)
    res=W.copy()
    for idx, row in W.iterrows():
        res.loc[idx]=row/row.sum()
    return res

def rescale_W(W, to_drop):
    W=drop_n_rescale(W,['Ireland'])
    countries=W.index
    assert (countries==W.columns).all()
    N,N1=W.shape
    assert N==N1
    return W, countries, N

def read_data(file, rename_mapping):
    data=pd.read_csv(file, index_col=0)
    data=data.interpolate()
    data.index=pd.to_datetime(data.index)
    data=data.rename(columns={v:k for k,v in rename_mapping.items()})
    
def make_percentage(Y, start=START, freq=FREQ, verbose=True):
    Y=Y.iloc[start::freq]
    Y=Y.pct_change()
    Y=Y.fillna(value=0)
    Y=Y*100
    return Y

def sar_logdet(rhos, W, T):
    '''
    rho: a vector containing the values of rho evaluated at 
         which the log determinants are computed
    W: the NxN spatial matrix, not the stacked form
    T: the length of the time series
    '''
    
    N,N1=W.shape
    assert N==N1
    logdets=[]
    for rho in rhos:
        matrix=np.dot(np.eye(N)-rho*W.T, np.eye(N)-rho*W)
        logdet=np.log(np.linalg.det(matrix))
        logdets.append(T*logdet/2)
    return {r:l for r,l in zip(rhos, logdets)}

def sar_mle(rho_logdets,y,X,W_out):
    nT=len(y)
    maxx=None
    rho_hat=None
    XX=np.linalg.inv(np.dot(X.T, X))
    XXX=np.dot(XX, X.T)
    for rho, logdet in rho_logdets.items():
        y_adj=np.dot((np.eye(nT)-rho*W_out), y)
        b_temp=np.dot((XXX, y_adj))
        e_temp=y_adj-np.dot(X,b_temp)
        sigmasqr_temp=np.dot(e_temp, e_temp)/nT
        loglik=logdet-(nT/2)*np.log(sigmasqr_temp)-(nT/2)-(nT/2)*np.log(2*np.pi)
        if not maxx or maxx<loglik:
            rho_hat, maxx=rho, loglik
    
    y_adj_true=np.dot((np.eye(nT)-rho_hat*W_out), y)
    b_hat=np.dot(XXX, y_adj_true)
    e_hat=y_adj_true-np.dot(X, b_hat)
    sigmasqr_hat=np.dot(e_hat.T, e_hat)/nT
    
    return rho_hat, b_hat, sigmasqr_hat

def s_apt_all(Y,G,W, verbose=False):
    '''
    This function provides the M.L.E. for rho, alpha, B and sigma.
    The stacking is different from Matlab implementation, but closer 
    to the paper.
    
    inputs:
    Y: an NxT matrix, the returns of assets. Here T is the length of 
       the dataset, N is the number of assets
    G: an KxT matrix, where K is the number of factors
    W: spatial weight matrix, NxN
    '''
    
    Y,G,W=Y.values,G.values,W.values
    N,T=np.shape(Y)
    K,T1=np.shape(G)
    assert T==T1
    if verbose:
        print(f"N: {N}")
        print(f"K: {K}")
        print(f"T: {T}")
    W_out=np.kron(np.eye(T), W)
    r=Y
    y=np.reshape(np.transpose(r), (-1,1))
    
    X=np.zeros((N*T, N*(K+1)))
    for t in range(T):
        Gt=np.ones(K+1)
        Gt[1:]=G[:,t]
        comp=np.kron(Gt.T, np.eye(N))
        X[t*N:(t+1)*N, :]=comp
        if verbose:
            print(f"X component: {comp}")
    
    rhos=list(np.arrange(0, 0.95, 0.01))
    rho_logdets=sar_logdet(rhos, W, T)
    
    rho_hat, b_hat, sigmasqr_hat=sar_mle(rho_logdets,y,X,W_out)
    
    return rho_hat, b_hat, sigmasqr_hat

def sapt_gen(N,T,W,K):
    
    rho=0.5
    sigma=0.3
    sigmasqr=sigma**2
    B=np.zeros((N,K))
    for n in range(N):
        for k in range(K):
            B[n,k]=n+1+(k+1)*0.1
    
    G=np.random.randn(K,T)*0.7+0.019
    epsilon=np.random.randn(N,T)*sigma
    Y=np.dot(np.linalg.inv(np.eye(N)-rho*W), np.dot(B,G)+epsilon)
    return Y,G,B

def sapt_mass(lookback,Y,G,W,factors,top_positive=3,top_negative=3):
    start=0
    K=len(G.columns)
    countries=W.index
    N,N1=W.shape
    assert N==N1
    
    in_sample_sharpes=[]
    rhos_hat=[]
    sigmas_hat=[]
    pnls=[]
    while start+lookback<len(Y):
        end=start+lookback
        Y_subset=Y.iloc[start:end].transpose()
        G_subset=G.iloc[start:end].transpose()
        rho_hat, b_hat, sigmasqr_hat=s_apt_all(Y_subset,G_subset,W)
        loadings=np.shape(b_hat.T, (K+1,-1))
        loadings=pd.DataFrame(loadings, columns=countries, index=['alpha']+factors)
        sigma_hat=np.sqrt(sigmasqr_hat[0][0])
        alpha_sorted=loadings.loc['alpha'].sort_values()
        in_sample_sharpe=(alpha_sorted[N-top_positive:].sum()-alpha_sorted[:top_negative].sum())/sigma_hat
        Y_next=Y.iloc[end]
        tmp=pd.Series(data=rho_hat*np.dot(W.values, Y_next.values), index=countries)
        residual=Y_next-tmp
        G_next=G.iloc[end]
        for f in factors:
            residual=residual-loadings.loc[f]*G_next[f]
        pnl=0
        for c in alpha_sorted.index[:top_negative]:
            pnl-=residual[c]
        for c in alpha_sorted.index[top_positive:]:
            pnl+=residual[c]
        in_sample_sharpes.append(in_sample_sharpe)
        rhos_hat.append(rho_hat)
        sigmas_hat.append(sigma_hat)
        pnls.append(pnl)
        start+=1
    return in_sample_sharpes, rhos_hat, sigmas_hat, pnls

class Sort_Portfolio:
    def __init__(self, char, apply_log=False):
        self.char=char
        if apply_log:
            self.char=np.log(self.char)
        self.long_short=None
    
    def sort_characteristics(self, num_extreme, lookback):
        def inner(row):
            date=row.name
            sorting=self.char.loc[(self.char.index>=date-lookback)&(self.char.index<=date)].sum()
            sorted_row=sorting.sort_values(ascending=True)
            bottom, top=sorted_row.index[:num_extreme], sorted_row.index[-num_extreme:]
            return pd.Series([bottom, top])
        return inner
    
    def make_sort(self, num_extreme=1, lookback=dt.timedelta(days=100)):
        sort_func=self.sort_characteristics(num_extreme, lookback)
        self.char[['bottom', 'top']]=self.char.apply(sort_func, axis=1)
       
    def get_char(self):
        return self.char
    
    def plot_char(self, size=10, logy=False):
        self.char.plot(figsize=(2*size, size), logy=logy)
    
    def form_portfolio(self, px, num_extreme=1, lookback=dt.timedelta(days=180), lookforth_period_in_price=10):
        self.make_sort(num_extreme=num_extreme, lookback=lookback)
        L=len(px)
        long_short=[]
        for idx in range(L-lookforth_period_in_price):
            excess_return=0
            date, next_date=px.index[idx], px.index[idx+lookforth_period_in_price]
            bottom, top=self.char.loc[date, 'bottom'], self.char.loc[date, 'top']
            for b in bottom:
                excess_return+=(px.loc[next_date,b]-px.loc[date,b])/px.loc[date, b]*100
            for t in top:
                excess_return-=(px.loc[next_date,t]-px.loc[date,t])/px.loc[date, t]*100
        self.long_short=pd.DataFrame(long_short)
        self.long_short=self.long_short.set_index('date')
    
    def get_portfolio(self):
        return self.long_short
    
    def plot_portfolio(self, size=10):
        fig,ax = plt.subplot(1,1, figsize=(size, 1.0*size/2.0))
        ax.plot(self.long_short.index, self.long_short['excess_return'], color='c')
        ax.plot(self.long_short.index, self.long_short['zero'], color='y')
        ax.grid()
        plt.show()























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        