import time
import os

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import boto3
import io
from io import StringIO, BytesIO
import datetime as dt
from dateutil.parser import parse
from dateutil import tz
import pytz
est = pytz.timezone('US/Eastern')

import logging
import warnings
warnings.filterwarnings('ignore')

class gap:
    def __init__(self, secid='', threshold=2, bad_quote_mul=2.5, bid_col='L2_BID', ask_col='L2_ASK', epoch_col='final_epoch', cutoff=dt.datetime(2021, 7, 19, tzinfo=est), long_only=False):
        self.data = None
        self.T = 0
        self.secid = secid
        self.threshold = threshold
        self.bad_quote_mul = bad_quote_mul * threshold
        self.bid_col = bid_col
        self.ask_col = ask_col
        self.epoch_col = epoch_col
        self.mtm = None
        self.capital = None
        self.sell_orders = []
        self.buy_orders = []
        self.cutoff = cutoff
        self.long_only = long_only
       
    def get_data(self, prefix='data/training', filter_market_hours=True):
        file_dir = f'{prefix}/{self.secid}.csv'
        if not os.path.exists(file_dir):
            return self.T
        data = pd.read_csv(file_dir, parse_dates=['timestamp'])
        if filter_market_hours:
            self.data = data.loc[(data.timestamp.dt.hour <= 21)&(data.timestamp.dt.hour >= 14)]
        data.set_index('timestamp', inplace=True)
        self.data = data.loc[data.index <= self.cutoff, [self.bid_col, self.ask_col, self.epoch_col]]
        self.T = len(data)
        return self.T
    
    def update_bas(self, old_bas, new_bas, time_elapsed, decay=1 / 1500):
        if old_bas is None:
            return new_bas
        decay = np.exp(- time_elapsed * decay)
        return (new_bas + decay * old_bas) / (1 + decay)
    
    def plot_portfolio_mtm(self):
        if self.mtm is None:
            _ = self.get_portfolio_mtm_and_capital()
        
        self.mtm['zero'] = 0
        self.mtm.plot(figsize=(15, 10))
        
    def get_capital(self):
        if self.capital is None:
            _ = self.get_portfolio_mtm_and_capital()
        
        return self.capital
    
    def plot_capital(self):
        if self.capital is None:
            _ = self.get_portfolio_mtm_and_capital()
        
        self.capital['zero'] = 0
        self.capital.plot(figsize=(15, 10))
    
    def order_duration(self, plot=True):
        if self.mtm is None:
            _ = self.get_portfolio_mtm_and_capital()

        if not self.long_only:
            buy_durations_hours = [(bo[5] - bo[2]) / 3600 for bo in self.buy_orders if bo[5] is not None]
            sell_durations_hours = [(so[5] - so[2]) / 3600 for so in self.sell_orders if so[5] is not None]
            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].hist(buy_durations_hours, bins=50)
                ax[0].set_title('Buy order durations (hrs)')
                ax[1].hist(sell_durations_hours, bins=50)
                ax[1].set_title('Sell order durations (hrs)')

                plt.show()
            else:
                b = max(buy_durations_hours) if buy_durations_hours else np.nan
                s = max(sell_durations_hours) if sell_durations_hours else np.nan
                return (b, s)
            
        buy_durations_hours = [(bo[5] - bo[2]) / 3600 for bo in self.buy_orders if bo[5] is not None]
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax[0].hist(buy_durations_hours, bins=50)
            ax[0].set_title('Buy order durations (hrs)')
            plt.show()
        else:
            b = max(buy_durations_hours) if buy_durations_hours else np.nan
            return b
        
    def get_sharpe(self, periods=1000):
        if self.mtm is None:
            _ = self.get_portfolio_mtm_and_capital()
        
        chg = self.mtm['mtm'].diff(periods=periods)
        try:
            res = chg.mean()/chg.std()
        except:
            res = np.nan
        return res
    
    def get_drawdown(self, periods=2500):
        if self.mtm is None:
            _ = self.get_portfolio_mtm_and_capital()
        
        chg = self.mtm['mtm'].diff(periods=periods)
        return chg.min(skipna=True)        
        
    def get_cumulative_gain(self):
        '''dud'''
        if not self.sell_orders and not self.buy_orders:
            _ = self.get_portfolio_mtm_and_capital()
            
        gain = 0
        gain += sum([bo[4] - bo[1] for bo in self.buy_orders])
        gain += sum([so[1] - so[4] for so in self.sell_orders])
        return gain
    
    def get_portfolio_mtm_and_capital(self):
        assert self.T > 0, "No data"
        if self.mtm is not None:
            return self.mtm
        
        capital = 0
        bas = None
        last_bid, last_ask = None, None
        last_epoch = None
        buy_orders, sell_orders = [], []
        self.mtm, self.capital = [], []
        for timestamp, row in self.data.iterrows():
            if bas is None:
                bas = row[self.ask_col] - row[self.bid_col]
                last_bid, last_ask = row[self.bid_col], row[self.ask_col]
                last_epoch = row[self.epoch_col]
                self.mtm.append({"timestamp": timestamp, "mtm": 0})
                self.capital.append({"timestamp": timestamp, "capital": 0})
                continue
            
            mtm = 0
            if not self.long_only:
                for so in self.sell_orders:
                    mtm += (so[1] - row[self.ask_col]) if so[4] is None else (so[1] - so[4])
                    if so[3] is None and so[1] >= row[self.ask_col]: # buying back
                        so[4] = row[self.ask_col]
                        so[3] = timestamp
                        so[5] = row[self.epoch_col]
                        capital -= so[1]
            for bo in self.buy_orders:
                mtm += (row[self.bid_col] - bo[1]) if bo[4] is None else (bo[4] - bo[1])
                if bo[3] is None and bo[1] <= row[self.bid_col]: # selling
                    bo[4] = row[self.bid_col]
                    bo[3] = timestamp
                    bo[5] = row[self.epoch_col]
                    capital += bo[1]
                              
            time_elapsed = row[self.epoch_col] - last_epoch
            if not self.long_only:
                if (row[self.bid_col] - last_bid > self.threshold * bas) and (row[self.bid_col] - last_bid <= self.bad_quote_mul * bas):
                    self.sell_orders.append([timestamp, row[self.bid_col], row[self.epoch_col], None, row[self.ask_col], None])
                    mtm += (row[self.bid_col] - row[self.ask_col])
                    capital += row[self.bid_col]
            if (row[self.ask_col] - last_ask < - self.threshold * bas) and (row[self.ask_col] - last_ask >= - self.bad_quote_mul * bas):
                self.buy_orders.append([timestamp, row[self.ask_col], row[self.epoch_col], None, row[self.bid_col], None])
                mtm += (row[self.bid_col] - row[self.ask_col])
                capital -= row[self.ask_col]
            self.mtm.append({"timestamp": timestamp, "mtm": mtm})
            self.capital.append({"timestamp": timestamp, "capital": capital})
            last_bid, last_ask = row[self.bid_col], row[self.ask_col]
            bas = self.update_bas(bas, row[self.ask_col] - row[self.bid_col], time_elapsed)
            last_epoch = row[self.epoch_col]
        
        self.mtm = pd.DataFrame(self.mtm)
        self.mtm['timestamp'] = pd.to_datetime(self.mtm['timestamp'])
        self.mtm.set_index('timestamp', inplace=True)
        self.capital = pd.DataFrame(self.capital)
        self.capital['timestamp'] = pd.to_datetime(self.capital['timestamp'])
        self.capital.set_index('timestamp', inplace=True)
        return self.mtm, self.capital
        

def gap_strategy_for_one_sec(secid, long_only=False):
    res = {"secid": secid}
    training = gap(secid=secid, threshold=2, long_only=long_only)
    T = training.get_data()
    if T < 1:
        res["sharpe"] = np.nan
        res["drawdown"] = np.nan
        res["buy_max_duration"] = np.nan
        res["sell_max_duration"] = np.nan
        res = pd.DataFrame([res])
        return res
    
    training.get_portfolio_mtm_and_capital()
    res["sharpe"] = training.get_sharpe()
    res["drawdown"] = training.get_drawdown()
    if not long_only:
        res["buy_max_duration (hrs)"], res["sell_max_duration (hrs)"] = training.order_duration(plot=False)
    else:
        res["buy_max_duration (hrs)"] = training.order_duration(plot=False)
    res = pd.DataFrame([res])
    
    return res