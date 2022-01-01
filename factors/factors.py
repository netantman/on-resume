import datetime as dt
import pandas as pd
import numpy as np
import boto3,io
# import bloomberg.ds.contrib.remoteio as RemoteIO
from io import StringIO
import datetime
import pyarrow.parquet as pq
import s3fs
import argparse
import sys
import os
import pytz
from constants import BLACKLIST_DATES

from joblib import Parallel, delayed

os.environ['https_proxy'] = 'http://bproxy.tdmz1.bloomberg.com:85'
os.environ['http_proxy'] = 'http://bproxy.tdmz1.bloomberg.com:85'
os.environ['no_proxy'] = '127.0.0.1,169.254.169.254,s3.dev.obdc.bcs.bloomberg.com,s3.dev.rrdc.bcs.bloomberg.com,bcpc.bloomberg.com,bcc.bloomberg.com,bdns.bloomberg.com,cloud.bloomberg.com,dev.bloomberg.com,dx.bloomberg.com,inf.bloomberg.com,localhost,localhost,prod.bloomberg.com'

def get_bep_available_dates(use_cache=False):
    if use_cache:
        res = pd.read_csv('/home/hzhong30/meta-pricing-research/factors/data/dates.csv')
        res = res['dates']
        res = pd.to_datetime(res).sort_values()
        return res
    
    prefix = 'Corporate Bonds'
    session = boto3.Session(profile_name='bvaleval-ob-prod')
    
    s3r = session.resource('s3',endpoint_url = 'http://s3.prod.obdc.bcs.bloomberg.com')
    bucket = s3r.Bucket('bval-bep-prod')
    dates=[]
    for obj in bucket.objects.filter(Prefix=prefix):
        if 'NY_4PM' in obj.key:
            dates.append(obj.key.split("/")[1])
    
    s3r = session.resource('s3',endpoint_url = 'http://s3.prod.rrdc.bcs.bloomberg.com')
    bucket = s3r.Bucket('bval-bep-prod')
    for obj in bucket.objects.filter(Prefix=prefix):
        if 'NY_4PM' in obj.key:
            dates.append(obj.key.split("/")[1])
    
    dates=list(d for d in set(dates) if d not in BLACKLIST_DATES)
    res=pd.Series(dates, name='dates')
    res=pd.to_datetime(res).sort_values()
    res.to_csv('/home/hzhong30/meta-pricing-research/factors/data/dates.csv', index=False)
    return res

def get_bep(date_short, corp_or_govt='corp', verbose=False, usecols=None, extra_filters=None):
    corp_or_govt=corp_or_govt.lower()
    assert corp_or_govt in ['corp', 'govt'], f"corp_or_govt has to be corp or govt"
    prefix = 'Corporate Bonds' if corp_or_govt=='corp' else 'Sovereign Bonds'
    prefix = f'{prefix}/{date_short}'
    if verbose:
        print(prefix)
    session = boto3.Session(profile_name='bvaleval-ob-prod')
    endpoint_url = 'http://s3.prod.obdc.bcs.bloomberg.com' if dt.datetime.strptime(date_short, '%Y%m%d') <= dt.datetime.strptime('20210329', '%Y%m%d') else 'http://s3.prod.rrdc.bcs.bloomberg.com'
    s3r = session.resource('s3', endpoint_url = endpoint_url)
    bucket = s3r.Bucket('bval-bep-prod')
    if verbose:
        print(list(bucket.objects.filter(Prefix=prefix)))
    trunks=[]
    bep=None
    for obj in bucket.objects.filter(Prefix=prefix):
        if 'NY_4PM' not in obj.key:
            continue
        if verbose:
            print(obj)
        body=obj.get()['Body']
        bep=pd.read_csv(body, usecols=usecols, skiprows=2, header=0,encoding="ISO-8859-1", na_values=["N.A.", 'NaNS', 'NaN', ""])#, low_memory=False)
        bep=bep.set_index('Cusip')
        if extra_filters:
            for k, v in extra_filters.items():
                bep = bep[bep[k]==v]
        break
#         trunk=trunk.set_index('Cusip')
#         trunks.append(trunk)
#     bep=pd.concat(trunks)
        
    return bep

def universe_of_cusips(use_cache=False):
    if use_cache:
        res = pd.read_csv('/home/hzhong30/meta-pricing-research/factors/data/universe_cusips.csv', index_col='Cusip')
        res = res[['dummy']]
        return res
    
    date_str="20190828"
    bep=get_bep(date_str, usecols=['Cusip', 'Security ID', 'Maturity', 'Coupon Type', 'Currency', 'Billing Class', 'Debt Type Enhanced', 'Market Issue', 'Still Callable'])

    tmp=bep['Maturity'].str.split('/', expand=True)
    tmp=pd.to_numeric(tmp[2])
    bep=bep[(~pd.isnull(tmp))&(tmp<3000)] # remove year==3000

    date=dt.datetime.strptime(date_str, "%Y%m%d")
    bep['Maturity']=pd.to_datetime(bep['Maturity'], format='%m/%d/%Y')
    bep['yrs_to_maturity']=(bep['Maturity']-date).dt.days/365
    bep=bep[(bep['yrs_to_maturity']>=7.0)&(bep['yrs_to_maturity']<=22.0)]
    
    bep=bep[bep['Coupon Type'].isin(['FIXED'])]
    bep=bep[bep['Currency']=='USD']
    bep=bep[bep['Billing Class'].isin(['High Yield Corporate', 'Investment Grade Corporate'])]
    bep=bep[~bep['Market Issue'].isin(['PRIV PLACEMENT', 'PRIVATE'])]
    bep=bep[bep['Debt Type Enhanced'].str.upper()=='SENIOR']
    bep=bep[bep['Still Callable']=="N"]
    bep['dummy']='dummy'
    bep=bep[['dummy']]
    
    bep.to_csv('/home/hzhong30/meta-pricing-research/factors/data/universe_cusips.csv', index=True)
    
    return bep

def process_char(char, dates, reverse=True, rename_cols=True, reindex_by_date=True):
    
    if reindex_by_date:
        char=char.reindex([d.strftime('%Y-%m-%d') for d in dates])
    char=char.fillna(method='ffill')
    char=char.fillna(method='bfill')
    char=char.dropna(how='all', axis=1) # 00774MAB Corp
    if reverse:
        char=(-1)*char # so that it is bottom-top later
    if rename_cols:
        columns={c:c[:8] for c in char.columns}
        char=char.rename(columns=columns)
    return char

def compute_long_short(date=None, future_date=None, GBM_of_date=None, char_of_date=None, number_bins=10, use_mid=True, usecols=None, universe=None, char_name=None, extra_filters=None):

    bid_px_col, ask_px_col, mid_px_col = 'Bid Price', 'Ask Price', 'Bid Price'
    holding_days=(future_date-date).days
    date_short=date.strftime('%Y%m%d')
    future_date_short=future_date.strftime('%Y%m%d')
#     print(f"Processing: {date_short}...")
    raw_bep=get_bep(date_short, usecols=usecols, extra_filters=extra_filters)
    raw_bep=raw_bep.merge(universe, how='inner', left_index=True, right_index=True)
    
    bep=raw_bep.merge(char_of_date, how='inner', left_index=True, right_index=True)
    if date in bep.columns:
        bep = bep.rename(columns={date: char_name})
    elif date.strftime("%Y-%m-%d") in bep.columns:
        bep=bep.rename(columns={date.strftime("%Y-%m-%d"): char_name})
    else:
        raise Exception(f'{bep.columns}')
#     print(bep.columns)
    bep['qcut']=pd.qcut(bep[[char_name]].rank(method='first')[char_name], number_bins, labels=[str(idx) for idx in range(10)], duplicates='drop')
    top, bottom=bep[bep['qcut']==str(number_bins-1)], bep[bep['qcut']=='0']
#     print(top[char_name].mean())
#     print(bottom[char_name].mean())
    
    try:
        if not use_mid:
            future_bep=get_bep(future_date_short, usecols=['Cusip', bid_px_col, ask_px_col])
            future_bep=future_bep.rename(columns={bid_px_col: f"Future {bid_px_col}", ask_px_col: f"Future {ask_px_col}"})
        else:
            future_bep=get_bep(future_date_short, usecols=['Cusip', mid_px_col])
            future_bep=future_bep.rename(columns={mid_px_col: f"Future {mid_px_col}"})
    except Exception:
        print(future_date_short)
        raise Exception(f"{future_date_short}")
    top=top.merge(future_bep, how='left', left_index=True, right_index=True)
    bottom=bottom.merge(future_bep, how='left', left_index=True, right_index=True)
    if not use_mid:
        top['pnl']=(top[f'Future {bid_px_col}']-top[ask_px_col]+top['Coupon']*holding_days/365.)/top[ask_px_col]
        bottom['pnl']=(bottom[f'Future {ask_px_col}']-bottom[bid_px_col]+bottom['Coupon']*holding_days/365.)/bottom[bid_px_col]
    else:
        top['pnl']=(top[f'Future {mid_px_col}']-top[mid_px_col]+top['Coupon']*holding_days/365.)/top[mid_px_col]
        bottom['pnl']=(bottom[f'Future {mid_px_col}']-bottom[mid_px_col]+bottom['Coupon']*holding_days/365.)/bottom[mid_px_col]
    
    output=[
        {
            'date': date, 
            'duration_diff': top['Duration'].mean() - bottom['Duration'].mean(), 
            'pnl': top['pnl'].mean() - bottom['pnl'].mean(),
            'duration-neutral-pnl': (top['pnl'].mean() - GBM_of_date/12.0/100.0) * bottom['Duration'].mean() - (bottom['pnl'].mean() - GBM_of_date/12.0/100.0) * top['Duration'].mean(),
            'long-only': top['pnl'].mean()
        }
    ]
    output=pd.DataFrame(output)
    output.set_index('date', inplace=True)
    return output

def produce_plots_given_res(res, char_name):
    res[['duration_diff', 'zero']].plot(figsize=(15, 10), title=char_name)
    res[['pnl', 'long-only', 'duration-neutral-pnl', 'zero']].plot(figsize=(15, 10), title=char_name)
    print('long-short summary')
    print(res['pnl'].describe())
    print(res['pnl'].mean()/res['pnl'].std())
    print("")
    print('duration-neutral long-short summary')
    print(res['duration-neutral-pnl'].describe())
    print(res['duration-neutral-pnl'].mean()/res['duration-neutral-pnl'].std())
    print("")    
    print('long-only summary')
    print(res['long-only'].describe())
    print(res['long-only'].mean()/res['long-only'].std())

def produce_plots(dates, char, GBM, use_mid = True, number_bins=10, lookforth=20, usecols=[], universe=None, char_name='', extra_filters=None):
    L = len(dates)
    curr_future_dates_char = [(dates[idx], dates[idx+lookforth], char.loc[dates[idx].strftime("%Y-%m-%d")]) for idx in range(L-lookforth)]
    curr_future_dates_char = [tu for tu in curr_future_dates_char if not pd.isnull(tu[2]).all()]
    out = Parallel(n_jobs=8)(delayed(compute_long_short)(date=date, future_date=future_date, GBM_of_date=GBM.loc[date]['GBM Govt'], char_of_date=char_of_date, number_bins=number_bins, use_mid=use_mid, usecols=usecols, universe=universe, char_name=char_name, extra_filters=extra_filters) for (date, future_date, char_of_date) in curr_future_dates_char)
    res = pd.concat(out, axis=0)
    res['zero'] = 0
    produce_plots_given_res(res, char_name)
    res.to_csv(f'/home/hzhong30/meta-pricing-research/factors/output/updated_factors/{char_name}.csv')
    return res

def produce_char_file_from_bep(char_name, bep_col, static=False, use_cache=False):
    if use_cache and os.path.exists(f'/home/hzhong30/meta-pricing-research/factors/data/{char_name}.csv'):
        char = pd.read_csv(f'data/{char_name}.csv')
        char = char.rename(columns={'Unnamed: 0': 'date'})
        char['date'] = pd.to_datetime(char['date'])
        char['date'] = char.apply(lambda r: r['date'].strftime('%Y-%m-%d'), axis=1)
        char = char.set_index('date')
        return char

    char = []
    dates = get_bep_available_dates(use_cache=True)
    dates = dates.tolist()
    universe = universe_of_cusips(use_cache=True)
    raw_bep = None
    for date in dates:
        if not static or raw_bep is None:
            print(date)
            date_short = date.strftime('%Y%m%d')
            raw_bep = get_bep(date_short, usecols=['Cusip', bep_col])
            raw_bep = raw_bep.merge(universe, how='inner', left_index=True, right_index=True)
            raw_bep = raw_bep.drop('dummy', axis=1)
        bep = raw_bep.rename(columns={bep_col: date})
        bep = bep.transpose()
        char.append(bep)
    char = pd.concat(char, axis=0)
    char = process_char(char, dates, reverse=False, rename_cols=False, reindex_by_date=False)
    char.index.name = 'date'
    char.to_csv(f'/home/hzhong30/meta-pricing-research/factors/data/{char_name}.csv', index=True)
    return char
    
def retrieve_res_from_csv(char_name):
    res = pd.read_csv(f'/home/hzhong30/meta-pricing-research/factors/output/updated_factors/{char_name}.csv', index_col='date')
    res.index = pd.to_datetime(res.index, format='%Y-%m-%d')
    return res
    