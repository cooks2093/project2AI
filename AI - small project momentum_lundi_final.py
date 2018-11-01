# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:30:38 2018

@author: Jonathan
@author: Charles 

Project 1: Momentum Strategy 
"""
# Import useful libraries
import pandas as pd
import numpy as np 
import datetime as dt
import math
import statsmodels.api as sm 
from operator import add

# Start time computation counter
a = dt.datetime.now()
print('Start time : ')
print(a)

# Define useful function to make the indexes compatible one another when associating two dataframes
def changeDates(y,z,o):
    x_l = y.iloc[:,y.columns.get_loc(z)].values.tolist()
    x_df = pd.DataFrame(x_l, index = o)
    return x_df

# Define the function that computes portfolio returns
def compute(x,bonus):
    
    # Counter
    u = 0
    
    # Define specific variables for further computations inside the loops
    summcap = 0
    summrkret = 0
    summcap_short = 0
    summkret_short = 0 
    
    # List of the sum of MCap(t-1) of winners for each period
    portfmcap = []
    
    # List of the sum of MCap(t-1)*Return(t) of winners for each period
    portfmrkret = []
    
    # List of the sum of MCap(t-1) of loser for each period
    portfmcap_short = []
    
    # List of the sum of MCap(t-1)*Return(t) of losers for each period
    portfmrkret_short = []
    
    # List of MCap(t-1) of winners for each period
    mcaplist = []
    
    # List of MCap(t-1)*Return(t) of winners for each period
    retlist = []
    
    # List of MCap(t-1) of losers for each period
    mcaplist_short = []
    
    # List of MCap(t-1)*Return(t) of losers for each period
    retlist_short = []
    
    # Loop over the time period (month)
    for i in x.index:
        val = x.loc[i]
        val = val.dropna()
        
        # Loop over the cumulative returns of each period
        for key in val:
            
            # Consider the full range of prices (not for the Bonus Question)
            if bonus == False:
                
                # Sort with the quantile of winners
                if key >= val.quantile(q=0.9,interpolation = 'linear'):
                    
                    # Add MCap(t-1) of the winner to a specific list
                    mcaplist.append(mcap.loc[i,val[val==key].index.values].iloc[0])
                    
                    # Add MCap(t-1)*Return(t) of the winner to a specific list        
                    retlist.append(mktret.loc[i,val[val==key].index.values].iloc[0])
                    
                # Consider short position only if we have more than 1 observation per period 
                if val.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='first').count() > 1: 
                    
                    # Sort with the quantile of losers
                    if(key <= val.quantile(q=0.1,interpolation = 'linear')):
                        
                        # Add MCap(t-1) of the loser to a specific list
                        mcaplist_short.append(mcap.loc[i,val[val==key].index.values].iloc[0])
                        
                        # Add MCap(t-1)*Return(t) of the loser to a specific list                       
                        retlist_short.append(mktret.loc[i,val[val==key].index.values].iloc[0])
                        
            # Consider the range of prices only > $5 (Bonus Question)
            else:
                # Sort with the quantile of winners and stocks with price > $5
                if key >= val.quantile(q=0.9,interpolation = 'linear') and price.loc[i,val[val==key].index.values].iloc[0] > 5:
                    
                    # Add MCap(t-1) of the winner to a specific list
                    mcaplist.append(mcap.loc[i,val[val==key].index.values].iloc[0])
                    
                    # Add MCap(t-1)*Return(t) of the winner to a specific list
                    retlist.append(mktret.loc[i,val[val==key].index.values].iloc[0])
                    
                # Consider short position only if we have more than 1 observation per period 
                if val.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='first').count() > 1: 
                    
                    # Sort with the quantile of losers and stocks with price > $5
                    if(key <= val.quantile(q=0.1,interpolation = 'linear')) and price.loc[i,val[val==key].index.values].iloc[0] > 5:
                        
                        # Add MCap(t-1) of the losers to a specific list
                        mcaplist_short.append(mcap.loc[i,val[val==key].index.values].iloc[0])
                        
                        # Add MCap(t-1)*Return(t) of the losers to a specific list
                        retlist_short.append(mktret.loc[i,val[val==key].index.values].iloc[0])
                        
        # At a specific period in the loop we sum all the MCap of stocks with long position
        summcap = 0
        for j in mcaplist:
            if not math.isnan(j):
                summcap += j
        portfmcap.append(summcap)
        mcaplist = []
    
        # At a specific period in the loop we sum all the MCap(t-1)*Return(t) of stocks with long position    
        summrkret = 0
        for k in retlist:
            if not math.isnan(k):
                summrkret += k
        portfmrkret.append(summrkret)
        retlist = []
        
        # At a specific period in the loop we sum all the MCap of stocks with short position
        summcap_short = 0
        for j in mcaplist_short:
            if not math.isnan(j):
                summcap_short += j
        portfmcap_short.append(summcap_short)
        mcaplist_short = []
    
        # At a specific period in the loop we sum all the MCap(t-1)*Return(t) of stocks with short position    
        summkret_short = 0 
        for k in retlist_short:
            if not math.isnan(k):
                
                # Since we are shorting we have to take the negative value of the return and add them up
                summkret_short -= k
        portfmrkret_short.append(summkret_short)
        retlist_short = []
         
        # Increments the counter by one in order to count the change of time period (next month)
        u += 1
    
    # Compute the long portfolio return for each period
    portfret = []
    t = 0
    for h in portfmcap:
        
        # Error handling in case of division by zero
        try:
            portfret.append(portfmrkret[t]/h)
        except ZeroDivisionError:
            
            # Input zero if the sum of MCaps for this period is zero  
            portfret.append(0)                   
        t += 1                                                                                           
            
    # Compute the short portfolio return for each period 
    portfret_short = []
    t = 0
    for h in portfmcap_short:
        try:
            portfret_short.append(portfmrkret_short[t]/h)
        except ZeroDivisionError:
            portfret_short.append(0)
        t += 1
    
    # Substract the short returns from the long returns since portfret_short is composed by negative elements
    ptf = list(map(add, portfret, portfret_short))
    
    # Return a list of total returns for each period
    return ptf
    
# ------------------------------ Initializing the data ------------------------------

# Training dataset : q0aefd4ada468968f.sas7bdat
# Full dataset : qf7769e1bcba227e8.sas7bdat
data = pd.read_sas(r'qf7769e1bcba227e8.sas7bdat', format= 'sas7bdat', index=None, encoding=None, chunksize=None, iterator=False)

# Create dataframe for returns
returns = data.pivot(index='DATE',columns='PERMNO',values='RET')

# Create dataframe for shares outstanding
shrout = data.pivot(index='DATE',columns='PERMNO',values='SHROUT')

# Create dataframe for price
price = data.pivot(index='DATE',columns='PERMNO',values='PRC')

# Adjusting the negative prices to NaN 
# The dataframe price will be used inside the function compute() for the bonus question
price[price < 0] = np.nan

# Create dataframe for MCap
mcap = shrout * price

# Shift the mcap for in order to multiply returns with last month mcap : MCap(t-1)*Return(t)
mcap = mcap.shift()

# Create dataframe for MCap multiplied by Returns
# The dataframe mktret will be used inside the function compute()
mktret = mcap * returns

# Computing the cumulative returns over the 6-2 periods
# The dataframe R62 will be used as input for the function compute()
R62 =((1+ returns.shift(periods=2))*(1+ returns.shift(periods=3))*(1+ returns.shift(periods=4))*(1+ returns.shift(periods=5))*(1+ returns.shift(periods=6)))-1

# Computing the cumulative returns over the 12-7 periods
# The dataframe R127 will be used as input for the function compute()
R127 =((1+ returns.shift(periods=7))*(1+ returns.shift(periods=8))*(1+ returns.shift(periods=9))*(1+ returns.shift(periods=10))*(1+ returns.shift(periods=11))*(1+ returns.shift(periods=12)))-1

# ------------------------------ Call of the function ------------------------------

## Call of the function
#PR127 = compute(R127,False)
#PR62 = compute(R62,False)

# Call of the function for the Bonus Question
PR127 = compute(R127,True)
PR62 = compute(R62,True)

# Create dataframes out of the lists of total returns for each period
months = returns.index
portfolioR62_df = pd.DataFrame(PR62,index = months)
portfolioR127_df = pd.DataFrame(PR127,index = months)

# ------------------------------ CAPM regression ------------------------------

# Initializing the fama french 3 data 
data = pd.read_csv('FF.csv')

# Converting dates to datetime objects 
data['date'] = pd.to_datetime(data['date'].astype(int), format='%Y%m')

# Setting the date to be the index 
data = data.set_index(['date'], drop=True)

# Here I assumed the returns were given in % on the Kenneth French website
# Therefore, I divided numbers by 100
data = data.divide(100, axis='columns', level=None, fill_value = np.nan)

# The times series in the dataset used for FF5 start on July 1963. 
# Therefore, we have standardized every regression from July 1963 to December 2017
data = data.loc['1963-07-01':'2017-12-01']

# Adapt monthly protfolio returns to the right dates
portfolioR62_df = portfolioR62_df.loc['1963-07-31' : '2017-12-31']
portfolioR127_df = portfolioR127_df.loc['1963-07-31' : '2017-12-31']

# Get monthly risk free rate as a list from the dataframe "data" of fama french website 
RiskFree_df = data.iloc[:,3]
RiskFree_l = RiskFree_df.values.tolist()

# Get monthly portfolio return for R62 as a list from the dataframe portfolioR62_df
portfolioR62_df = portfolioR62_df.iloc[:,0]
portfolioR62_l = portfolioR62_df.values.tolist()

# Get monthly portfolio return for R127 as a list from the dataframe portfolioR127_df
portfolioR127_df = portfolioR127_df.iloc[:,0]
portfolioR127_l = portfolioR127_df.values.tolist()

# Create list containing the excess returns of the portfolios R127 for each period
h = 0
ExcessRetPortf62_l = []
ExcessRetPortf127_l = []
for i in RiskFree_l:
    try:
        ExcessRetPortf127_l.append(portfolioR127_l[h]-i)
    except IndexError:
        ExcessRetPortf127_l.append(np.nan)
    h += 1

# Create list containing the excess returns of the portfolios R62 for each period
h = 0  
for i in RiskFree_l:
    try:
        ExcessRetPortf62_l.append(portfolioR62_l[h]-i)
    except IndexError:
        ExcessRetPortf62_l.append(np.nan)
    h += 1

# Get the right dates
periods = pd.to_datetime(portfolioR62_df).index

# Adapt the dates for the excess returns of portfolios R62 and R127
ExcessRetPortf62_df = pd.DataFrame(ExcessRetPortf62_l, index = periods)
ExcessRetPortf127_df = pd.DataFrame(ExcessRetPortf127_l, index = periods)

# Add a constant to the independent value
X1 = sm.add_constant(changeDates(data,'Mkt-RF',periods))

# Make regression model 
model = sm.OLS(ExcessRetPortf62_df, X1, missing = 'drop')
model127 = sm.OLS(ExcessRetPortf127_df, X1, missing = 'drop')

# fit model and print results
print('\n')
print('----------------------------- Simple Regression ------------------------------')
print('---------------------------------- CAPM R26 ----------------------------------')
print(model.fit().summary())
print('\n')
print('----------------------------- Simple Regression ------------------------------')
print('---------------------------------- CAPM R712 ---------------------------------')
print(model127.fit().summary())

# ------------------------------ Fama French Implementation  ------------------------------ 

# Initialize the fama french 3 data 
dataFF = pd.read_csv(r'FF.csv')

# Convert dates to datetime objects 
dataFF['date'] = pd.to_datetime(dataFF['date'].astype(int), format='%Y%m')

# Set the date to be the index 
dataFF = dataFF.set_index(['date'], drop=True)

# Turn % into decimals 
dataFF = dataFF.divide(100, axis='columns', level=None, fill_value=np.nan)

# The times series in the dataset used for FF5 start on July 1963. 
# Therefore, we have standardized every regression from July 1963 to December 2017
ReturnsFF3 = dataFF.loc['1963-07-01':'2017-12-01']
Y = ExcessRetPortf62_df
Z = ExcessRetPortf127_df

# Standardize the dates
df = pd.DataFrame(index=pd.date_range(start = dt.datetime(1963,7,1), end = dt.datetime(2018,1,1), freq='M'))
month = df.index.to_series().apply(lambda x: dt.datetime.strftime(x,'%Y-%m-' + '01')).tolist()

# ------------------------------ FF3 Regressions ------------------------------ 

# Implement the standardization of the periods
Y = Y.set_index([month], drop=True)
Z = Z.set_index([month], drop=True)

# Set parameters for the regressions
X = ReturnsFF3[['Mkt-RF','SMB','HML','RF']]

# Add a constant
X = sm.add_constant(X) 

# Define the models
model = sm.OLS(Y, X, missing = 'drop').fit()
model127 = sm.OLS(Z, X, missing = 'drop').fit()

print('\n')
print('----------------------------- Multiple Regression ----------------------------')
print('---------------------------------- FF3 R26 -----------------------------------')
print(model.summary()) 
print('\n')
print('----------------------------- Multiple Regression ----------------------------')
print('---------------------------------- FF3 R712 ----------------------------------')
print(model127.summary())  

# ------------------------------ FF5 Regressions ------------------------------ 

# Initialize the fama french 5 data 
data = pd.read_csv(r'FF5.csv')

# Convert dates to datetime objects 
data['date'] = pd.to_datetime(data['date'].astype(int), format='%Y%m')

# Set the date to be the index 
data =data.set_index(['date'], drop=True)

# Turn percentage into decimals 
data = data.divide(100, axis='columns', level=None, fill_value=None)

# Match the indices of both datasets 
ReturnsFF5 = data.loc['1963-07-01':'2017-12-01']

# The times series in the dataset used for FF5 start on July 1963. 
# Therefore, we have standardized every regression from July 1963 to December 2017
Y = ExcessRetPortf62_df.loc['1963-07-31':'2017-12-31']
Z = ExcessRetPortf127_df.loc['1963-07-31':'2017-12-31']

# Implement the standardization of the periods
Y = Y.set_index([month], drop=True)
Z = Z.set_index([month], drop=True)

# Set parameters for the regressions
X = ReturnsFF5[['Mkt-RF','SMB','HML','RMW','CMA','RF']]

# Add a constant
X = sm.add_constant(X)

# Define the models
model = sm.OLS(Y, X, missing = 'drop').fit()
model127 = sm.OLS(Z, X, missing = 'drop').fit()

print('\n')
print('----------------------------- Multiple Regression ----------------------------')
print('---------------------------------- FF5 R26 -----------------------------------')
print(model.summary())
print('\n')
print('----------------------------- Multiple Regression ----------------------------')
print('---------------------------------- FF5 R712 ----------------------------------')
print(model127.summary())

# End computations time counter
b = dt.datetime.now()
print('\n')
print('End time : ')
print(b)
print('\n')
print('Total time : ')
print(b-a)
