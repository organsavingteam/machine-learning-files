# import classes
from multiprocessing import Pool
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
import datetime
import time
import math

# load the data and format columns
data = pd.read_excel('/Users/Frank Yu/Downloads/PGF.xlsx')
data['Date'] = pd.to_datetime(data['DatePrimGraftFailure'])
data.drop(['Tx', 'PrimaryGraftFAilure', 'Name', 'DOB'], inplace = True, axis = 1)

col_name = ["MRN", "PGF", "Date"]
ndf = pd.DataFrame(columns = col_name)

# goes through the specified columns and remove redundant data. In this case, it takes the earliest date and uses that as the single new value
x = 0
while x < len(data)-2:
    if data.iloc[x, 0] in ndf.values:
        print(data.iloc[x, 0])
        a = ndf.where(ndf == data.iloc[x, 0]).dropna(how='all').dropna(axis=1)
        y = a.index.values
        for b in range(len(y)):
            print(b)
            if(data.iloc[x, 1] < ndf.iloc[y[b], 2]):
                ndf.drop(ndf.index[y[b]])
                length = len(ndf)
                append = [data.iloc[x, 0], 1, data.iloc[x, 1]]
                ndf.loc[length] = append
        x = x + len(y)
    else:    
        length = len(ndf)
        append = [data.iloc[x, 0], 1, data.iloc[x, 1]]
        ndf.loc[length] = append
        x = x + 1
        
ndf.to_csv('/Users/Frank Yu/Downloads/new cav.csv', index = False)
