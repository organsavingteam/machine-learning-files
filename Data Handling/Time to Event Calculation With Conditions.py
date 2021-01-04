# import the classes
from multiprocessing import Pool
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
import datetime
import time
import math

# intializing the documents and formatting the columns
data = pd.read_excel('/Users/Frank Yu/Downloads/Post-tx creatinine.xlsx')
data['Date'] = pd.to_datetime(data['DATE_OF_LAB'])
data.drop(['DATE_OF_LAB','DATE_OF_TRANS'], inplace = True, axis = 1)

col_name = ["MRN", "Severe Renal Dysfunction", "Date"]
ndf = pd.DataFrame(columns = col_name)

# calculating the earliest date of event and see if event lasted longer than 90 days
x = 0
while x < len(data)-2:
    if data.iloc[x,2] == 1:
        z = x
        counter = 0
        while counter == 0:
            if data.iloc[z, 0] == data.iloc[x, 0]:
                if data.iloc[z, 2] == 1:
                    if data.iloc[z + 1, 0] != data.iloc[x, 0]:
                        if data.iloc[z, 3] - data.iloc[x, 3] > datetime.timedelta(days=90):
                            if data.iloc[x, 0] in ndf.values:
                                print(data.iloc[x, 0])
                                a = ndf.where(ndf == data.iloc[x, 0]).dropna(how='all').dropna(axis=1)
                                y = a.index.values
                                for b in range(len(y)):
                                    print(b)
                                    if(data.iloc[x, 3] < ndf.iloc[y[b], 2]):
                                        ndf.drop(ndf.index[y[b]])
                                        length = len(ndf)
                                        append = [data.iloc[x, 0], 1, data.iloc[x, 3]]
                                        ndf.loc[length] = append
                                x = z + 1
                                counter = 1
                            else:    
                                length = len(ndf)
                                append = [data.iloc[x, 0], 1, data.iloc[x, 3]]
                                ndf.loc[length] = append
                                x = z + 1
                                counter = 1
                        else:
                            x = z + 1
                            counter = 1
                    else:
                        z = z + 1
                else:
                    if data.iloc[z-1, 3] - data.iloc[x, 3] > datetime.timedelta(days=90):
                        if data.iloc[x, 0] in ndf.values:
                                print(data.iloc[x, 0])
                                a = ndf.where(ndf == data.iloc[x, 0]).dropna(how='all').dropna(axis=1)
                                y = a.index.values
                                for b in range(len(y)):
                                    print(b)
                                    if(data.iloc[x, 3] < ndf.iloc[y[b], 2]):
                                        ndf.drop(ndf.index[y[b]])
                                        length = len(ndf)
                                        append = [data.iloc[x, 0], 1, data.iloc[x, 3]]
                                        ndf.loc[length] = append
                                x = z + 1
                                counter = 1
                        else:    
                            length = len(ndf)
                            append = [data.iloc[x, 0], 1, data.iloc[x, 3]]
                            ndf.loc[length] = append
                            x = z + 1
                            counter = 1
                    else:
                        x = z + 1
                        counter = 1
    else:
        x = x + 1
    
    ndf.to_csv('/Users/Frank Yu/Downloads/HTX Creatinine.csv', index = False)
