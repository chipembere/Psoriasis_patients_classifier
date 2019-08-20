# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Random_state = np.random.RandomState(0)

# load the datasets
dataframe = pd.read_excel(r"/Users/brianmusonza/Documents/venv/Data/data_for_classification.xlsx")
# Define missing values as np.NaN
df2 = dataframe.replace(r'\s+', np.NaN)
# Drop Weeks 7 to 11
data_= df2.drop(['PASI.END.WEEK.11'], axis = 1)
# Split the dataset by rows
ds = data_.iloc[0:20]
ds = ds.drop([12])
ds = ds.iloc[:,0:]
ds = ds.drop(['CLASS', 'ID'], axis=1)
ds.columns = ['W0-0','W0-1', 'W0-2', 'W0-3', 'W0-4', 'W0-5','W0-6','W0-7','W0-8','W0-9','W0-10']
ds = ds.dropna()

ds.T.plot(color=['green', 'blue', 'red'],figsize=(9,8))

plt.xlabel('Time', fontsize=14)
plt.ylabel('Trajectories', fontsize=14)
plt.title('Line graph of PASI Trajectories', fontsize=14)

plt.show()
