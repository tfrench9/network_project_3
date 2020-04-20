import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

features_df = pd.read_csv("DATASET1_variables.csv")
timeseries_df = pd.read_csv("DataSet2_time_series.csv")

n = len(features_df.index)

features = features_df.to_numpy()[:,3:]
labels = features_df.to_numpy()[:,2].reshape((n, 1))
coviddata = timeseries_df.to_numpy()[:,5:]

cases = coviddata[::2, :]
deaths = coviddata[1::2, :]

cases_total = np.sum(cases, 1)
deaths_total = np.sum(deaths, 1)

cases_prev = np.hstack((np.zeros((n, 1)), cases[:,:-1]))
deaths_prev = np.hstack((np.zeros((n, 1)), deaths[:,:-1]))

rate_cases = np.mean(np.divide(cases, cases_prev + 1), 1)
rate_deaths = np.mean(np.divide(deaths, deaths_prev + 1), 1)

# Sort Cities by Total number of Cases
inds = np.argsort(cases_total)
print(features[:,3])

# plt.title("Average Rates of Growth vs. Total Cases")
# plt.xscale("symlog")
# plt.scatter(cases_total[inds], rate_cases[inds])
# plt.scatter(cases_total[inds], rate_deaths[inds])
# plt.xlabel("Total Cases (log Scale)")
# plt.ylabel("Average Daily Rate of Increase (log Scale)")
# plt.legend(["Cases", "Deaths"])
# plt.show()

# Feature Normalization
pop_features = features[:, 4:] / features[:, 3][:, None]

print(pop_features.shape)
print(features[0:4, :].shape)

X = np.hstack((pop_features, features[:, 0:4]))
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
print(X)

#X = featuresfeatures[:,[]]
#or i in range(n):
    #cases[] = np.append(cases, )
#deaths =
