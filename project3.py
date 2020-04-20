import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

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

# plt.title("Average Rates of Growth vs. Total Cases")
# plt.xscale("symlog")
# plt.scatter(cases_total[inds], rate_cases[inds])
# plt.scatter(cases_total[inds], rate_deaths[inds])
# plt.xlabel("Total Cases (log Scale)")
# plt.ylabel("Average Daily Rate of Increase (log Scale)")
# plt.legend(["Cases", "Deaths"])
# plt.show()

# Feature Normalization
pop = features[:, 3]
pop_features = features[:, 4:] / features[:, 3][:, None]

X = np.hstack((pop_features, features[:, 0:4]))
min_max_scaler = preprocessing.MinMaxScaler()
# X = min_max_scaler.fit_transform(X)


y = cases_total/pop
# y = np.hstack((cases_total.reshape(n, 1), deaths_total.reshape(n, 1), rate_cases.reshape(n, 1), rate_deaths.reshape(n, 1))).astype(float)
# y = min_max_scaler.fit_transform(y)

X_means = np.mean(X, 0)

clf = svm.SVR(gamma = "auto", kernel = "rbf")
clf.fit(X, y)

clf = RandomForestRegressor().fit(X, y)
clf2 = linear_model.LinearRegression().fit(X, y)
#clf.predict(X[:2, :])


reg1 = linear_model.Lasso(alpha=0.1)
reg1.fit(X, y)

reg2 = linear_model.BayesianRidge()
reg2.fit(X, y)

feature_names = [" % GENDER_M", " % GENDER_F", " % AGE 0-19", " % AGE 20-49", " % AGE 50 ABOVE", " POVEST", " POVRATE", " MEDINCOME", " TOTALPOP"]

for i in range(9):
    X_test = np.multiply(np.ones((100, 9)), X_means[None, :])
    X_vec = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100)
    X_test[:, i] = X_vec

    y_pred = clf.predict(X_test)
    y_pred2 = clf2.predict(X_test)

    plt.plot(X_vec, y_pred)
    plt.plot(X_vec, y_pred2 - np.min(y_pred2))
    plt.scatter(X[:, i], y[:])


    plt.xlabel("Input" + feature_names[i])
    plt.ylabel("Learned COVID-19 Case Density")
    plt.title(feature_names[i])
    plt.show()




print(y)
# define model
#
# model = Sequential()
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(n_outputs, activation='sigmoid'))
#
# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # fit the model
# model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
#
# yhat = model.predict(X_test)
# print(yhat)
#
# #, y_test, verbose=0)
# print('Test Accuracy: %.3f' % acc)


#model.predict()
#print('Predicted: %.3f' % yhat)

#X = featuresfeatures[:,[]]
#or i in range(n):
    #cases[] = np.append(cases, )
#deaths =
