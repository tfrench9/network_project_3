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
from sklearn.neural_network import MLPRegressor
from matplotlib.markers import MarkerStyle

features_df = pd.read_csv("DATASET1_variables.csv")
timeseries_df = pd.read_csv("DataSet2_time_series.csv")

n = len(features_df.index)

features = features_df.to_numpy()[:,3:]
labels = features_df.to_numpy()[:,2].reshape((n, 1))
coviddata = timeseries_df.to_numpy()[:,5:]

cases = coviddata[::2, :]
deaths = coviddata[1::2, :]

for i in range(cases.shape[0]):
    plt.plot(cases[i, :], "b--")
plt.title("COVID-19 Case Growth over Time")
plt.xlabel("time (days since 1/22/2020)")
plt.ylabel("Total Cases")

plt.show()

cases_total = cases[:, -1]
deaths_total = deaths[:, -1]

cases_prev = np.hstack((np.zeros((n, 1)), cases[:,:-1]))
deaths_prev = np.hstack((np.zeros((n, 1)), deaths[:,:-1]))

rate_cases = np.mean(np.divide(cases, cases_prev + 1), 1)
rate_deaths = np.mean(np.divide(deaths, deaths_prev + 1), 1)

# Sort Cities by Total number of Cases
inds = np.argsort(cases_total)

plt.title("Average Rates of Growth vs. Total Cases")
plt.xscale("symlog")
plt.scatter(cases_total[inds], rate_cases[inds])
plt.scatter(cases_total[inds], rate_deaths[inds])
plt.xlabel("Total Cases (log Scale)")
plt.ylabel("Average Daily Rate of Increase (log Scale)")
plt.legend(["Cases", "Deaths"])
plt.show()

plt.title("Total Deaths vs. Total Cases")
plt.xscale("symlog")
plt.yscale("symlog")
plt.scatter(cases_total, deaths_total)
plt.xlabel("Total Cases (log Scale)")
plt.ylabel("Total Deaths (log Scale)")
plt.show()

# Feature Normalization
pop = features[:, 3]
pop_features = features[:, 4:] / features[:, 3][:, None]

y_vec = []
y_vec.append(cases_total/pop)
y_vec.append(cases_total)
y_vec.append(deaths_total/pop)

y_labels = ["Total Case Density", "Total Cases", "Total Death Density", "Total Deaths", "Average Case Growth Rate", "Deaths / Cases"]

plt.title("Total Cases vs. Population")
plt.scatter(pop[:159], cases_total[:159])
plt.scatter(pop[159:], cases_total[159:])
plt.legend(["GA", "NY"])
plt.show()

X0 = np.hstack((pop_features, features[:, 0:4]))

plt.title("Median Income vs. Poverty Rate")
plt.scatter(X0[:, 6], X0[:, 7])
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Median Income ($)")
plt.show()

for yi in range(len(y_vec)):
    y0 = y_vec[yi];

    min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(X)

    #Mask GA:
    X1 = X0[:159, :]
    y1 = y0[:159]

    X2 = X0[159:, :]
    y2 = y0[159:]
    # y = np.hstack((cases_total.reshape(n, 1), deaths_total.reshape(n, 1), rate_cases.reshape(n, 1), rate_deaths.reshape(n, 1))).astype(float)
    # y = min_max_scaler.fit_transform(y)

    X_means1 = np.mean(X1, 0)
    X_means2 = np.mean(X2, 0)

    clf1 = svm.SVR(gamma = "auto", kernel = "rbf").fit(X1, y1)
    clf2 = svm.SVR(gamma = "auto", kernel = "rbf").fit(X2, y2)
    rfr1 = RandomForestRegressor(n_estimators = 30).fit(X1, y1)
    rfr2 = RandomForestRegressor(n_estimators = 30).fit(X2, y2)
    lf1 = linear_model.LinearRegression().fit(X1, y1)
    lf2 = linear_model.LinearRegression().fit(X2, y2)

    feature_names = [" % GENDER_M", " % GENDER_F", " % AGE 0-19", " % AGE 20-49", " % AGE 50 ABOVE", " POVEST", " POVRATE", " MEDINCOME", " TOTALPOP"]

    fi = [2, 3, 4, 6, 7, 8]

    for i in range(len(fi)):

        X_test1 = np.multiply(np.ones((100, 9)), X_means1[None, :])
        X_test2 = np.multiply(np.ones((100, 9)), X_means2[None, :])

        X_vec1 = np.linspace(np.min(X1[:, fi[i]]), np.max(X1[:, fi[i]]), 100)
        X_vec2 = np.linspace(np.min(X2[:, fi[i]]), np.max(X2[:, fi[i]]), 100)

        X_test1[:, fi[i]] = X_vec1
        X_test2[:, fi[i]] = X_vec2

        y_pred1 = clf1.predict(X_test1)
        y_pred2 = clf1.predict(X_test2)

        y_pred12 = rfr1.predict(X_test1)
        y_pred22 = rfr2.predict(X_test2)

        y_pred13 = lf1.predict(X_test1)
        y_pred23 = lf2.predict(X_test2)


        plt.subplot(2,3,i + 1)
        fig = plt.gcf()
        fig.suptitle(y_labels[yi])

        plt.plot(X_vec1, y_pred12, "r--")
        plt.plot(X_vec2, y_pred22, "r--")

        plt.plot(X_vec1, y_pred13, "r:")
        plt.plot(X_vec2, y_pred23, "b:")

        plt.scatter(X1[:, fi[i]], y1[:], marker = "x", c = "Red", s = 2)
        plt.scatter(X2[:, fi[i]], y2[:], marker = "x", c = "Blue", s = 2)

        if i == len(fi) - 1:
            plt.legend(["SVR GA","SVR NY","RandomForest GA","RandomForest NY","Linear GA","Linear NY"])

        plt.xlabel("Input" + feature_names[fi[i]])
        plt.ylabel(y_labels[yi])
    plt.show()

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
