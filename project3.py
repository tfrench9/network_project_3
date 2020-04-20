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

# Import datasets
features_df = pd.read_csv("DATASET1_variables.csv")
timeseries_df = pd.read_csv("DataSet2_time_series.csv")

# Check length of datasets
n = len(features_df.index)

# Extract numeric features and city labels
features = features_df.to_numpy()[:,3:]
labels = features_df.to_numpy()[:,2].reshape((n, 1))
coviddata = timeseries_df.to_numpy()[:,5:]

# Parse through COVID data
cases = coviddata[::2, :]
deaths = coviddata[1::2, :]

# Plot Covid 19 Case Growth over time
for i in range(cases.shape[0]):
    plt.plot(cases[i, :], "b--")
plt.title("COVID-19 Case Growth over Time")
plt.xlabel("time (days since 1/22/2020)")
plt.ylabel("Total Cases")
plt.show()

# Find cases and death totals
cases_total = cases[:, -1]
deaths_total = deaths[:, -1]

# Calculate rate of growth of cases and deaths
cases_prev = np.hstack((np.zeros((n, 1)), cases[:,:-1]))
deaths_prev = np.hstack((np.zeros((n, 1)), deaths[:,:-1]))
rate_cases = np.mean(np.divide(cases, cases_prev + 1), 1)
rate_deaths = np.mean(np.divide(deaths, deaths_prev + 1), 1)

# Sort Cities by Total number of Cases
inds = np.argsort(cases_total)

# plot Average Rates of Growth vs. Total Cases
plt.title("Average Rates of Growth vs. Total Cases")
plt.xscale("symlog")
plt.scatter(cases_total[inds], rate_cases[inds])
plt.scatter(cases_total[inds], rate_deaths[inds])
plt.xlabel("Total Cases (log Scale)")
plt.ylabel("Average Daily Rate of Increase (log Scale)")
plt.legend(["Cases", "Deaths"])
plt.show()

# Plot Total Deaths vs. Total Cases
plt.title("Total Deaths vs. Total Cases")
plt.xscale("symlog")
plt.yscale("symlog")
plt.scatter(cases_total, deaths_total)
plt.xlabel("Total Cases (log Scale)")
plt.ylabel("Total Deaths (log Scale)")
plt.show()

# Feature Normalization, Population based features to percentages
pop = features[:, 3]
pop_features = features[:, 4:] / features[:, 3][:, None]

# Choose multiple labels to test
y_vec = []
y_vec.append(cases_total/pop)
y_vec.append(cases_total)
y_vec.append(deaths_total/pop)

# Get information on current label and format
y_labels = ["Total Case Density", "Total Cases", "Total Death Density"]

# Plot Total Cases vs. Population
plt.title("Total Cases vs. Population")
plt.scatter(pop[:159], cases_total[:159])
plt.scatter(pop[159:], cases_total[159:])
plt.legend(["GA", "NY"])
plt.show()

# Initialize feature array after normalization
X0 = np.hstack((pop_features, features[:, 0:4]))

# Plot Median Income vs. Poverty rate from features
plt.title("Median Income vs. Poverty Rate")
plt.scatter(X0[:, 6], X0[:, 7])
plt.xlabel("Poverty Rate (%)")
plt.ylabel("Median Income ($)")
plt.show()

# Loop through all testing label options
for yi in range(len(y_vec)):

    # Get specific labels for regression
    y0 = y_vec[yi];

    # Mask GA and NY values to process seperately
    X1 = X0[:159, :]
    y1 = y0[:159]
    X2 = X0[159:, :]
    y2 = y0[159:]

    # Find mean values for features
    X_means1 = np.mean(X1, 0)
    X_means2 = np.mean(X2, 0)

    # Train ML models: Support Vector regression, Random Forest Regressor, Linear Regression
    clf1 = svm.SVR(gamma = "auto", kernel = "rbf").fit(X1, y1)
    clf2 = svm.SVR(gamma = "auto", kernel = "rbf").fit(X2, y2)
    rfr1 = RandomForestRegressor(n_estimators = 30).fit(X1, y1)
    rfr2 = RandomForestRegressor(n_estimators = 30).fit(X2, y2)
    lf1 = linear_model.LinearRegression().fit(X1, y1)
    lf2 = linear_model.LinearRegression().fit(X2, y2)

    # List out feature names for plotting
    feature_names = [" % GENDER_M", " % GENDER_F", " % AGE 0-19", " % AGE 20-49", " % AGE 50 ABOVE", " POVEST", " POVRATE", " MEDINCOME", " TOTALPOP"]

    # Select features to plot
    fi = [2, 3, 4, 6, 7, 8]

    # Iterate through features to test model behavior with altered input
    for i in range(len(fi)):

        # Create testing data: hold all features at mean but sweep feature of interest
        # from minimum value to maximum value to see behavior
        X_test1 = np.multiply(np.ones((100, 9)), X_means1[None, :])
        X_test2 = np.multiply(np.ones((100, 9)), X_means2[None, :])
        X_vec1 = np.linspace(np.min(X1[:, fi[i]]), np.max(X1[:, fi[i]]), 100)
        X_vec2 = np.linspace(np.min(X2[:, fi[i]]), np.max(X2[:, fi[i]]), 100)
        X_test1[:, fi[i]] = X_vec1
        X_test2[:, fi[i]] = X_vec2

        # Predict outputs givin synthetic inputs
        y_pred1 = clf1.predict(X_test1)
        y_pred2 = clf1.predict(X_test2)
        y_pred12 = rfr1.predict(X_test1)
        y_pred22 = rfr2.predict(X_test2)
        y_pred13 = lf1.predict(X_test1)
        y_pred23 = lf2.predict(X_test2)

        # Plot results
        plt.subplot(2,3,i + 1)
        fig = plt.gcf()
        fig.suptitle(y_labels[yi])
        plt.plot(X_vec1, y_pred12, "r--")
        plt.plot(X_vec2, y_pred22, "r--")
        plt.plot(X_vec1, y_pred13, "r:")
        plt.plot(X_vec2, y_pred23, "b:")

        # Plot original data on top
        plt.scatter(X1[:, fi[i]], y1[:], marker = "x", c = "Red", s = 2)
        plt.scatter(X2[:, fi[i]], y2[:], marker = "x", c = "Blue", s = 2)

        # Add Legend
        if i == 1:
            plt.legend(["RandomForest GA","RandomForest NY","Linear GA","Linear NY", "Data GA","Data NY"])

        plt.xlabel("Input" + feature_names[fi[i]])
        plt.ylabel(y_labels[yi])
    plt.show()
