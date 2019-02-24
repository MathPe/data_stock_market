import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        i = 0;
        for row in csvFileReader:
            i = i + 1
            dates.append(i)
            prices.append(float(row[2]))
    return

def predict_prices(dates, prices):
    print(dates)
    dates = np.reshape(dates, (len(dates), 1))
    new_dates = [62, 63, 64, 65, 66, 67, 68, 69, 70]
    new_dates = np.reshape(new_dates, (len(new_dates), 1))

    svr_lin = SVR(kernel= 'linear', C=1e3)
    #svr_lin.fit(dates, prices)
    svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
    #svr_poly.fit(dates, prices)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
    svr_rbf.fit(dates, prices)
    print(svr_rbf.predict(new_dates)[1])

    plt.scatter(dates, prices, color='black', label='Data')
    #plt.plot(dates, svr_lin.predict(dates), color='red', label='Linear model')
    #plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.plot(dates, svr_rbf.predict(dates), color='green', label='RBF model')
    plt.plot(new_dates, svr_rbf.predict(new_dates), color='blue', label='RBF model prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return

get_data('AAPL.csv')
predict_prices(dates, prices)
