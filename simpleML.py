import numpy as np
import pandas as pd
from sklearn import linear_model
import warnings
import os
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings('ignore')
script_dir = os.path.dirname(__file__)


# Linear regression model
def lin_reg(X_train, Y_train, X_test):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    return regr, y_pred


def get_data(country_n, date_s, date_e):
    d_train = pd.read_csv(os.path.join(script_dir, "train.csv"))
    d_train = d_train.loc[
        (d_train['Country_Region'] == country_n) & (d_train['Date'] >= date_s) & (d_train['Date'] <= date_e)]
    d_train[['ConfirmedCases', 'Fatalities']] = d_train[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log(x))
    d_train.replace([np.inf, -np.inf], 0, inplace=True)
    d_train.insert(1, 'new_ind', list(range(0, d_train.shape[0])))
    d_train.insert(2, 'new_ind_2', list(range(0, d_train.shape[0])))
    confirmed_data = d_train[['new_ind', 'new_ind_2', 'ConfirmedCases']]
    fatalities_data = d_train[['new_ind', 'new_ind_2', 'Fatalities']]
    return confirmed_data, fatalities_data


def classify(confirmed, fatalities):
    test = list(range(max(confirmed['new_ind']) + 1, max(confirmed['new_ind']) + 6))
    x_test = pd.DataFrame(zip(test, test))
    model_conf, pred_conf = lin_reg(confirmed.drop(columns='ConfirmedCases'), confirmed['ConfirmedCases'], x_test)
    model_fatal, pred_fatal = lin_reg(fatalities.drop(columns='Fatalities'), fatalities['Fatalities'], x_test)

    return pred_conf, pred_fatal


def show_results(past_conf, past_fatal, pred_conf, pred_fatal, country_n, date_e):
    true_data = pd.read_csv(os.path.join(script_dir, "train.csv"))
    true_data = true_data.loc[(true_data['Country_Region'] == country_n) & (true_data['Date'] > date_e)].head(5)

    #future_conf = true_data['ConfirmedCases'].apply(lambda x: np.log(x))
    future_conf = true_data['ConfirmedCases']
    future_fatal = true_data['Fatalities']
    pred_conf = np.exp(pred_conf)
    pred_fatal = np.exp(pred_fatal)

    total_conf = list(list(past_conf['ConfirmedCases']) + list(future_conf))
    total_fatal = list(list(past_fatal['Fatalities']) + list(future_fatal))
    total_conf_pred = list(list(past_conf['ConfirmedCases']) + list(pred_conf))
    total_fatal_pred = list(list(past_fatal['Fatalities']) + list(pred_fatal))

    x_axis = list(range(0, len(total_conf)))

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(x_axis, total_conf, color='olive', marker='o', linestyle='dashed')
    axarr[0].plot(x_axis, total_conf_pred, color='blue', marker='o', markerfacecolor='blue')
    axarr[0].set_ylabel("Confirmed Cases")
    axarr[0].legend(['Confirmed Cases', 'Predicted Confirmed Cases'])
    axarr[0].set_title(country_n + " results")
    axarr[1].plot(x_axis, total_fatal, color='olive', marker='o', linestyle='dashed')
    axarr[1].plot(x_axis, total_fatal_pred, color='blue', marker='o', markerfacecolor='blue')
    axarr[1].set_xlabel("Days")
    axarr[1].set_ylabel("Fatalities Cases")
    axarr[1].legend(['Fatalities Cases', 'Predicted Fatalities Cases'])
    plt.savefig('a.png')
    plt.show()


if __name__ == "__main__":
    confirmed_d, fatalities_d = get_data(sys.argv[1], sys.argv[2], sys.argv[3])
    pred_conf, pred_fatal = classify(confirmed_d, fatalities_d)
    show_results(confirmed_d, fatalities_d, pred_conf, pred_fatal, 'Spain', '2020-03-20')
