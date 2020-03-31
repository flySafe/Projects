import sys
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


test = pd.read_csv(r"C:\Users\Naora\Desktop\studys\hackorona\test.csv")
train = pd.read_csv(r"C:\Users\Naora\Desktop\studys\hackorona\train.csv")

# confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})
# fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})
# total_date = confirmed_total_date.join(fatalities_total_date)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
# total_date.plot(ax=ax1)
# ax1.set_title("Global confirmed cases", size=13)
# ax1.set_ylabel("Number of cases", size=13)
# ax1.set_xlabel("Date", size=13)
# fatalities_total_date.plot(ax=ax2, color='orange')
# ax2.set_title("Global deceased cases", size=13)
# ax2.set_ylabel("Number of cases", size=13)
# ax2.set_xlabel("Date", size=13)
# plt.show()

######## 3 Data Enrichment ###########################

# Merge train and test, exclude overlap
dates_overlap = ['2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25',
                 '2020-03-26', '2020-03-27']
train2 = train.loc[~train['Date'].isin(dates_overlap)]
all_data = pd.concat([train2, test], axis=0, sort=False)

# Double check that there are no informed ConfirmedCases and Fatalities after 2020-03-11
all_data.loc[all_data['Date'] >= '2020-03-19', 'ConfirmedCases'] = np.nan
all_data.loc[all_data['Date'] >= '2020-03-19', 'Fatalities'] = np.nan
all_data['Date'] = pd.to_datetime(all_data['Date'])

# Create date columns
le = preprocessing.LabelEncoder()
all_data['Day_num'] = le.fit_transform(all_data.Date)
all_data['Day'] = all_data['Date'].dt.day
all_data['Month'] = all_data['Date'].dt.month
all_data['Year'] = all_data['Date'].dt.year

# Fill null values given that we merged train-test datasets
all_data['Province_State'].fillna("None", inplace=True)
all_data['ConfirmedCases'].fillna(0, inplace=True)
all_data['Fatalities'].fillna(0, inplace=True)
all_data['Id'].fillna(-1, inplace=True)
all_data['ForecastId'].fillna(-1, inplace=True)

# display(all_data)
# display(all_data.loc[all_data['Date'] == '2020-03-19'])

missings_count = {col: all_data[col].isnull().sum() for col in all_data.columns}
missings = pd.DataFrame.from_dict(missings_count, orient='index')


# print(missings.nlargest(30, 0))

def calculate_trend(df, lag_list, column):
    for lag in lag_list:
        trend_column_lag = "Trend_" + column + "_" + str(lag)
        temp=df[column]
        temp2=df[column].shift(lag, fill_value=-999)
        df[trend_column_lag] = (df[column] - df[column].shift(lag, fill_value=-999)) / df[column].shift(lag,fill_value=0)
    return df


def calculate_lag(df, lag_list, column):
    for lag in lag_list:
        column_lag = column + "_" + str(lag)
        df[column_lag] = df[column].shift(lag, fill_value=0)
    return df


ts = time.time()
all_data = calculate_lag(all_data, range(1, 7), 'ConfirmedCases')
all_data = calculate_lag(all_data, range(1, 7), 'Fatalities')
all_data = calculate_trend(all_data, range(1, 7), 'ConfirmedCases')
all_data = calculate_trend(all_data, range(1, 7), 'Fatalities')
all_data.replace([np.inf, -np.inf], 0, inplace=True)
all_data.fillna(0, inplace=True)
# print("Time spent: ", time.time()-ts)
display(all_data[all_data['Country_Region'] == 'Spain'].iloc[40:50][['Id', 'Province_State', 'Country_Region', 'Date',
                                                                     'ConfirmedCases', 'Fatalities', 'ForecastId',
                                                                     'Day_num', 'ConfirmedCases_1',
                                                                     'ConfirmedCases_2', 'ConfirmedCases_3',
                                                                     'Fatalities_1', 'Fatalities_2',
                                                                     'Fatalities_3']])

world_population = pd.read_csv(r"C:\Users\Naora\Desktop\studys\hackorona\population_by_country_2020.csv")

# Select desired columns and rename some of them
world_population = world_population[
    ['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]
world_population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age',
                            'Urban Pop']

# Replace United States by US
world_population.loc[world_population['Country (or dependency)'] == 'United States', 'Country (or dependency)'] = 'US'

# Remove the % character from Urban Pop values
world_population['Urban Pop'] = world_population['Urban Pop'].str.rstrip('%')

# Replace Urban Pop and Med Age "N.A" by their respective modes, then transform to int
world_population.loc[world_population['Urban Pop'] == 'N.A.', 'Urban Pop'] = int(
    world_population.loc[world_population['Urban Pop'] != 'N.A.', 'Urban Pop'].mode()[0])
world_population['Urban Pop'] = world_population['Urban Pop'].astype('int16')
world_population.loc[world_population['Med Age'] == 'N.A.', 'Med Age'] = int(
    world_population.loc[world_population['Med Age'] != 'N.A.', 'Med Age'].mode()[0])
world_population['Med Age'] = world_population['Med Age'].astype('int16')

# display(world_population)

# Now join the dataset to our previous DataFrame and clean missings (not match in left join)- label encode cities
# print("Joined dataset")
all_data = all_data.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)', how='left')
all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = all_data[
    ['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)
# display(all_data)

# Label encode countries and provinces. Save dictionary for exploration purposes
all_data.drop('Country (or dependency)', inplace=True, axis=1)
all_data['Country_Region'] = le.fit_transform(all_data['Country_Region'])
number_c = all_data['Country_Region']
countries = le.inverse_transform(all_data['Country_Region'])
country_dict = dict(zip(countries, number_c))
all_data['Province_State'] = le.fit_transform(all_data['Province_State'])
number_p = all_data['Province_State']
province = le.inverse_transform(all_data['Province_State'])
province_dict = dict(zip(province, number_p))

# display(all_data)
##plt.table(all_data)
##plt.show()

app = QApplication(sys.argv)
model = pandasModel(all_data.loc[all_data['Country_Region']==country_dict['Spain']][45:65])
view = QTableView()
view.setModel(model)
view.resize(800, 600)
view.show()
sys.exit(app.exec_())

######## 4 ML ###########################

data = all_data.copy()
features = ['Id', 'ForecastId', 'Country_Region', 'Province_State', 'ConfirmedCases', 'Fatalities',
            'Day_num', 'Day', 'Month', 'Year']
data = data[features]
# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends
data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')
data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log(x))

# Replace infinites
data.replace([np.inf, -np.inf], 0, inplace=True)
# display(data)
x_train = data[data.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)


# Split data into train/test
def split_data(data):
    # Train set
    x_train = data[data.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)
    y_train_1 = data[data.ForecastId == -1]['ConfirmedCases']
    y_train_2 = data[data.ForecastId == -1]['Fatalities']

    # Test set
    x_test = data[data.ForecastId != -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)

    # Clean Id columns and keep ForecastId as index
    x_train.drop('Id', inplace=True, errors='ignore', axis=1)
    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    x_test.drop('Id', inplace=True, errors='ignore', axis=1)
    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    return x_train, y_train_1, y_train_2, x_test
