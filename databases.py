import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



class Database(object):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    def __init__(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)

def get_db(name):
    if name == "income":
        return create_income_database()
    elif name == "fake jobs":
        return create_fake_jobs_database()
    elif name == "weather":
        return create_weather_database()

def create_income_database():
    data = pd.read_csv('resources/income_evaluation.csv')
    data.columns = [x.strip() for x in data.columns]
    x_temp = data.drop(['education-num'], axis=1, inplace=False)

    x_temp['education labeled'] = x_temp.apply(lambda row: label_education(row), axis=1)

    x_temp.replace(' ?', np.nan, inplace=True)
    x_temp = x_temp.dropna() # drop all null rows
    x = x_temp.drop(['education'], axis=1, inplace=False)

    encoder = LabelEncoder()
    x['workclass'] = encoder.fit_transform(x['workclass'])
    x['education labeled'] = encoder.fit_transform(x['education labeled'])
    x['marital-status'] = encoder.fit_transform(x['marital-status'])
    x['occupation'] = encoder.fit_transform(x['occupation'])
    x['relationship'] = encoder.fit_transform(x['relationship'])
    x['race'] = encoder.fit_transform(x['race'])
    x['sex'] = encoder.fit_transform(x['sex'])
    x['native-country'] = encoder.fit_transform(x['native-country'])

    y = np.array(x['income'])

    x = x.drop(['income'], axis=1, inplace=False)

    income_db = Database(x, y)

    # encoder = ce.OneHotEncoder(
    #    cols=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
    #         'native-country'])
    # income_db.X_train = encoder.fit_transform(income_db.X_train)
    # income_db.X_test = encoder.transform(income_db.X_test)

    return income_db

def label_education (row):
    if row['education'] in [' Preschool']:
        return 'not educated'
    if row['education'] in [' 1st-4th', ' 5th-6th']:
        return 'very low'
    if row['education'] in [' 7th-8th', ' 9th', ' 10th']:
        return 'low'
    if row['education'] in [' 11th', ' 12th']:
        return 'low to medium'
    if row['education'] in [' HS-grad', ' Prof-school']:
        return 'medium'
    if row['education'] in [' Assoc-acdm', ' Assoc-voc', ' Some-college']:
        return 'medium to high'
    if row['education'] in [' Bachelors', ' Masters']:
        return 'high'
    if row['education'] in [' Doctorate']:
        return 'highest'

    return '?'

def create_fake_jobs_database():
    data = pd.read_csv('resources/fake_job_postings.csv')
    # data.columns = [x.strip() for x in data.columns]
    # data['Native Country'].fillna(data['Native Country'].mode()[0], inplace=True)

    x = data.drop(['title', 'location', 'department', 'fraudulent', 'salary_range','description','requirements','benefits','company_profile','job_id'], axis=1, inplace=False)

    x = x.fillna(x.mode().iloc[0])

    encoder = LabelEncoder()
    # x['location'] = encoder.fit_transform(x['location'])
    x['employment_type'] = encoder.fit_transform(x['employment_type'])
    x['required_experience'] = encoder.fit_transform(x['required_experience'])
    x['required_education'] = encoder.fit_transform(x['required_education'])
    x['industry'] = encoder.fit_transform(x['industry'])
    x['function'] = encoder.fit_transform(x['function'])

    y = np.array(data['fraudulent'])

    fake_jobs_db = Database(x, y)

    return fake_jobs_db

def create_weather_database():
    data = pd.read_csv('resources/weatherAUS.csv')

    data = data.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'RISK_MM', 'Date'], axis=1)
    data = data.dropna(how='any')

    data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
    # transform the categorical columns
    data = pd.get_dummies(data, columns=categorical_columns)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

    x = data.loc[:, data.columns != 'RainTomorrow']
    y = np.array(data[['RainTomorrow']])


    weather_db = Database(x, y)

    return weather_db
