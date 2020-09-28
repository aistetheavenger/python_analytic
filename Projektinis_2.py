import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
import tkinter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import datetime

df = pd.read_csv('../PROJEKTINIS/London bike sharing.csv')
# df.info()
# print(df.describe())
# print(df.iloc[:,1:].corr())

# ar yra null reiksmiu - nera
# df.isnull().values.any()

# pasiziurime stulpelius
# for i,col in enumerate(df.columns):
#     print(i+1,". column is ",col)

# pervadiname stulpelius
df.rename(columns=({
    'cnt': 'shared_bikes',
    't1': 'temperature_C',
    't2': 'Temperature_feels_like',
    'hum': 'humidity', }), inplace=True)
# df.info()

# priskiriame seasons pavadinimus
df['season'] = df['season'].replace([0, 1, 2, 3], ['spring', 'summer', 'fall', 'winter'])
# print(df['season'])

# ----grafikai----

# pie chart
# labels = 'spring', 'summer', 'fall', 'winter'
# colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
# explode = (0, 0, 0.1, 0)
# plt.pie(df.groupby(['season'])['shared_bikes'].agg('sum'), explode=explode, colors=colors, labels=labels,
#         autopct='%1.1f%%', shadow=True, startangle=90)
# plt.axis('equal')
# plt.title('Bike rent by seasons')
# plt.show()

# heatmap, kad pasiziureti, kurie duomenys koreliuoja
# sns.heatmap(df.corr(), annot=True, vmin=1, vmax=1)
# plt.show()

# Kaip oro temperatura itakoja dviraciu pasidalinijo skaiciu, pagal sezonus.
# sns.lmplot(x='cnt',y='t1',hue='season', data=df, markers=['x','o', 'v', 'o'])
# plt.xlabel('Bike shares')
# plt.ylabel('Temperature')
# plt.title('Bike shares vs Temperature')
# plt.show()

# KDE pasiskirstymas, oro salygu
# sns.kdeplot(df['temperature_C'],shade=True,color='r')
# sns.kdeplot(df['Temperature_feels_like'],shade=True,color='b')
# sns.kdeplot(df['humidity'],shade=True,color='c')
# sns.kdeplot(df['wind_speed'],shade=True,color='r')
# plt.xlabel('Temperature, C')
# plt.ylabel('Temperature "feels like"')
# plt.title('Temperature, C vs Temperature "feels like" Kde Plot System Analysis')
# plt.show()

# kokie duomenys koreliuoja
# sns.axes_style("white")
# mask = np.zeros_like(df.corr())
# mask[np.triu_indices_from(mask)] = True
# sns.heatmap(df.corr(),vmax=.3,mask=mask,square=True, annot=True)
# plt.show()

# swarmplot - oro salygos
df['weather_code'] = df['weather_code'].replace([1, 2, 3, 4, 7, 10, 26, 94], ['Clear',
                                                                              'Few clouds',
                                                                              'Broken clouds',
                                                                              'Cloudy', 'Rain',
                                                                              'Thunderstorm',
                                                                              'Snowfall',
                                                                              'Freezing Fog'])
# sns.swarmplot(x=df['weather_code'],y=df['shared_bikes'],hue=df['season'],palette='Set2',dodge=True)
# plt.xticks(rotation=90)
# plt.show()

# Vidutines parametru reiksmes pagal sezonus
# print(df.groupby('season')[['t1','t2', 'hum', 'wind_speed']].mean())

# ----pasitvarkome timestamp----
df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df['month'] = df['timestamp'].apply(lambda x: str(x).split(' ')[0].split('-')[1])
df['day'] = df['timestamp'].apply(lambda x: str(x).split(' ')[0].split('-')[2])
df['hour'] = df['timestamp'].apply(lambda x: str(x).split(' ')[1].split(':')[0])
df['day_of_week'] = df['timestamp'].apply(lambda time: time.dayofweek)

dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['day_of_week'] = df['day_of_week'].map(dmap)


# ----padarome nauja stulpeli bolean, diena suskirsto i dienos laika ir nakties laika.

def time_of_day_boolean(hour):
    if int(hour) > 6 and int(hour) < 20:
        return True
    return False


df['hour'] = pd.to_datetime(df.timestamp).dt.strftime('%H')
# print(df['hour'].apply(time_of_day_boolean))

df['time_of_day'] = df['hour'].apply(time_of_day_boolean)


# sns.countplot(data=df, x='time_of_day', hue='time_of_day', palette='YlGnBu')
# plt.legend(title='Bike count vs Time of a Day', loc='lower center', labels=['Night', 'Day'])
# plt.show()

# print(df.groupby(df['hour'].apply(time_of_day_boolean)).count())

def day_hour(hour):
    if int(hour) >= 0 and int(hour) <= 6:
        return 'between 0h-6h'
    elif int(hour) > 6 and int(hour) <= 12:
        return 'between 6h-12h'
    elif int(hour) > 12 and int(hour) <= 18:
        return 'between 12h-18h'
    elif int(hour) > 18 and int(hour) < 24:
        return 'between 18h-24h'


# print(df['hour'].apply(day_hour))
# sns.countplot(data=df, x=df['hour'].apply(day_hour), palette='YlGnBu')
# plt.legend(title='Bike count vs Time of a Day', loc='lower center')
# plt.xticks(rotation=45)
# plt.show()

# ----pasibandymui---
# def day_hour_number(hour):
#     if int(hour) >= 0 and int(hour) <= 6:
#         return 1
#     elif int(hour) > 6 and int(hour) <= 12:
#         return 2
#     elif int(hour) > 12 and int(hour) <= 18:
#         return 3
#     elif int(hour) > 18 and int(hour) < 24:
#         return 4

# ----
# figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
# figure.set_size_inches(12, 8)
# sns.barplot(data=df, x='month', y='shared_bikes', ax=ax1, palette='YlGnBu')
# sns.barplot(data=df, x='day', y='shared_bikes', ax=ax2, palette='YlGnBu')
# sns.barplot(data=df, x='day_of_week', y='shared_bikes', ax=ax3, palette='YlGnBu')
# sns.barplot(data=df, x='hour', y='shared_bikes', ax=ax4, palette='YlGnBu')
# plt.show()

# print(round(df.groupby('season')['temperature_C'].mean()))

# fig,(ax1, ax2)= plt.subplots(nrows=2)
# fig.set_size_inches(18,25)
# sns.pointplot(data=df, x='hour', y='shared_bikes', hue='is_holiday', ax=ax1, palette='YlGnBu')
# sns.pointplot(data=df, x='hour', y='shared_bikes', hue='is_weekend', ax=ax2, palette='YlGnBu')
# plt.show()

# fig,(ax3, ax4)= plt.subplots(nrows=2)
# fig.set_size_inches(18,25)
# sns.pointplot(data=df, x='hour', y='shared_bikes', hue='season', ax=ax3, palette='YlGnBu')
# sns.pointplot(data=df, x='hour', y='shared_bikes', hue='weather_code',ax=ax4, palette='YlGnBu')
# plt.show()

# ----coreliacija itraukiant time_of_day
# columns = ['shared_bikes',
#            'temperature_C',
#            'humidity',
#            'wind_speed',
#            'weather_code',
#            'is_holiday',
#            'is_weekend',
#            'season',
#            'time_of_day']
# sns.axes_style("white")
# mask = np.zeros_like(df[columns].corr())
# mask[np.triu_indices_from(mask)] = True
# sns.heatmap(df[columns].corr(), vmax=.3, mask=mask, square=True, annot=True)
# plt.xticks(rotation=45)
# plt.show()

# ----R^2 pritaikau funkcija
from scipy import stats

# x=df['shared_bikes']
# y=df['temperature_C']
# def r2(x, y):
#     return stats.pearsonr(x, y)[0] ** 2
#
# sns.jointplot(x, y, kind="reg", stat_func=r2)
# plt.show()

x = df['shared_bikes']
y = df['humidity']


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

# sns.jointplot(x, y, kind="reg", stat_func=r2)
# plt.show()

# x=df['shared_bikes']
# y=df['hour'].apply(day_hour_number)
# def r2(x, y):
#     return stats.pearsonr(x, y)[0] ** 2
#
# sns.jointplot(x, y, kind="reg", stat_func=r2)
# plt.show()

# ----Predict----

# Predict kaip pasikeis bike shares nukritus oro temperaturai
# kaip pasikeist pasidalinimo skaiciai padaugejus sventiniu dienu

# ----Train test split - NESUSKAICIUOJA, neauztenka resursu?
# X = df[['t1','t2']].values
# X = df.drop(['cnt', 'timestamp'], axis=1)

# y = df.cnt.values.reshape(-1,1)
# lg = LogisticRegression(max_iter=5000)
#
# xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.15)
# lg.fit(xTrain,yTrain)
# print(lg.score(xTest, yTest))

# print(xTest.to_string())

# print(lg.predict(np.array([[1, 95, 74, 21, 73, 25.9, 0.673, 36]]))[0])
# ---------------------


# num_cols = ['temperature_C', 'humidity', 'wind_speed']
# fig, axs = plt.subplots(1, 3, figsize=(10,6))
#
# i = 0
# for col in num_cols:
#     sns.lineplot(x=col, y='shared_bikes', data=df, ax=axs[i])
#     i+=1
# plt.show()
