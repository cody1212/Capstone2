import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import folium
from folium import plugins
# import seaborn as sns
from folium.plugins import HeatMap
pd.set_option("display.max_rows",1300)
pd.set_option("display.max_columns",12000)
pd.set_option('display.max_rows',200000)
pd.set_option('display.width', 500)
df = pd.read_csv('denverairbnb/listings.csv')
# print(df.describe())
df = df.filter(items=['availability_30','availability_60','availability_90','availability_365',
                      'price','cleaning_fee','security_deposit','accomodates','bedrooms',
                      'bathrooms','property_type','room_type','latitude','longitude','zipcode'])
pri = df['price'].astype(float)
del df['price']
# p_type = df['property_type']
# del df['property_type']
# r_type= df['room_type']
# del df['room_type']
lat_long_df = pd.DataFrame()
lat_long_df['latitude']=df['latitude']
lat_long_df['longitude']=df['longitude']
X = lat_long_df
y = pri
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# p_small = p[p <= 500]
model = sm.OLS(pri, X)
results = model.fit()
model.predict(model)
print(results.summary())
zips = df['zipcode'].unique()
zc_dict = {}
for i in zips:
    zip_specific=df[df.zipcode==i]
    zc_dict.update({i:zip_specific})
# p_small = p[p <= 500]
means = {}
lens = {}
for i in zc_dict:
    p = zc_dict.get(i)
    p = p['price'].astype(float)
    means.update({i:np.mean(p)})
    lens.update({i:len(p)})

# std = np.std(p)
# x = np.linspace(0, np.max(p), 200)

# y = stats.lognorm.pdf(x, std, np.mean(p), np.exp(np.mean(p)))
# plt.xlim(0, 2000)
# plt.plot(x, y)
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.hist(p_small, bins = 25)
# plt.show()
# x = np.linspace(0,100, 500)
# ax.set_title('Airbnb Prices')
# ax.set_xlabel('Prices')
# ax.set_ylabel('Counts')
# s = np.std(p_small)
# l = np.mean(p_small)
# sc = np.log(l)
# param = stats.lognorm.fit(p_small)
# rand_pdf = stats.lognorm(.5,loc=0,scale=1).rvs(1000)
# paramrand = stats.lognorm.fit(rand_pdf)
# y1 = stats.lognorm.pdf(x,param[0],loc=param[1],scale=param[2])
# y2 = stats.lognorm.pdf(x,paramrand[0],loc=paramrand[1],scale=paramrand[2])
# ax.plot(x, y2)
# ax.hist(p_small)
# print(x, s, l, sc)
# plt.savefig('Airbnb Prices')
# plt.show()
# availability 30,60,90,365
# price, monthly, weekly,cleaning, security deposit
# square feet
# accomodates
# bedrooms
# bathrooms
# property type
# room type
# latitude
# longitude
# zipcode