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
df = pd.read_csv('../data/denverairbnb/listings.csv')
# print(df.describe())
df = df.filter(items=['availability_30','availability_60','availability_90','availability_365',
                      'price','cleaning_fee','security_deposit','accomodates','bedrooms',
                      'bathrooms','property_type','room_type','latitude','longitude','zipcode'])
pri = df['price'].astype(float)
# del df['price']
# p_type = df['property_type']
# del df['property_type']
# r_type= df['room_type']
# del df['room_type']
lat_long_df = pd.DataFrame()
lat_long_df['latitude']=df['latitude']
lat_long_df['longitude']=df['longitude']
# X = lat_long_df
# y = pri
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# p_small = p[p <= 500]
# model = sm.OLS(pri, X)
# results = model.fit()
# model.predict(model)
# print(results.summary())
zips = df['zipcode'].unique()
zc_dict = {}

for i in zips:
    zip_specific=df[df.zipcode==i]
    zc_dict.update({i:zip_specific})
# p_small = p[p <= 500]
# print(zc_dict)
means = {}
num_listings = {}
for i in zc_dict:
    p = zc_dict.get(i)
    p = p['price'].astype(float)
    means.update({i:np.mean(p)})
    num_listings.update({i:len(p)})
z_df=pd.DataFrame.from_dict(means,orient='index')
z_df = z_df.sort_values(by=[0],ascending=False)
# print(z_df)
z_df=z_df.T
temp = pd.DataFrame.from_dict(num_listings,orient='index')
temp = temp.T
z_df=z_df.append(temp,sort=False)
# print(z_df.T)
# # print(means)
# # print(num_listings)
# print(zc_dict.get('80212')['price'])
def euclidean_distance(a, b):
    """Compute the euclidean distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """
    return np.sqrt(np.dot(a - b, a - b))
"""
Functions and classes to complete non-parametric-learners individual exercise.

Implementation of kNN algorithm modeled on sci-kit learn functionality.

TODO: Improve '__main__' to allow flexible running of script
    (different ks, different number of classes)
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import sys


def euclidean_distance(a, b):
    """Compute the euclidean distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """
    return np.sqrt(np.dot(a - b, a - b))


def cosine_distance(a, b):
    """Compute the cosine dissimilarity between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """
    return 1 - np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

def manhattan_distance(a, b):
    """Compute the manhattan distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """
    return np.sum(np.abs(a - b))


class KNNRegressor:
    """Regressor implementing the k-nearest neighbors algorithm.

    Parameters
    ----------
    k: int, optional (default = 5)
        Number of neighbors that are included in the prediction.
    distance: function, optional (default = euclidean)
        The distance function to use when computing distances.
    """

    def __init__(self, k=5, distance=euclidean_distance, weighted=False):
        """Initialize a KNNRegressor object."""
        self.k = k
        self.distance = distance
        self.weighted = weighted

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        According to kNN algorithm, the training data is simply stored.

        Parameters
        ----------
        X: numpy array, shape = (n_observations, n_features)
            Training data.
        y: numpy array, shape = (n_observations,)
            Target values.

        Returns
        -------
        self
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Return the predicted values for the input X test data.

        Assumes shape of X is [n_test_observations, n_features] where
        n_features is the same as the n_features for the input training
        data.

        Parameters
        ----------
        X: numpy array, shape = (n_observations, n_features)
            Test data.

        Returns
        -------
        result: numpy array, shape = (n_observations,)
            Predicted values for each test data sample.

        """
        num_train_rows, num_train_cols = self.X_train.shape
        num_X_rows, _ = X.shape
        X = X.reshape((-1, num_train_cols))
        distances = np.zeros((num_X_rows, num_train_rows))
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                distances[i, j] = self.distance(x_train, x)
        # Sort and take top k
        k_closest_idx = distances.argsort()[:, :self.k]
        top_k = self.y_train[k_closest_idx]
        if self.weighted:
            # This is serious advanced numpy indexing. Both indexes are
            # arrays, so the dimensions are braodcase together
            # and the final shape is the shape of the result.
            top_k_distances = distances[np.arange(num_X_rows)[:, None],
                                        k_closest_idx]
            # note this will break if predicting on training data, dividing by 0
            result = np.average(top_k, axis=1, weights=1/top_k_distances)
        else:
            result = top_k.mean(axis=1)
        return result
knn = KNNRegressor(5)

first = zc_dict.get('80212')
l1=np.array(first['latitude'])
l2=np.array(first['longitude'])

# lls = np.column_stack((l1,l2))
# print(lls)
# dist = {}
# # def get_nearest():
#
# for i in range(1,len(lls)):
#     dists = []
#     for j in lls:
#         d = euclidean_distance(lls[i],j)
#         dists.append(d)
#     dists=sorted(dists)
#     dist.update({lls[i]:dists[:5:]})
# print(dist)
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