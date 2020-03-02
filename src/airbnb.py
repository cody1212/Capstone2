import seaborn as sns
import pickle
import math
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import requests
import folium
import statistics
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from folium import plugins
from folium.plugins import HeatMap
from geopy.distance import distance
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

pd.set_option("display.max_rows",1300)
pd.set_option("display.max_columns",12000)
pd.set_option('display.max_rows',200000)
pd.set_option('display.width', 500)

dista=pd.read_csv('../data/distances_df.csv')
dista = dista.rename(columns={'Unnamed: 0': 'ind'})

df = pd.read_csv('../data/denverairbnb/listings.csv')
# print(df.describe())
df = df.filter(items=['summary','description','neighborhood_overview','availability_30',
                      'availability_60','availability_90','availability_365',
                      'price','cleaning_fee','security_deposit','accommodates','bedrooms',
                      'bathrooms','property_type','room_type','latitude','longitude','zipcode'])

# #
# df = df[df.price<2000]
df['price_bin']=pd.cut(df['price'],[0,100,200,max(df['price'])],labels=['under_100','100_200','200_or_more']).astype(str)
df['Log_Price']=np.log10(df.price)
types= list(df.property_type.unique())

def mk_area(z):
    s='non'
    areas = {'northeast': ['80022', '80249', '80239', '80226', '80238'], 'east': ['80207', '80220'],
             'southeast': ['80230', '80247', '80012', '80014', '80231', '80222', '80224', '80237', '80246'],
             'south': ['80210', '80209', '80223', '80219'], 'southwest': ['80123', '80236', '80235', '80227', '80127'],
             'west': ['80228', '80232', '80226', '80214', '80215'], 'northwest': ['80033', '80212', '80211', '80221'],
             'north': '80216'}
    for i in areas:
        try:
            if areas.get(i).index(z)!=None:
                s=i
        except:
            x=5
    if s!='non':
        return s
    else:
        return 'central'

def mk_graphs(df):
    cnter = 0
    for i in ['under_100','100_200','200_or_more']:
        i = df[(df.price_bin == i)&(df.price<1500)]
        sns.jointplot(i.price,i.Complement_of_Availability_Next_90_Days,kind='kde',color='darkblue',space=1,ax=ax[0][cnter])
        i.hist('price', bins=20, color='maroon',ax=ax[1][cnter])
        cnter+=1
        plt.savefig(f"{i}",format='png',dpi=300)

def rand_for(df):
    df2 = df.filter(items=[
                          'price', 'security_deposit', 'accomodates',
                           'bedrooms', 'bathrooms', 'property_type',
                           'room_type', 'latitude', 'longitude',
                           'housing_type','price_bin','amount'
                           'areas','Complement_of_Availability_Next_90_Days','cleaning_fee'])
    print(df2.head(),len(df2))
    df3 = pd.DataFrame()
    # Random Forest ************************************************************************************************
    # for i in df2.price_bin.unique():
    #     t = df2[df2.price_bin==i]
    t = df2
    t =t.fillna(0)
    t = pd.get_dummies(t)
    print(t.columns,len(t))
    y = t.pop('price').values
    X = t.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf = RFR(n_estimators=500)
    mod = rf.fit(X_train, y_train,)
    rf_rmse = '%.2f'%np.sqrt(mse(y_test,rf.predict(X_test)))
    print('rmse:',rf_rmse)
    rf_score = '%.3f'%rf.score(X_test, y_test)
    print("Random Forest score:", rf_score)
    imp =  (rf.feature_importances_)
    ord = np.argsort(rf.feature_importances_)[::-1]
    _cols = t.columns.tolist()
    imp_cols = ord[:6]
    # feats = _cols[imp_cols]
    feats = []
    for i in range(len(imp_cols)):
        for j in _cols:
            if _cols.index(j) == imp_cols[i]:
                feats.append(j)
    print(feats)
    x = sorted(imp,reverse=True)[:6]
    imp_feats = {}
    for i in range(len(feats)):
        imp_feats.update({feats[i]:'%.4f'%x[i]})
    print(imp_feats)
    # breakpoint()
    tempdf = pd.DataFrame.from_dict(imp_feats,orient='index').T

    df3 = df3.append(tempdf,sort=True)
    print(df3)
# df3.to_csv('feature_importance_table.csv')
    # x = np.array(df.columns.tolist())[idx]
    # y = np.array(x)[idx]
    #     model = sm.OLS(y_train, X_train)
    #     results = model.fit()
    #     model.predict(X_test,y_test)
    #     print(results.summary())
    return (imp_cols,_cols,imp,imp_feats,rf_rmse,rf_score)

def get_addy_info(x,y,f):
   geolocator = Nominatim(user_agent=str(f))
   location = geolocator.reverse((x,y))
   return location.address

def plot_feat_imp(idx, features, feat_importances, n=6, fname='images/test.jpeg'):
    '''
    Plot the top n features.
    '''
    idx = idx[::-1]
    labels = np.array(features)[idx[:n]]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.barh(range(n), feat_importances[idx[:n]], color='darkblue', alpha=0.85)
    # ax.set_xticklabels(labels)
    ax.set_title('Overall Feature Importances')
    plt.yticks(ticks=range(n), labels=labels)
    plt.tight_layout(pad=1)
    plt.savefig(fname)
    plt.close()

def select_df(df,feature,value,feature2=None):
    df=df[df[feature]==value]
    return df

def deep_search_sample(df):
    '''
    storing zillowAPIdataset into a csv file
    '''
    api_url_base = 'http://www.zillow.com/webservice/GetDeepSearchResults.htm'

    columns = ['address', 'amount', 'zipcode', 'city', 'state', 'latitude', 'longitude']

    # lst that will be used to create dataframe
    for i in range(len(df)):
        try:
            # grabs data from df
            address_param = df[i:i+1].address.tolist()[0]
            citystatezip_param = df[i:i+1].citystatezip.tolist()[0]

            # upload data as param
            payload = {'zws-id': os.environ['ZWID_API_KEY'], 'address': address_param, 'citystatezip': citystatezip_param, \
                       'rentzestimate': 'False'}

            # uploads api
            current_house_info = single_query(api_url_base, payload)

            # api to dataframe
            html_soup = BeautifulSoup(current_house_info, features='html.parser')

            dict = {}
            # creates dictionary
            for child in html_soup.recursiveChildGenerator():
                if child.name in columns:
                    dict[child.name] = html_soup.find(child.name).text
            if i == 0:
                deep_search_df = pd.DataFrame(dict,index=[0])
            else:
                deep_search_df = deep_search_df.append(dict, ignore_index=True)
        except:x=5
    return deep_search_df

def single_query(link, payload):
    '''
    returns api xml file
    '''
    response = requests.get(link, params=payload)
    if response.status_code != 200:
        print('WARNING', response.status_code)
    else:
        return response.text

df['areas'] = df.zipcode.apply(lambda x: mk_area(x))
df['Complement_of_Availability_Next_90_Days']=df.availability_90.apply(lambda x: 90-x)
df['types']= df.property_type.apply(lambda x: types.index(x))
under_1 = df[df.price_bin=='under_100']
between1_2 = df[df.price_bin == '100_200']
over2 =df[df['price_bin'] == '200_or_more']

area_dfs = {}
for area in df.areas.unique():
    area_dfs.update({area:df[df.areas==area]})

plt.style.use('ggplot')
fig,ax = plt.subplots(2,3,figsize=(12,6))
mk_graphs(df)
for i in range(3):
    ax[0][i].set_ylabel("Complement of 90 Day Availability")
    ax[0][i].set_xlabel("Log price")
    ax[1][i].set_ylabel("Number of Properties")
    ax[1][i].set_xlabel('Price')
    ax[1][i].set_title("")

df1 = df.copy()
df1 = df1[(df1.price<200)&(df1.price>100)&(df1.availability_60 !=0)]
lat_long_df = pd.DataFrame()
lat_long_df['latitude']=df['latitude']
lat_long_df['longitude']=df['longitude']
X = lat_long_df
X = X.fillna(0)

zips = df['zipcode'].unique()
zc_dict = {}
for i in zips:
    zip_specific=df[df.zipcode==i]
    zc_dict.update({i:zip_specific})
means = {}
num_listings = {}
for zcode in zc_dict:
    p = zc_dict.get(zcode)
    p = p['price'].astype(float)
    means.update({zcode:np.mean(p)})
    num_listings.update({zcode:len(p)})
z_df=pd.DataFrame.from_dict(means,orient='index')
z_df = z_df.sort_values(by=[0],ascending=False)
z_df=z_df.T
cols = [num_listings]
for i in cols:
    temp = pd.DataFrame.from_dict(i,orient='index')
    temp = temp.T
    z_df=z_df.append(temp,sort=False)
for zi in zc_dict:
    zcode = zc_dict.get(zi)
    for typ in zcode.property_type.unique():
        c=len(zcode[zcode.property_type==typ])
sdu = ['Guesthouse' ,'House','Cottage' ,
          'Bungalow','Bed and breakfast' ,
          'Castle','Villa','Earth house']
mdu = ['Apartment', 'Guest suite',
        'Townhouse','Condominium', 'Loft',
        'Hostel' ,'Camper/RV','Serviced apartment',
        'Tent','Aparthotel', 'Hotel']
df['housing_type'] = df.property_type.apply(lambda x: 'sdu' if x in sdu else 'mdu')
df['z_means'] = df.zipcode.apply(lambda x: '%.2f' % means.get(x) if x==x else 0)

# for i in df.areas.unique():
#     temp = df[df.areas==i]
houses_df = df[df.property_type=='House']
roomtypes = {}
for typ in houses_df.room_type.unique():
    cnt = len(houses_df[houses_df.room_type==typ])
    roomtypes.update({typ:cnt})
roomtypes_dfs_dict = {}

df = df.fillna(0)

lat_long_df = pd.DataFrame()
lat_long_df['latitude']=df['latitude']
lat_long_df['longitude']=df['longitude']
X = lat_long_df


# print(lat_long_df.head())
# loc = geocoder.elevation((39.76703, -105.00256))
# print(loc.meters)

# arr2 = []
# for i in range(len(lat_long_df)):
#     arr2.append(arr[i])
# lat_long_df['key'] = arr
# lat_long_df = lat_long_df[1800:2100]
# lat_long_df['address']=lat_long_df.apply(lambda col: get_addy_info(float(col[0]),float(col[1]),float(col[2])),axis=1)
# lat_long_df.to_csv('address_info.csv')
# adds = pd.read_csv('all_adds.csv')
# print(adds.head(),'length',len(adds))
# cnt = 0
# mask = df['latitude'].isin(adds['latitude']) & df['longitude'].isin(adds['longitude'])
# df = df[mask]
# stop_words = ENGLISH_STOP_WORDS.union(
#     {'home', 'house', 'denver', 'airbnb', 'bedroom', 'bathroom', 'bed','located','access','close','distance','walk','walking','living','enjoy','kitchen','park'})
# vectorizer = TfidfVectorizer(stop_words=stop_words)
# X = vectorizer.fit_transform(df['description'].astype(str))
# features = vectorizer.get_feature_names()
# kmeans = KMeans(n_jobs=-1)
# kmeans.fit(X)
# top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
# print("\n3) top features (words) for each cluster:")
# for num, centroid in enumerate(top_centroids):
#     print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))
# assigned_cluster = kmeans.transform(X).argmin(axis=1)
# df['nlp_topic']=assigned_cluster
# for i in range(8):
#     temp = df[df.nlp_topic==i]
#     print(i,len(temp))
# df.to_csv('checkout.csv')
# df['desc_len'] = df.description.apply(lambda x: len(str(x)))
print(df.head())
# print(df['desc_len'])
df = pd.read_csv('../data/checkout.csv')
del df['Unnamed: 0']
# sns.catplot(x="desc_len", y='price', kind="boxen", data=df)
# plt.show()

# area_means = {}
# area_num_listings = {}
# area_avail_90 = {}
# for area in df.areas.unique():
#     pr = df[df.areas==area]
#     p = pr['price'].astype(float)
#     area_means.update({area: '%.2f' % np.mean(p)})
#     area_avail_90.update({area: '%.2f' % np.mean(pr.availability_90)})
#     area_num_listings.update({area:len(p)})
# df['ar_means'] = df.areas.apply(lambda x: area_means.get(x) if x == x else 0)
# housing_group = df.groupby(by=['areas','housing_type','room_type']).agg({'room_type':'count'})
# df['ar_avail']=df.areas.apply(lambda x: area_avail_90.get(x) if x == x else 0)
# df['w_avail']=df.areas.apply(lambda x: (((len(area_dfs.get(x))/len(df))) * float(area_avail_90.get(x))) if x == x else 0).astype(float)
# def weights(avail_90,weight):
#     return avail_90*weight
# df = df[(df.availability_90!=0)]
#
# feats=['Reservations_In_Next_90_Days','Log_Price']
# for i in feats:
#     sns.catplot(x="zipcode", y=i, kind="boxen", data=df) #save this and resereved 90
#     fname = str(i)
#     plt.savefig(fname,dpi=300)
# plt.tight_layout(pad=3)
#
#
df2 = df.filter(items=['availability_30', 'availability_60',
                       'availability_90', 'availability_365',
                      'price', 'security_deposit', 'accomodates',
                       'bedrooms', 'bathrooms', 'property_type',
                       'room_type', 'latitude', 'longitude',
                       'zipcode','housing_type','nlp_topic','price_bin'])
print(df2.head(),len(df2))
#
# Random Forest ********************************************************************************************************
# for i in df2.price_bin.unique():
#     t = df2[df2.price_bin==i]
# t = df2
# t =t.fillna(0)
# t = pd.get_dummies(t)
# print(t.columns,len(t))
# y = t.pop('price').values
# X = t.values
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# rf = RFR(n_estimators=500)
# rf.fit(X_train, y_train)
# print('rmse:',np.sqrt(mse(y_test,rf.predict(X_test))))
# print("Random Forest score:", rf.score(X_test, y_test))
# imp =  (rf.feature_importances_)
# ord = np.argsort(rf.feature_importances_)[::-1]
# _cols = t.columns.tolist()
# imp_cols = ord[:6]
# # feats = _cols[imp_cols]
# feats = []
# for i in range(len(imp_cols)):
#     for j in _cols:
#         if _cols.index(j) == imp_cols[i]:
#             feats.append(j)
# print(feats)
# x = sorted(imp,reverse=True)[:6]
# imp_feats = {}
# for i in range(len(feats)):
#     imp_feats.update({feats[i]:'%.4f'%x[i]})
# print(imp_feats)
# x = np.array(df.columns.tolist())[idx]
# y = np.array(x)[idx]
#     model = sm.OLS(y_train, X_train)
#     results = model.fit()
#     model.predict(X_test,y_test)
#     print(results.summary())
#
# # ###############################################  AdaBoost  ###########################################################
#
# derf=df
# # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.1)
# x = derf[(derf.property_type=='House')&(derf.room_type=='Entire home/apt')]
# z = x.copy()
# y = z.pop('price').values
# x1 = z.latitude.tolist()
# x2 =z.longitude.tolist()
# near_df = pd.DataFrame()
# near_df['x1']=x1
# near_df['x2']=x2
# X = near_df.values
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.10)
# model_df = pd.DataFrame()
# model_df['lati']= X_train[:, 0]
# model_df['longy']= X_train[:, 1]
# model_df['price']=y_train
# def k_n_n(df,lat,long,neighbs=3):
#     top_n = []
#     ds=[]
#     df['dis'] = df.apply(lambda col: distance((float(col[0]),float(col[1])),(lat,long)).m,axis=1)
#     d = df.dis.tolist()
#     for _ in range(neighbs):
#         ds.append(float(min(d)))
#         top_n.append(df[df.dis==(min(d))].price.tolist()[0])
#         d.remove(min(d))
#         # print(top_n)
#     y = np.array(top_n)
#
#     x = np.mean(y)
#     # print('prices to avg:', y)
#     return x,y,np.array(ds)
#
# # print('%.2f' % distance((39.756896,-104.962213),(39.766896,-105.00018)))
# # # print(X_test)
# arr = []
# p_means=[]
# nn_prices=[]
# nn_dists=[]
# n = 5
# lats = X_test[:,0]
# longs = X_test[:,1]
# for i in range(len(y_train)):
#     a,b,c=k_n_n(model_df, model_df.lati.tolist()[i], model_df.longy.tolist()[i], neighbs=n)
#     a = float(a)
#     p_means.append('%.2f' % a)
#     nn_prices.append(b)
#     nn_dists.append(c)
#     # print('guess:','%.2f' %a,'  act: ',y_train[i], '   diff: ', '%.2f' %(y_train[i]-a))
#     arr.append(float('%.2f' % ((y_train[i]-a)**2)))
# # print('%.2f' % np.sqrt(sum(arr)/len(y_train)),'********************************************')
# res_df=pd.DataFrame()
# nn_prices=np.array(nn_prices)
# nn_dists=np.array(nn_dists)
# for i in range(n):
#     res_df['closest_p'+str(i)]=nn_prices[:,i]
#     res_df['closest_d'+str(i)]=nn_dists[:,i]
# res_df['guess']=p_means
# res_df['act']=y_train
# print(res_df)
# rng = np.random.RandomState(1)
# y = np.array(res_df.pop('act').values)
# X = res_df.values
# X_train,X_test,y_train,y_test= train_test_split(X,y)
# regr_1 = DecisionTreeRegressor(max_depth=4)
# regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
#                           n_estimators=300, random_state=rng)
# regr_1.fit(X_train, y_train)
# regr_2.fit(X_train, y_train)
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
# for i in range(len(y_1)):
#     a=y_2[i]
#     print('guess:','%.2f' %a,'  act: ',y[i], '   diff: ', '%.2f' %(y[i]-a))
#     arr.append(float('%.2f' % ((y[i]-a)**2)))
# print('%.2f' % np.sqrt(sum(arr)/len(y)),'********************************************')
#
# # # ##############################################  OLS  ###########################################################
#
#
# pri = df.price.tolist()
# y = pri
#
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# # model = sm.OLS(pri, X)
# # results = model.fit()
# # model.predict(model)
#
# # # ##############################################  KNN  ###########################################################
#
# temp = df[(df.price>100)&(df.price<200)]
# y = temp.sample(200,axis=0)
# def k_n_n(df,lat,long,neighbs=3):
#     # print(df.head())
#     top_n = []
#     ds=[]
#     df['dis'] = df.apply(lambda col: distance((float(col[15]),float(col[16])),(lat,long)).m,axis=1)
#     d = df.dis.tolist()
#     d.remove(min(d))
#     for _ in range(neighbs):
#         ds.append(float(min(d)))
#         top_n.append(df[df.dis==(min(d))].price.tolist()[0])
#         d.remove(min(d))
#         # print(top_n)
#     y = np.array(top_n)
#
#     x = np.mean(y)
#     # print('prices to avg:', y)
#     return x,y,np.array(ds)

# print('%.2f' % distance((39.756896,-104.962213),(39.766896,-105.00018)))
# # print(X_test)
# arr = []
# p_means=[]
# nn_prices=[]
# nn_dists=[]
# n = 10
# lats = X_test[:,0]
# longs = X_test[:,1]
# prices = y.price.tolist()
# for i in range(len(y)):
#     a,b,c=k_n_n(temp, y.latitude.tolist()[i], y.longitude.tolist()[i], neighbs=n)
#     a = float(a)
#     p_means.append('%.2f' % a)
#     nn_prices.append(b)
#     nn_dists.append(c)
#     # print('guess:','%.2f' %a,'  act: ',y_train[i], '   diff: ', '%.2f' %(y_train[i]-a))
#     arr.append(float('%.2f' % ((prices[i]-a)**2)))
# print('%.2f' % np.sqrt(sum(arr)/len(y)),'********************************************')
# res_df=pd.DataFrame()
# nn_prices=np.array(nn_prices)
# nn_dists=np.array(nn_dists)
# for i in range(n):
#     res_df['closest_p'+str(i)]=nn_prices[:,i]
#     res_df['closest_d'+str(i)]=nn_dists[:,i]
# res_df['guess']=p_means
# res_df['act']=y.price.tolist()
# print(res_df)
# print()
# res_df.to_csv('predictions.csv')
# # print(results.summary())
    # point = point.latitude
# # plt.tight_layout()
# # plt.show()
# # first = zc_dict.get('80212')
# # l1=zc_dict.get('80205').latitude.tolist()
# # l2=zc_dict.get('80205').longitude.tolist()
# # dists = {}
# # for i,j in enumerate(l1):
# #     each_dist=[]
# #     for g in range(1,len(l1)):
# #         dist = int(distance((j,l2[i]),(l1[g],l2[g])).m)
# #         each_dist.append((dist,g))
# #         # print(each_dist)
# #     dists.update({i:sorted(each_dist)[1:4]})
# #     print(dists)
# # print(dists)
# # a = pd.DataFrame.from_dict(dists,orient='index')
# # with open('dists.pickle', 'wb') as handle:
# #     pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # with open('dists.pickle', 'rb') as handle:
# #     b = pickle.load(handle)
# # dist = {}
# # for i in range(1,len(lls)):
# #     dists = []
# #     for j in lls:
# #         d = euclidean_distance(lls[i],j)
# #         dists.append(d)
# #     dists=sorted(dists)
# #     dist.update({lls[i]:dists[:5:]})
# # print(dist)
#
#
# # print(select_df(o2,'property_type','Tent'))
#
# # areas = {'northeast':['80022','80249','80239','80226','80238'],
# #          'east':['80207','80220'],'southeast':['80230','80247',
# #          '80012','80014','80231','80222','80224','80237','80246'],
# #          'south':['80210','80209','80223','80219'],
# #          'southwest':['80123','80236','80235','80227','80127'],
# #          'west':['80228','80232','80226','80214','80215'],
# #          'northwest':['80033','80212','80211','80221'],'north':'80216'}

address_df = pd.read_csv('../data/all_adds.csv')
address_df['is_address']=address_df.address.apply(lambda x: 1 if str(x).split(',')[0].isnumeric() else 0)
address_df = address_df[address_df.is_address==1]
del address_df['is_address']
del address_df['Unnamed: 0']
del address_df['key']
mask = df['latitude'].isin(address_df['latitude']) & df['longitude'].isin(address_df['longitude'])
df = df[mask]
dups = pd.DataFrame()
df['dup_lats'] = df.latitude.duplicated()
df['dup_longs'] = df.longitude.duplicated()
df = df[(df.dup_lats==False)&(df.dup_longs==False)]
del df['dup_lats']
del df['dup_longs']
df.to_csv('dedup_df.csv')
df[(df['availability_90']>70)].to_csv('avail_90.csv')

# # print(len(df))



citystate_arr = []
print(address_df.head())
for i in range(len(address_df)):
    citystate_arr.append('Denver Colorado')
address_df['citystatezip']=citystate_arr
address_df['address']=address_df.address.apply(lambda x: str(str(x).split(',')[0])+str(str(x).split(',')[1]))
x,y = 100,101
print(address_df[x:y])
# x = deep_search_sample(address_df[2000:len(address_df)])
# print(x)
prop_cost_df=pd.read_csv('../data/property_cost.csv')
p2 = pd.read_csv('../data/property_cost2.csv')
prop_cost_df.append(p2)
prop_cost_df=prop_cost_df.fillna(0)
print(prop_cost_df.head())
print(len(prop_cost_df[prop_cost_df.amount!=0]))
# prop_cost_df[prop_cost_df.amount!=0].to_csv('prop_cost.csv')
prop_cost_df = pd.read_csv('../data/prop_cost.csv')
prop_cost_df = prop_cost_df.filter(items=['amount','address'])
address_df = pd.merge(prop_cost_df,address_df,how='left',on=['address'])
address_df = address_df.filter(items=['amount','latitude','longitude'])
print(len(address_df))
address_df = address_df.drop_duplicates()
print(len(address_df))
df = pd.merge(df,address_df,how='left',on=['latitude','longitude'])
df=df.fillna(0)
df =df[df.amount!=0]
print(df.head())
df['cap_rate'] = df.apply(lambda col: (col[7]*(90-col[5])*4.05)/col[24],axis=1)
print(df.head())
# mask = address_df['latitude'].isin(df['latitude']) & address_df['longitude'].isin(df['longitude'])
# address_df = address_df[mask]
# print(address_df.head(),len(address_df))
# for i in range(len(df)):
#     try:
#         tmp_lat = df[i:i+1].latitude.tolist()[0]
#         tmp_long = df[i:i+1].longitude.tolist()[0]
#         amount_arr.append(address_df[(address_df.latitude==tmp_lat)&(address_df.longitude==tmp_long)].amount.tolist()[0])
#     except:
#         x=5
# df['property_cost']=amount_arr
# print(len(df),len(amount_arr))
print(len(df.columns))


df = df.rename(columns={'Reservations_In_Next_90_Days':'Complement_of_Availability_Next_90_Days'})
print(df.columns)
mod = rand_for(df)
plot_feat_imp(mod[0],mod[1],mod[2],fname='feature importances plot')
# m_rf = [0,0]
# for i in range(1000):
#     mod = rand_for(df)
#     rmse =float(mod[4])
#     rfs = float(mod[5])
#     if mod[3].keys() == ['price_bin_200_or_more','cleaning_fee',
#                        'bathrooms','latitude','longitude',
#                        'Compliment_of_Availability','bedrooms'] or 'amount' in mod[3].keys():
#         plot_feat_imp(mod[0],mod[1],mod[2],fname='feature importance plot')
#     if float(rmse)<60 and float(rfs)>75:
#         d = {'RMSE':rmse,'Random Forest Score': rfs}
#         d = pd.DataFrame.from_dict(d,orient='index')
#         d.to_csv('Random Forest Results.csv')
#         break
#     else:
#         if rmse<m_rf[0]:
#             m_rf[0]=rmse
#         if rfs>m_rf[1]:
#             m_rf[1]=rfs
# print('Max RMSE: ', m_rf[0])
# print('Max Random Forest Score: ',m_rf[1])

# print('Average Cap Rate', '%.4f'%np.mean(df.Complement_of_Availability_Next_90_Days))
# plt.style.use('ggplot')
# fig,ax = plt.subplots(2,3,figsize=(12,6))
# mk_graphs(df)
# for i in range(3):
#     ax[0][i].set_ylabel("Complement of 90 Day Availability")
#     ax[0][i].set_xlabel("Price")
#     ax[1][i].set_ylabel("Number of Properties")
#     ax[1][i].set_xlabel('Price')
#     ax[1][i].set_title("")

plt.show()
# df['cost_bin']=pd.cut(df['amount'],[0,436115,744835,max(df.amount)],labels=['low','medium','high']).astype(str)
# by(by=['areas','cost_bin']).agg({'cost_bin':'count'})
