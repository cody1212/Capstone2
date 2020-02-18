import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import math
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

pd.set_option("display.max_rows",1300)
pd.set_option("display.max_columns",12000)
pd.set_option('display.max_rows',200000)
pd.set_option('display.width', 500)

dista=pd.read_csv('distances_df.csv')
dista = dista.rename(columns={'Unnamed: 0': 'ind'})

df = pd.read_csv('../data/denverairbnb/listings.csv')
# print(df.describe())
df = df.filter(items=['availability_30','availability_60','availability_90','availability_365',
                      'price','cleaning_fee','security_deposit','accomodates','bedrooms',
                      'bathrooms','property_type','room_type','latitude','longitude','zipcode'])
lat_long_df = pd.DataFrame()
lat_long_df['latitude']=df['latitude']
lat_long_df['longitude']=df['longitude']
print(df.head())
X = lat_long_df
df = df[df.price<2000]
df['price_bin']=pd.cut(df['price'],[0,100,200,max(df['price'])],labels=['under_100','100_200','200_or_more']).astype(str)
df['Log_Price']=np.log10(df.price)
types= list(df.property_type.unique())

def select_df(df,feature,value,feature2=None):
    df=df[df[feature]==value]
    return df

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
        i = df[(df.price_bin == i)&(df.price<2000)]
        sns.jointplot(i.Log_Price,i.Reservations_In_Next_90_Days,kind='kde',color='green',space=1,ax=ax[0][cnter])
        i.hist('price', bins=20, color='maroon',ax=ax[1][cnter])
        cnter+=1
        # plt.savefig(f"{i}",format='png',dpi=300)

df['areas'] = df.zipcode.apply(lambda x: mk_area(x))
df['Reservations_In_Next_90_Days']=df.availability_90.apply(lambda x: 90-x)
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
    ax[0][i].set_ylabel("Reservations In Next 90 days")
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
houses_df = df[df.property_type=='House']
roomtypes = {}
for typ in houses_df.room_type.unique():
    cnt = len(houses_df[houses_df.room_type==typ])
    roomtypes.update({typ:cnt})

sns.catplot(x="zipcode", y='cleaning_fee', kind="boxen", data=df)

sdu = ['Guesthouse' ,'House','Cottage' ,
          'Bungalow','Bed and breakfast' ,
          'Castle','Villa','Earth house']
mdu = ['Apartment', 'Guest suite',
        'Townhouse','Condominium', 'Loft',
        'Hostel' ,'Camper/RV','Serviced apartment',
        'Tent','Aparthotel', 'Hotel']
df['housing_type'] = df.property_type.apply(lambda x: 'sdu' if x in sdu else 'mdu')
df['z_means'] = df.zipcode.apply(lambda x: '%.2f' % means.get(x) if x==x else 0)
for i in df.areas.unique():
    temp = df[df.areas==i]
area_means = {}
area_num_listings = {}
area_avail_90 = {}
for area in df.areas.unique():
    pr = df[df.areas==area]
    p = pr['price'].astype(float)
    area_means.update({area: '%.2f' % np.mean(p)})
    area_avail_90.update({area: '%.2f' % np.mean(pr.availability_90)})
    area_num_listings.update({area:len(p)})
df['ar_means'] = df.areas.apply(lambda x: area_means.get(x) if x == x else 0)
housing_group = df.groupby(by=['areas','housing_type','room_type']).agg({'room_type':'count'})
df['ar_avail']=df.areas.apply(lambda x: area_avail_90.get(x) if x == x else 0)
df['w_avail']=df.areas.apply(lambda x: (((len(area_dfs.get(x))/len(df))) * float(area_avail_90.get(x))) if x == x else 0).astype(float)
def weights(avail_90,weight):
    return avail_90*weight
df = df[(df.availability_90!=0)]

feats=['Reservations_In_Next_90_Days','Log_Price']
for i in feats:
    sns.catplot(x="zipcode", y=i, kind="boxen", data=df) #save this and resereved 90
    fname = str(i)
    plt.savefig(fname,dpi=300)
plt.tight_layout(pad=3)


df2 = df.filter(items=['availability_30', 'availability_60', 'availability_90', 'availability_365',
                      'price', 'cleaning_fee', 'security_deposit', 'accomodates', 'bedrooms',
                      'bathrooms', 'property_type', 'room_type', 'latitude', 'longitude', 'zipcode','housing_type'])


#Random Forest ********************************************************************************************************
# df2 =df2.fillna(0)
# df2 = pd.get_dummies(df2)
# y = df2.pop('price').values
# X = df2.values
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# rf = RFR(n_estimators=500)
# rf.fit(X_train, y_train)
# print('rmse:',np.sqrt(mse(y_test,rf.predict(X_test))))
# print("Random Forest score:", rf.score(X_test, y_test))
# imp =  (rf.feature_importances_)
# ord = np.argsort(rf.feature_importances_)[::-1]
# _cols = df2.columns.tolist()
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

# model = sm.OLS(pri, df2)
#
# results = model.fit()
# model.predict(model)
# print(results.summary())


# ###############################################  AdaBoost  ###########################################################

derf=df
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.1)
x = derf[(derf.property_type=='House')&(derf.room_type=='Entire home/apt')]
z = x.copy()
y = z.pop('price').values
x1 = z.latitude.tolist()
x2 =z.longitude.tolist()
near_df = pd.DataFrame()
near_df['x1']=x1
near_df['x2']=x2
X = near_df.values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.10)
model_df = pd.DataFrame()
model_df['lati']= X_train[:, 0]
model_df['longy']= X_train[:, 1]
model_df['price']=y_train
# print(shitfuck)
def k_n_n(df,lat,long,neighbs=3):
    top_n = []
    ds=[]
    df['dis'] = df.apply(lambda col: distance((float(col[0]),float(col[1])),(lat,long)).m,axis=1)
    d = df.dis.tolist()
    for _ in range(neighbs):
        ds.append(float(min(d)))
        top_n.append(df[df.dis==(min(d))].price.tolist()[0])
        d.remove(min(d))
        # print(top_n)
    y = np.array(top_n)

    x = np.mean(y)
    # print('prices to avg:', y)
    return x,y,np.array(ds)

# print('%.2f' % distance((39.756896,-104.962213),(39.766896,-105.00018)))
# # print(X_test)
arr = []
p_means=[]
nn_prices=[]
nn_dists=[]
n = 5
lats = X_test[:,0]
longs = X_test[:,1]
for i in range(len(y_train)):
    a,b,c=k_n_n(model_df, model_df.lati.tolist()[i], model_df.longy.tolist()[i], neighbs=n)
    a = float(a)
    p_means.append('%.2f' % a)
    nn_prices.append(b)
    nn_dists.append(c)
    # print('guess:','%.2f' %a,'  act: ',y_train[i], '   diff: ', '%.2f' %(y_train[i]-a))
    arr.append(float('%.2f' % ((y_train[i]-a)**2)))
# print('%.2f' % np.sqrt(sum(arr)/len(y_train)),'********************************************')
res_df=pd.DataFrame()
nn_prices=np.array(nn_prices)
nn_dists=np.array(nn_dists)
for i in range(n):
    res_df['closest_p'+str(i)]=nn_prices[:,i]
    res_df['closest_d'+str(i)]=nn_dists[:,i]
res_df['guess']=p_means
res_df['act']=y_train
print(res_df)
rng = np.random.RandomState(1)
y = np.array(res_df.pop('act').values)
X = res_df.values
X_train,X_test,y_train,y_test= train_test_split(X,y)
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
for i in range(len(y_1)):
    a=y_2[i]
    print('guess:','%.2f' %a,'  act: ',y[i], '   diff: ', '%.2f' %(y[i]-a))
    arr.append(float('%.2f' % ((y[i]-a)**2)))
print('%.2f' % np.sqrt(sum(arr)/len(y)),'********************************************')

# # ##############################################  OLS  ###########################################################


pri = df.price.tolist()
y = pri

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# model = sm.OLS(pri, X)
# results = model.fit()
# model.predict(model)

# # ##############################################  KNN  ###########################################################

# print(results.summary())
# n_neighbors = 3
# nn = KNR(3)
# nn.fit(X_train,y_train)
# x = nn.predict(X_test)
# print(len(X_train),len(y_train))
# plt.tight_layout()
# plt.show()
# first = zc_dict.get('80212')
# l1=zc_dict.get('80205').latitude.tolist()
# l2=zc_dict.get('80205').longitude.tolist()
# dists = {}
# for i,j in enumerate(l1):
#     each_dist=[]
#     for g in range(1,len(l1)):
#         dist = int(distance((j,l2[i]),(l1[g],l2[g])).m)
#         each_dist.append((dist,g))
#         # print(each_dist)
#     dists.update({i:sorted(each_dist)[1:4]})
#     print(dists)
# print(dists)
# a = pd.DataFrame.from_dict(dists,orient='index')
# with open('dists.pickle', 'wb') as handle:
#     pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('dists.pickle', 'rb') as handle:
#     b = pickle.load(handle)
# dist = {}
# for i in range(1,len(lls)):
#     dists = []
#     for j in lls:
#         d = euclidean_distance(lls[i],j)
#         dists.append(d)
#     dists=sorted(dists)
#     dist.update({lls[i]:dists[:5:]})
# print(dist)


# print(select_df(o2,'property_type','Tent'))

# areas = {'northeast':['80022','80249','80239','80226','80238'],
#          'east':['80207','80220'],'southeast':['80230','80247',
#          '80012','80014','80231','80222','80224','80237','80246'],
#          'south':['80210','80209','80223','80219'],
#          'southwest':['80123','80236','80235','80227','80127'],
#          'west':['80228','80232','80226','80214','80215'],
#          'northwest':['80033','80212','80211','80221'],'north':'80216'}
