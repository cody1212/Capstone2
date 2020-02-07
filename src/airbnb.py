import pandas as pd
import numpy as np
import seaborn as sns
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
pri = df.price.tolist()
lat_long_df = pd.DataFrame()
lat_long_df['latitude']=df['latitude']
lat_long_df['longitude']=df['longitude']
X = lat_long_df
X = X.fillna(0)
print(len(X))
y = pri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print()
model = sm.OLS(pri, X)
results = model.fit()
model.predict(model)
print(results.summary())
pri = pri.fillna(0)
print(len(pri))
df = df[df.price<2000]
# del df['price']
# p_type = df['property_type']
# del df['property_type']
# r_type= df['room_type']
# del df['room_type']
# >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),
# ...               index=['a', 'b', 'c', 'd', 'e'])
# >>> pd.cut(s, [0, 2, 4, 6, 8, 10], labels=False, retbins=True, right=False)
# bins = 0
# prices =[bins]
# for i in range(10):
#     bins+=50
#     prices.append(bins)
# for i in range(5):
#     bins+=100
#     prices.append(bins)
# for i in range(18):
#     bins+=500
#     prices.append(bins)


# print(df)
df['price_bin']=pd.cut(df['price'],[0,100,200,max(df['price'])],labels=['under_100','100_200','200_or_more']).astype(str)
df['log_price']=np.log10(df.price)
types= list(df.property_type.unique())
t = list(df.price_bin.unique())
areas = {'northeast':['80022','80249','80239','80226','80238'],'east':['80207','80220'],'southeast':['80230','80247','80012','80014','80231','80222','80224','80237','80246'],'south':['80210','80209','80223','80219'],'southwest':['80123','80236','80235','80227','80127'],'west':['80228','80232','80226','80214','80215'],'northwest':['80033','80212','80211','80221'],'north':'80216'}
# for i in areas:
#     for j in areas.get(i):
#         j = float(j)
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
area_dfs = {}
df['areas'] = df.zipcode.apply(lambda x: mk_area(x))
df['reserved_90']=df.availability_90.apply(lambda x: 90-x)
df['types']= df.property_type.apply(lambda x: types.index(x))
# df['t']= df.price_bin.apply(lambda x: t.index(x))
under_1 = df[df.price_bin=='under_100']
b1_2 = df[df.price_bin=='100_200']
o2 =df[df['price_bin']=='200_or_more']
for area in df.areas.unique():
    area_dfs.update({area:df[df.areas==area]})
# print(area_dfs)
# print(df.room_type.unique())


# under_1.hist('areas')
# b1_2.hist('areas')
# o2.hist('areas',color='orange')
# under_1.hist('availability_30')
# b1_2.hist('availability_30')
# o2.hist('availability_30',color='orange')
# under_1.hist('bedrooms')
# b1_2.hist('bedrooms')

# o2.hist('property_type',color='orange')
fig,ax = plt.subplots(2,3,figsize=(12,6))

def mk_hist(df):
    cnter = 0
    for i in ['under_100','100_200','200_or_more']:
        i = df[(df.price_bin == i)&(df.price<2000)]
        g = sns.jointplot(i.log_price,i.reserved_90,kind='kde',color='green',space=1,ax=ax[0][cnter])
        i.hist('price', bins=20, color='maroon',ax=ax[1][cnter])
        cnter+=1

        # plt.savefig(f"{i}",format='png',dpi=300)
mk_hist(df)
for i in range(3):
    ax[0][i].set_ylabel("reserved_within_90_days")
    ax[0][i].set_xlabel("log_price")
    ax[1][i].set_ylabel("counts")
    ax[1][i].set_xlabel('price')
    ax[1][i].set_title("")

# sns.set('10^x')
# df['avail']=df['availability_365']-365
# sns.catplot(x="price_bin", y="reserved_90", kind="box", data=df)

# under_1['property_type'].value_counts().plot(kind='bar',color='gold')
# b1_2['property_type'].value_counts().plot(kind='bar',color='red')
# o2['property_type'].value_counts().plot(kind='bar',color='green')
# print(len(df.property_type.unique()))


def select_df(df,feature,value,feature2=None):
    df=df[df[feature]==value]
    return df

# print(select_df(o2,'property_type','Tent'))
plt.style.use('ggplot')
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde',range_padding=.2)
# plt.xticks(rotation=90)



df1 = df.copy()
df1 = df1[(df1.price<200)&(df1.price>100)&(df1.availability_60 !=0)]
# print(df1)
lat_long_df = pd.DataFrame()
lat_long_df['latitude']=df['latitude']
lat_long_df['longitude']=df['longitude']
X = lat_long_df
X = X.fillna(0)
print(len(X))
y = pri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
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
# print(zc_dict)
means = {}
num_listings = {}
for zcode in zc_dict:
    p = zc_dict.get(zcode)
    p = p['price'].astype(float)
    means.update({zcode:np.mean(p)})
    num_listings.update({zcode:len(p)})
z_df=pd.DataFrame.from_dict(means,orient='index')
z_df = z_df.sort_values(by=[0],ascending=False)
# print(z_df)
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
# print(roomtypes)
# print(z_df.T)
# # print(means)
# # print(num_listings)
# print(zc_dict.get('80212')['price'])

# print(df.property_type.unique())
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
ar_means = {}
ar_num_listings = {}
ar_avail_90 = {}
for area in df.areas.unique():
    pr = df[df.areas==area]
    p = pr['price'].astype(float)
    ar_means.update({area:'%.2f' % np.mean(p)})
    ar_avail_90.update({area:'%.2f' % np.mean(pr.availability_90)})
    ar_num_listings.update({area:len(p)})
df['ar_means'] = df.areas.apply(lambda x: ar_means.get(x) if x==x else 0 )
housing_group = df.groupby(by=['areas','housing_type','room_type']).agg({'room_type':'count'})
df['ar_avail']=df.areas.apply(lambda x: ar_avail_90.get(x) if x==x else 0)
df['w_avail']=df.areas.apply(lambda x: (((len(area_dfs.get(x))/len(df)))*float(ar_avail_90.get(x))) if x ==x else 0).astype(float)
def weights(avail_90,weight):
    # avail_90=float(avail_90)
    # weight=float(weight)
    return avail_90*weight
df = df[(df.availability_90!=0)]

# ws = np.array(df.w_avail.astype(float))/np.array(df.reserved_90.astype(float))
# df['ws']=df.apply(lambda x: '%.2f' % weights(float(x[2]),float(x[23])),axis=1)
# df['ws']=ws
feats=['reserved_90','log_price']
for i in feats:
    sns.catplot(x="zipcode", y=i, kind="boxen", data=df) #save this and resereved 90
    fname = str(i)
    plt.savefig(fname,dpi=300)
plt.tight_layout(pad=3)
# plt.show()


from sklearn.ensemble import RandomForestRegressor as RFR


df2 = df.filter(items=['availability_30', 'availability_60', 'availability_90', 'availability_365',
                      'price', 'cleaning_fee', 'security_deposit', 'accomodates', 'bedrooms',
                      'bathrooms', 'property_type', 'room_type', 'latitude', 'longitude', 'zipcode','housing_type'])
df2 =df2.fillna(0)
df2 = pd.get_dummies(df2)
# print(df2.head(10))
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import pandas as pd

y = df2.pop('price').values
X = df2.values
X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RFR(n_estimators=500)
rf.fit(X_train, y_train)
print('rmse:',np.sqrt(mse(y_test,rf.predict(X_test))))
print("Random Forest score:", rf.score(X_test, y_test))
imp =  (rf.feature_importances_)
ord = np.argsort(rf.feature_importances_)[::-1]
_cols = df2.columns.tolist()
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
# x = np.array(df.columns.tolist())[idx]
# y = np.array(x)[idx]

# print(housing_group)

# p_small = p[p <= 500]
model = sm.OLS(pri, df2)

results = model.fit()
model.predict(model)
print(results.summary())

y = df.pop('price').values
x1 = df.latitude.tolist()
x2 =df.longitude.tolist()
near_df = pd.DataFrame()
near_df['x1']=x1
near_df['x2']=x2

# print(x1,near_df)
X = near_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Add noise to targets
# y[::5] += 1 * (0.5 - np.random.rand(8))
#
# # #############################################################################
# # Fit regression model
n_neighbors = 3
from sklearn.neighbors import KNeighborsRegressor as KNR
nn = KNR(3)
nn.fit(X_train,y_train)
x = nn.predict(X_test)
print(len(X_train),len(y_train))


plt.scatter(X_train, y_train, color='darkorange', label='data')
plt.plot(X_test, x, color='navy', label='prediction')
plt.axis('tight')
plt.legend()
plt.title("KNeighborsRegressor")
#
plt.tight_layout()
plt.show()
# first = zc_dict.get('80212')
# l1=np.array(first['latitude'])
# l2=np.array(first['longitude'])

plt.show()

from geopy.distance import distance
# dist = int(distance(reversed((l1[0],l2[0])),reversed((l1[1],l2[1]))).m)
# print(l1[:2],l2[:2])
# print(dist)
# print(lls )
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