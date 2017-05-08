---
title: "map_summary"
original_title: "map_summary"
original_file_path: "/Users/dave/Projects/vegas_property/notebooks/map_summary/map_summary.ipynb"
original_extension: ipynb
lines_of_code: 478
tags:
    - jupyter
    - python
    - notebook
layout: notebook

---

{% highlight python %}
%matplotlib inline
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import folium
import numpy as np
from ipywidgets import widgets, interact
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
{% endhighlight %}

## I. Load property locations


{% highlight python %}
property_raw = pd.read_csv('../data/realtor_data.csv')
property_raw.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>address</th>
      <th>bedrooms</th>
      <th>full_bathrooms</th>
      <th>half_bathrooms</th>
      <th>type</th>
      <th>size_sqft</th>
      <th>lot_size</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6812 Mystic Plain Ct Las Vegas NV 89149</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>Single Family Home</td>
      <td>1992</td>
      <td>6098.0</td>
      <td>220000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3416 Goldyke St Las Vegas NV 89115</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>Mfd/Mobile Home</td>
      <td>1782</td>
      <td>8276.0</td>
      <td>60000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9222 Cowboy Rain Dr Las Vegas NV 89178</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>Single Family Home</td>
      <td>1864</td>
      <td>2178.0</td>
      <td>173000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>845 Trotter Cir Las Vegas NV 89107</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>Single Family Home</td>
      <td>4608</td>
      <td>15246.0</td>
      <td>600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>10914 Iris Canyon Ln Las Vegas NV 89135</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>Single Family Home</td>
      <td>3951</td>
      <td>13503.6</td>
      <td>799000</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
property_raw.type.unique()
{% endhighlight %}




    array(['Single Family Home', 'Mfd/Mobile Home',
           'Single Family Home; Ready to build',
           'Condo/Townhome/Row Home/Co-Op', 'Multi-Family Home; Ready to build'], dtype=object)




{% highlight python %}
geocode_raw = pd.read_csv('../data/geocode_results.csv')
geocode_raw.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>ADDRESS</th>
      <th>MATCH_IND</th>
      <th>MATCH_TYPE</th>
      <th>MATCH_ADDRESS</th>
      <th>GEOCODE</th>
      <th>COL7</th>
      <th>COL8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>6812 Mystic Plain Ct, Las Vegas, NV, 89149</td>
      <td>Match</td>
      <td>Non_Exact</td>
      <td>6812 N MYSTIC PLAIN CT, LAS VEGAS, NV, 89149</td>
      <td>-115.29752,36.284943</td>
      <td>626989366.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3416 Goldyke St, Las Vegas, NV, 89115</td>
      <td>Match</td>
      <td>Exact</td>
      <td>3416 GOLDYKE ST, LAS VEGAS, NV, 89115</td>
      <td>-115.07684,36.222168</td>
      <td>201873991.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9222 Cowboy Rain Dr, Las Vegas, NV, 89178</td>
      <td>Match</td>
      <td>Exact</td>
      <td>9222 COWBOY RAIN DR, LAS VEGAS, NV, 89178</td>
      <td>-115.29426,36.010292</td>
      <td>635104692.0</td>
      <td>L</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>845 Trotter Cir, Las Vegas, NV, 89107</td>
      <td>Match</td>
      <td>Exact</td>
      <td>845 TROTTER CIR, LAS VEGAS, NV, 89107</td>
      <td>-115.176125,36.160988</td>
      <td>202003849.0</td>
      <td>L</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>10914 Iris Canyon Ln, Las Vegas, NV, 89135</td>
      <td>Match</td>
      <td>Exact</td>
      <td>10914 IRIS CANYON LN, LAS VEGAS, NV, 89135</td>
      <td>-115.33189,36.139626</td>
      <td>201985267.0</td>
      <td>R</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}
def load_property_data():

    def fix_colnames(df):
        """
        normalize column names to lower case
        """
        df.columns = df.columns.str.lower()
        return df

    def parse_geocode(df):
        """
        split geocode string into lat/lon columns
        """
        geocodes = (df.geocode.str.split(',',1,True)
                    .rename(columns = {0: 'lat', 1: 'lon'})
                    .assign(lat = lambda x: x['lat'].astype('float'),
                            lon = lambda x: x['lon'].astype('float')))
        return df.join(geocodes)
    
    def parse_property_type(df):
        """
        define ready to build properties and 
        remap `Multi-Family Home to `Condo/Townhome`
        """
        prop_types = (df['type'].str.split(';', 1, True)
                      .rename(columns={0: 'prop_type', 1: 'ready_to_build'})
                      .assign(
                          prop_type = lambda x: x['prop_type'].map(
                              lambda y: 'Condo/Townhome/Row Home/Co-Op' 
                              if y=='Multi-Family Home' else y)
                          .astype('category'),
                          ready_to_build = lambda x: ~pd.isnull(x['ready_to_build']) * 1))
        return df.drop(['type'], axis=1).join(prop_types)


    def impute_missing_geocodes(df):
        """
        for locations with a missing geocode impute with average lat/lon for zipcode
        """
        geocode_by_zip = (df.groupby('zip')[['lat','lon']].mean().reset_index()
                          .rename(columns = {'lat': 'avg_lat', 'lon': 'avg_lon'}))
        return (df.merge(geocode_by_zip, on='zip')
                .assign(lat = lambda x: x['lat'].fillna(x['avg_lat']),
                        lon = lambda x: x['lon'].fillna(x['avg_lon']))
                .drop(['avg_lat','avg_lon'], axis=1))

    geocode = (
        pd.read_csv('../data/geocode_results.csv')
        .pipe(fix_colnames)
        .pipe(parse_geocode)
        .assign(zip = lambda x: x['address'].map(lambda y: y.split(',')[-1]))
        .pipe(impute_missing_geocodes)
        .drop(['address', 'match_ind', 'match_type', 'match_address', 'geocode', 'col7', 'col8'], axis=1)
        .assign(geopoint = lambda x: [Point(x) for x in zip(x['lat'], x['lon'])],
                zip = lambda x: x['zip'].str.lstrip()))
    
    median_income_by_zip = (
        pd.read_csv('../data/nv_median_income_by_zip.csv')
        .rename(columns={'zipcode':'zip'})
        .assign(zip = lambda x: x['zip'].map(str)))


    property = (
        pd.read_csv('../data/realtor_data.csv')
        .drop(['address'], axis=1)
        .pipe(parse_property_type)
        .assign(ppsf = lambda x: x['price'] / x['size_sqft'],
                lot_size = lambda x: x['lot_size'].fillna(0)))
    
    merged = (
        property
        .merge(geocode, on='id')
        .merge(median_income_by_zip, how='left', on='zip'))
    
    crs = {'init': u'epsg:4326'}
    
    return gpd.GeoDataFrame(merged, crs=crs, geometry='geopoint')
{% endhighlight %}


{% highlight python %}
property = load_property_data()
property.head()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bedrooms</th>
      <th>full_bathrooms</th>
      <th>half_bathrooms</th>
      <th>size_sqft</th>
      <th>lot_size</th>
      <th>price</th>
      <th>prop_type</th>
      <th>ready_to_build</th>
      <th>ppsf</th>
      <th>lat</th>
      <th>lon</th>
      <th>zip</th>
      <th>geopoint</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1992</td>
      <td>6098.0</td>
      <td>220000</td>
      <td>Single Family Home</td>
      <td>0</td>
      <td>110.441767</td>
      <td>-115.297520</td>
      <td>36.284943</td>
      <td>89149</td>
      <td>POINT (-115.29752 36.284943)</td>
      <td>76.116883</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1782</td>
      <td>8276.0</td>
      <td>60000</td>
      <td>Mfd/Mobile Home</td>
      <td>0</td>
      <td>33.670034</td>
      <td>-115.076840</td>
      <td>36.222168</td>
      <td>89115</td>
      <td>POINT (-115.07684 36.222168)</td>
      <td>29.354028</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1864</td>
      <td>2178.0</td>
      <td>173000</td>
      <td>Single Family Home</td>
      <td>0</td>
      <td>92.811159</td>
      <td>-115.294260</td>
      <td>36.010292</td>
      <td>89178</td>
      <td>POINT (-115.29426 36.010292)</td>
      <td>62.925543</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>4608</td>
      <td>15246.0</td>
      <td>600000</td>
      <td>Single Family Home</td>
      <td>0</td>
      <td>130.208333</td>
      <td>-115.176125</td>
      <td>36.160988</td>
      <td>89107</td>
      <td>POINT (-115.176125 36.160988)</td>
      <td>47.966547</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>3951</td>
      <td>13503.6</td>
      <td>799000</td>
      <td>Single Family Home</td>
      <td>0</td>
      <td>202.227284</td>
      <td>-115.331890</td>
      <td>36.139626</td>
      <td>89135</td>
      <td>POINT (-115.33189 36.139626)</td>
      <td>168.004549</td>
    </tr>
  </tbody>
</table>
</div>



## II. Load map layers from OpenStreetMap (via overpass-turbo.eu)


{% highlight python %}
def custom_leisure_map(x):
    """
    create custom groups of leisure types
    """
    if x['leisure'] == 'track':
        if x['sport'] == 'motor':
            return 'motor_track'
        else:
            return 'running_track'
    elif x['leisure'] == 'private':
        return 'swimming_pool'
    elif x['leisure'] == 'pool':
        return 'water_park'
    elif x['leisure'] in ['plaza', 'amusement_arcade', 'recreation_ground', 'sport', 
                          'aviation', 'trampoline_park', 'video_arcade']:
        return None
    else:
        return x['leisure']

leisure = (
    gpd.read_file('../data/leisure.geojson')
    .assign(leisure = lambda x: x.apply(custom_leisure_map, axis=1)))

leisure.leisure.value_counts()[0:20].plot.bar();
{% endhighlight %}


![png](data:image/png;base64,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
gpaTAAAAAElFTkSuQmCC
)



{% highlight python %}
def show_feature(df, attribute, value):
    rows = df[attribute] == value
    return df.ix[rows, df.columns[df.ix[rows].count() > 0]]

#show_feature(leisure, 'leisure', 'casino')
show_feature(amenity, 'amenity', 'smog')
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




{% highlight python %}
def custom_amenity_map(x):
    """
    create custom groups of leisure types
    """
    if x['amenity'] == 'casino':
        if x['name'] in ['Bellagio','The Mirage',
                         'Treasure Island Hotel and Casino',
                         'Aria Resort & Casino',
                         'Monte Carlo Resort and Casino',
                         'The Venetian Las Vegas','Mandalay Bay',
                         "Bally's Las Vegas","Harrah's Las Vegas",
                   #      'Stratosphere Las Vegas',
                         'The Palazzo',
                         'Encore Casino at Wynn','Wynn Las Vegas Casino']:
            return 'casino_strip'
        else:
            return 'casino_offstrip'
    elif x['amenity'] == 'yes':
        if x['name'] == "Nate's Smog N Go":
            return 'fuel'
        else:
            return None
    elif x['amenity'] == 'prison':
        if x['name'] == "Florence McClure Women's Correctional Center":
            return 'correctional_center'
        else:
            return 'prison'
    elif x['amenity'] in ['bbq', 'drinking_water', 'spa', 'finish line', 'lounge', 
                          'whirlpool', 'bench', 'adult day care', 'courthouse', 
                          'loading_dock', 'parking_entrance', 'bat', 'fountain',
                          'smog', 'ice_cream', 'coworking_space', 'nightclub', 
                          'childcare', 'bicycle_rental', 'self_storage']:
        return None
    else:
        return x['amenity']

amenity = (
    gpd.read_file('../data/amenity.geojson')
    .assign(amenity = lambda x: x.apply(custom_amenity_map, axis=1)))

amenity.amenity.value_counts()[0:20].plot.bar();
{% endhighlight %}


![png](data:image/png;base64,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
/3cUwZC6ZCLLAAAAAElFTkSuQmCC
)



{% highlight python %}
aeroway = gpd.read_file('../data/aeroway.geojson')

aeroway.aeroway.value_counts()[0:20].plot.bar();
{% endhighlight %}


![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAFECAYAAAAp0PVNAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzt3XmcpFV97/HPl0VBFIZlIIRBB3VUFGVxEAIkChiVTYyA
QsBwyehoNIrRm0jujddAzA14r+JyvQYCIioKCBrWaAi7IsIMu2JeICCMbIMBxADK8s0f59RMTU/P
dA9d9TzVD9/369Wvqjr1dJ/f9HT96tRZZZuIiOiu1doOICIihiuJPiKi45LoIyI6Lok+IqLjkugj
IjouiT4iouOS6CMiOi6JPiKi45LoIyI6bo22AwDYaKONPHv27LbDiIiYVhYuXPiA7ZkTXTcSiX72
7NksWLCg7TAiIqYVST+fzHXpuomI6Lgk+oiIjkuij4jouCT6iIiOS6KPiOi4JPqIiI5Loo+I6Lgk
+oiIjhuJBVMTmX3EeVP+GXccvdcAIomImH7Soo+I6Lgk+oiIjkuij4jouCT6iIiOS6KPiOi4JPqI
iI5Loo+I6Lgk+oiIjkuij4jouCT6iIiOS6KPiOi4JPqIiI5Loo+I6Lgk+oiIjkuij4jouCT6iIiO
S6KPiOi4JPqIiI5Loo+I6LhJJXpJd0i6UdJ1khbUsg0kXSDplnq7fi2XpM9LulXSDZK2G+Y/ICIi
Vm5VWvS72t7G9tz6+AjgQttzgAvrY4A9gDn1az7wpUEFGxERq24qXTf7AifX+ycDb+sr/6qLK4EZ
kjadQj0RETEFk030Bv5V0kJJ82vZJrbvAai3G9fyzYC7+r53US1bhqT5khZIWrB48eJnFn1ERExo
jUlet7PtuyVtDFwg6acruVbjlHm5Avt44HiAuXPnLvd8REQMxqRa9Lbvrrf3A98BXgfc1+uSqbf3
18sXAZv3ffss4O5BBRwREatmwkQvaR1JL+jdB94E3AScDRxaLzsUOKvePxv4kzr7Zkfg4V4XT0RE
NG8yXTebAN+R1Lv+G7a/K+lq4HRJ84A7gQPq9ecDewK3Ao8Chw086oiImLQJE73t24Ctxyn/JbD7
OOUGPjCQ6CIiYsqyMjYiouOS6CMiOi6JPiKi45LoIyI6Lok+IqLjkugjIjouiT4iouOS6CMiOi6J
PiKi45LoIyI6Lok+IqLjkugjIjouiT4iouOS6CMiOi6JPiKi45LoIyI6Lok+IqLjkugjIjouiT4i
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
TkSuQmCC
)



{% highlight python %}
def map_feature(df, attribute, value):
    filtered_df = df[df[attribute] == value]

    m = folium.Map([36.1699, -115.1398], zoom_start=10, tiles='cartodbpositron')

    polygons = filtered_df[filtered_df.geometry.type == 'Polygon']
    linestrings = filtered_df[filtered_df.geometry.type == 'LineString']
    points = filtered_df[filtered_df.geometry.type == 'Point']

    folium.features.GeoJson(polygons, style_function=lambda x: {'weight' : 2}).add_to(m)
    folium.features.GeoJson(linestrings, style_function=lambda x: {'weight' : 2}).add_to(m)

    for _, feature in points.iterrows():
        x, y = feature.geometry.coords.xy
        folium.CircleMarker([y[0],x[0]], radius=1, color='blue', weight=1).add_to(m)
    return m
{% endhighlight %}


{% highlight python %}

{% endhighlight %}


{% highlight python %}
locations = (
    leisure.assign(feature = lambda x: x['leisure'].map('leisure_{}'.format))
    .append(amenity.assign(feature = lambda x: x['amenity'].map('amenity_{}'.format)))
    .append(aeroway.assign(feature = lambda x: x['aeroway'].map('aeroway_{}'.format))))

location_features = locations.feature.unique().tolist()
location_features.sort()

def f(x):
    if x is None:
        pass
    else:
        return map_feature(locations, 'feature', x)

interact(f, x=widgets.Dropdown(options=location_features, description="select feature:"));
{% endhighlight %}



## III. Enrich properties with minimum distances to locations


{% highlight python %}
%%time
def min_distance_to_feature(df, locations, attribute, value):
    """
    find the minimum distance between properties in `df` and locations 
    filtered by attribute value.
    ex. minimum distance to `leisure=golf_course` for each property
    """
    features = locations[locations[attribute] == value]
    return df.geometry.map(lambda p: min(map(lambda x: x.distance(p), features.geometry))) 

def append_feature_distances(df, locations, feature_type):
    df_enriched = df.copy()
    features = [x for x in locations[feature_type].unique() if x is not None]
    for feature in features:
        df_enriched['{}_{}'.format(feature_type, feature)] = \
            min_distance_to_feature(df, locations, feature_type, feature)
    return df_enriched
        
def append_all_feature_distance(df):
    df_enriched = df.copy()
    df_enriched = append_feature_distances(df_enriched, leisure, 'leisure')
    df_enriched = append_feature_distances(df_enriched, amenity, 'amenity')
    df_enriched = append_feature_distances(df_enriched, aeroway, 'aeroway')
    return df_enriched
        
property_enriched = property.pipe(append_all_feature_distance)
{% endhighlight %}

    CPU times: user 7min 1s, sys: 928 ms, total: 7min 2s
    Wall time: 7min 3s



{% highlight python %}
# define target and predictors
ignore_columns = ['amenity_charging_station', 'amenity_telephone', 'amenity_antiques', 'amenity_oil_tank', 
                  'leisure_yes', 'amenity_post_box', 'aeroway_gate', 'amenity_parking_space',
                  'leisure_picnic_table', 'leisure_disc_golf_course', 'leisure_spa', 'aeroway_apron',
                  'aeroway_parking_position', 'aeroway_hangar', 'amenity_veterinary', 'amenity_grave_yard', 
                  'leisure_dog_park']

y = property_enriched['ppsf']
X_all = (property_enriched
         .drop(['id', 'price', 'geopoint', 'ppsf'], axis=1)
         .drop(ignore_columns, axis=1)
         .assign(prop_type = lambda x: pd.Categorical(x['prop_type'], 
                                              categories=['Single Family Home',
                                                          'Condo/Townhome/Row Home/Co-Op',  
                                                          'Mfd/Mobile Home'],
                                              ordered=True)))

selected_features = ['bedrooms', 'full_bathrooms', 'half_bathrooms', 'size_sqft', 'lot_size', 'prop_type', 
                     'ready_to_build', 'lat', 'lon']

X = (X_all[selected_features]
     .assign(prop_type = lambda x: x['prop_type'].cat.codes))
#             zip = lambda x: pd.Categorical(x['zip'], ordered=False).codes))

for c in X.select_dtypes(['object']):
    X.ix[:,c] = X[c].astype('category').cat.codes
    
X = pd.get_dummies(X)
{% endhighlight %}


{% highlight python %}
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from util import GridSearchBagging
{% endhighlight %}


{% highlight python %}
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_score_mse(estimator, X, y):
    estimator.fit(X, y)
    y_pred = estimator.oob_prediction_
    return -mean_squared_error(y, y_pred)

def get_score_gini(estimator, X, y):
    estimator.fit(X, y)
    y_pred = estimator.oob_prediction_
    return util.weighted_gini(y, y_pred, 1.0)
{% endhighlight %}


{% highlight python %}
model = RandomForestRegressor(n_estimators=100,
                              max_features=0.4,
                              max_depth=25,
                              min_samples_leaf=2,
                              oob_score=True, 
                              random_state=42, 
                              n_jobs=-1)

parameters = {'max_depth': [17, 18, 19],
              'max_features': [0.15, 0.20, 0.25],
              'min_samples_leaf': [1, 2, 4]}

grid = GridSearchBagging(model, param_grid=parameters, scorer=get_score_mse)
{% endhighlight %}


{% highlight python %}
model = GradientBoostingRegressor(random_state=42, verbose=0)

parameters = {
     'n_estimators': [200],
     'learning_rate': [0.05],
     'max_depth': [13, 14, 15],
     'max_features': [0.4, 0.45, 0.5],
     'min_samples_leaf': [13, 14, 15],
     'subsample': [0.6, 0.65, 0.7]}

grid = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
{% endhighlight %}


{% highlight python %}
grid.fit(X, y);
{% endhighlight %}


{% highlight python %}
print grid.best_params_
#print -grid.best_score_
print np.sqrt(-grid.best_score_)
{% endhighlight %}

    {'learning_rate': 0.05, 'min_samples_leaf': 14, 'n_estimators': 200, 'subsample': 0.65, 'max_features': 0.45, 'max_depth': 14}
    94.1260643808



{% highlight python %}
#93.4505890381 - 200
#93.6306889655
{% endhighlight %}


{% highlight python %}
print grid.best_params_
print np.sqrt(-grid.best_score_)
{% endhighlight %}

    {'learning_rate': 0.15, 'min_samples_leaf': 1, 'subsample': 0.9, 'max_features': 0.4, 'alpha': 0.9, 'max_depth': 6}
    94.841438411



{% highlight python %}
feature_importance = grid.best_estimator_.feature_importances_
features_list = X.columns
 
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

# A threshold below which to drop features from the final data set. Specifically, this number represents
# the percentage of the most important feature's importance value
fi_threshold = 0
 
# Get the indexes of all features over the importance threshold
important_idx = np.where(feature_importance > fi_threshold)[0]
 
# Create a list of all the feature names above the importance threshold
important_features = features_list[important_idx]
#print "n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance):n", important_features

# Get the sorted indexes of important features
sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
#print "nFeatures sorted by importance (DESC):n", important_features[sorted_idx]

# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
plt.yticks(pos, important_features[sorted_idx[::-1]])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.draw()
plt.show();
{% endhighlight %}


![png](data:image/png;base64,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
NyshB75ZCf1/pbqoVA5x0sgAAAAASUVORK5CYII=
)



{% highlight python %}
import util
{% endhighlight %}


{% highlight python %}
model = RandomForestRegressor(n_estimators=100,
                              max_features=0.3,
                              max_depth=14,
                              min_samples_leaf=1,
                              oob_score=True, 
                              random_state=42, 
                              n_jobs=-1)
{% endhighlight %}


{% highlight python %}
X = X_all.drop(['zip'], axis=1).assign(prop_type = lambda x: x['prop_type'].cat.codes)

for c in X.select_dtypes(['object']).columns:
    X.ix[:,c] = X[c].astype('category').cat.codes
{% endhighlight %}


{% highlight python %}
['bedrooms', 'full_bathrooms', 'half_bathrooms', 'size_sqft',
 'lot_size', 'prop_type', 'ready_to_build', 'lat',
 'lon']

{% endhighlight %}




    Index([u'id', u'bedrooms', u'full_bathrooms', u'half_bathrooms', u'size_sqft',
           u'lot_size', u'price', u'prop_type', u'ready_to_build', u'ppsf', u'lat',
           u'lon', u'zip', u'geopoint', u'median_income'],
          dtype='object')




{% highlight python %}
initial_selection = ['leisure_motor_track', 'median_income', 
                     'ready_to_build', 'bedrooms', 'lot_size', 'prop_type', 
                     'size_sqft']

forward_scores = util.forward_selection(model, X, y, get_score_gini, initial_selected=initial_selection, k=15)
{% endhighlight %}

      7%|▋         | 1/15 [00:36<08:29, 36.42s/it]


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-1055-b2eba5e5b94f> in <module>()
          3                      'size_sqft']
          4 
    ----> 5 forward_scores = util.forward_selection(model, X, y, get_score_gini, initial_selected=initial_selection, k=15)
    

    /Users/dave/Projects/vegas_property/notebooks/util.pyc in forward_selection(estimator, X, y, get_score, initial_selected, k)
         48             trial = list(set.union(set(selected), set([c])))
         49             X_trial = X[trial]
    ---> 50             trial_score = get_score(estimator, X_trial, y)
         51             if best_score is None or trial_score > best_score['score']:
         52                 best_score = {'n': len(trial), 'score' : trial_score, 'features': trial, 'add': c}


    <ipython-input-980-5f27a56cc5ce> in get_score_gini(estimator, X, y)
          8 
          9 def get_score_gini(estimator, X, y):
    ---> 10     estimator.fit(X, y)
         11     y_pred = estimator.oob_prediction_
         12     return util.weighted_gini(y, y_pred, 1.0)


    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/site-packages/sklearn/ensemble/forest.pyc in fit(self, X, y, sample_weight)
        312             for i in range(n_more_estimators):
        313                 tree = self._make_estimator(append=False,
    --> 314                                             random_state=random_state)
        315                 trees.append(tree)
        316 


    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/site-packages/sklearn/ensemble/base.pyc in _make_estimator(self, append, random_state)
        122 
        123         if random_state is not None:
    --> 124             _set_random_states(estimator, random_state)
        125 
        126         if append:


    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/site-packages/sklearn/ensemble/base.pyc in _set_random_states(estimator, random_state)
         50 
         51     if to_set:
    ---> 52         estimator.set_params(**to_set)
         53 
         54 


    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/site-packages/sklearn/base.pyc in set_params(self, **params)
        270             # Simple optimisation to gain speed (inspect is slow)
        271             return self
    --> 272         valid_params = self.get_params(deep=True)
        273         for key, value in six.iteritems(params):
        274             split = key.split('__', 1)


    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/site-packages/sklearn/base.pyc in get_params(self, deep)
        233         """
        234         out = dict()
    --> 235         for key in self._get_param_names():
        236             # We need deprecation warnings to always be on in order to
        237             # catch deprecated param values.


    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/site-packages/sklearn/base.pyc in _get_param_names(cls)
        205         init_signature = signature(init)
        206         # Consider the constructor parameters excluding 'self'
    --> 207         parameters = [p for p in init_signature.parameters.values()
        208                       if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        209         for p in parameters:


    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/collections.pyc in values(self)
        121     def values(self):
        122         'od.values() -> list of values in od'
    --> 123         return [self[key] for key in self]
        124 
        125     def items(self):


    KeyboardInterrupt: 



{% highlight python %}
print 'best score: {:.4f}'.format(np.max(forward_scores.score))
print 'best features: {}'.format(forward_scores.ix[np.argmax(forward_scores.score),:]['features'])
forward_scores.sort_values('n').plot(x='n', y='score');
{% endhighlight %}

    best score: 0.2732
    best features: ['amenity_swingerclub', 'amenity_veterinary', 'leisure_motor_track', 'median_income', 'ready_to_build', 'bedrooms', 'amenity_grave_yard', 'amenity_correctional_center', 'lot_size', 'prop_type', 'leisure_dog_park', 'size_sqft']



![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAY0AAAEKCAYAAADuEgmxAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xl4lOW5+PHvPdlD9g0IIRuy72SCiIIbBawVrSJCa6tW
tMdqT7W1tra1/rqc01a7n9paV9wOi7ttMSAWAVkkYU8IO9mALBAgkJD9+f0xgyfGAEOSmXeW+3Nd
XEzeed/nvSfL3PPsYoxBKaWUcoXN6gCUUkr5Dk0aSimlXKZJQymllMs0aSillHKZJg2llFIu06Sh
lFLKZZo0lFJKuUyThlJKKZdp0lBKKeWyYKsD6A1JSUkmMzPT6jCUUsqnbNq06agxJvlirvGLpJGZ
mUlBQYHVYSillE8RkdKLvUabp5RSSrlMk4ZSSimXadJQSinlMr/o01BKKVe0tLRQUVFBY2Oj1aF4
VHh4OGlpaYSEhPS4LE0aSqmAUVFRQXR0NJmZmYiI1eF4hDGGY8eOUVFRQVZWVo/L0+YppVTAaGxs
JDExMWASBoCIkJiY2Gu1K00aSqmAEkgJ46zefM2aNNzgVGMLi/PL0K10lVL+xqWkISIzRWS3iOwT
kR928fx3RWSniGwXkQ9FJMN5/GoR2drhX6OI3OR87nkR2ea85g0RiTpfWb7k2dUH+MGbO9hcdtzq
UJRSqlddMGmISBDwFHAdMAKYJyIjOp22BbAbY8YAbwBPABhjVhpjxhljxgHXAA3Acuc1Dxljxjqv
KQMeOF9ZvqK1rZ0lBRUA5Jdo0lBKuU9ra6vH7+lKTWMisM8Yc8AY0wwsAm7seIIzOTQ4v9wApHVR
zmzg/bPnGWPqAMTR2BYBmIsoy2ut2lNDZV0jQTahQJOGUqqT+vp6rr/+esaOHcuoUaNYvHgx+fn5
TJ48mbFjxzJx4kROnTpFY2Mjd911F6NHj2b8+PGsXLkSgAULFnDrrbdyww03MH36dACefPJJcnNz
GTNmDI8//rhb43dlyO0AoLzD1xXApec5/27g/S6OzwV+3/GAiLwIfBHYCXzvIsryWgs3lpMUFcbU
wUms3F2NMSYgO96U8nY/+0cROw/X9WqZI1JjePyGkec9Jy8vj9TUVP71r38BcPLkScaPH8/ixYvJ
zc2lrq6OiIgI/vSnPwGwY8cOdu3axfTp09mzZw8A69evZ/v27SQkJLB8+XL27t3Lxo0bMcYwa9Ys
Vq9ezdSpU3v1tZ3lSk2jq3e8Lnt4ReR2wA482el4f2A0sOwzhRhzF5AKFAO3uVJWh+fvFZECESmo
qalx4WW4X1VdIyt3VzM7J41J2Ykcb2hhf0291WEppbzI6NGjWbFiBT/4wQ9Ys2YNZWVl9O/fn9zc
XABiYmIIDg7m448/5mtf+xoAw4YNIyMj49Ok8YUvfIGEhAQAli9fzvLlyxk/fjwTJkxg165d7N27
123xu1LTqAAGdvg6DTjc+SQRmQb8GLjSGNPU6ek5wNvGmJbO1xlj2kRkMfB94EUXyjp73TPAMwB2
u90rhim9XlBOW7thbu5A2pwjpzaV1nJJSpTFkSmlOrtQjcBdhgwZwqZNm1i6dCmPPvoo06dP77I1
4nyjL/v06fOZ8x599FG++c1vuiXezlypaeQDg0UkS0RCcTQzvdfxBBEZD/wdmGWMqe6ijHnAwg7n
i4hccvYxcAOwy8WyvFJ7u2FxQTmXZSeSmdSH7KQ+JPQJ1c5wpdRnHD58mMjISG6//XYefvhhNmzY
wOHDh8nPzwfg1KlTtLa2MnXqVF577TUA9uzZQ1lZGUOHDv1ceTNmzOCFF17g9OnTABw6dIjqave9
dV6wpmGMaRWRB3A0LQUBLxhjikTk50CBMeY9HE1IUcDrzoxZZoyZBSAimThqKqs6FCvASyIS43y8
DbjP+dw5y/Jm6/Yfo7z2DA9Pd/xQRYScjHg2lWrSUEr9nx07dvD9738fm81GSEgIf/vb3zDG8O1v
f5szZ84QERHBihUr+Na3vsV//Md/MHr0aIKDg1mwYAFhYWGfK2/69OkUFxdz2WWXARAVFcWrr75K
SkqKW+IXf5iAZrfbjdWbMN3/v5tZu+8oGx69lvCQIAD+vmo/v3p/F/k/nkZy9Od/2EopzyouLmb4
8OFWh2GJrl67iGwyxtgvphydEd4Ljp1uYnlRJTePT/s0YQDYMx0dVVrbUEr5C00aveCtzYdoaTPM
nTjwM8dHDYghNNhGQUmtRZEppVTv0qTRQ8YYFuaXMSE9jiF9oz/zXFhwEOPS4ijQmoZSXsMfmuQv
Vm++Zk0aPZRfcpwDNfXMnZje5fM5mfEUHjrJmeY2D0emlOosPDycY8eOBVTiOLufRnh4eK+Up5sw
9dCi/DKiw4L50pj+XT6fmxnP3z4ybKs4waTsRA9Hp5TqKC0tjYqKCrxlQrCnnN25rzdo0uiBk2da
WLrjCLdMSCMytOtv5YT0eMDRGa5JQylrhYSE9MrudYFMm6d64N2th2hsaWfeOZqmAOIiQxnSN4p8
7QxXSvkBTRrdZIxh4cZyRg2IYdSA2POem5ORwKbS47S3B047qlLKP2nS6KbtFScpPlLHbbnnrmWc
Zc+I51RjK3uqT3kgMqWUch9NGt20KL+MiJAgbhyXesFzc52T/HR/DaWUr9Ok0Q31Ta28t/Uw14/p
T0x4yAXPH5gQQXJ0mE7yU0r5PE0a3fDP7Yepb25jXqcZ4OciIuRmxuskP6WUz9Ok0Q0LN5YzOCXq
0+G0rsjJSKDi+BkqTza6MTKllHIvTRoXaVdlHVvLTzB3YvpFbeOam+lIMAWl2kSllPJdmjQu0qKN
5YQG2fjy+AEXdd3w/jFEhARpZ7hSyqdp0rgIjS1tvLW5ghmj+pHQJ/Sirg0JsjE+PU5rGkopn6ZJ
4yK8X3iEusZW5uW61gHemT0jnp2H6zjd1NrLkSmllGdo0rgIizaWk5EY2e01pOyZCbQb2Fp2opcj
U0opz9Ck4aIDNaf55GAtt+UOxGZzvQO8o/HpcdhEO8OVUr5Lk4aLFueXE2wTZud0f3nh6PAQhvaL
0c5wpZTP0qThgubWdt7YVMG1w1NIie7ZRia5mfFsKTtOa1t7L0WnlFKeo0nDBSuKqzhW38xcFxYn
vJCcjHjqm9vYVamLFyqlfI8mDRcs3FhGamw4U4ck97is/1u8UPs1lFK+R5PGBZTXNvDxvqPcah9I
UDc7wDtKjYsgNTacfF2HSinlgzRpXMDrBeUAzOnm3Iyu2DMTKCipDajN7ZVS/kGTxnm0trWzpKCC
K4ckMyAuotfKtWfGU1XXRMXxM71WplJKeYImjfNYtaeGyrrGXukA78ie4ejX2KRNVEopH6NJ4zwW
biwnKSqMa4en9Gq5Q/tFEx0WrJP8lFI+R5PGOVTVNbJydzWzc9IICerdb1OQTRifEa+T/JRSPkeT
xjm8samCtnbD3F7sAO/InhHP7qpTnDzT4pbylVLKHTRpdKG93bAov4zLshPJTOrjlnvYM+IxBjaX
aW1DKeU7NGl0Yd3+Y5TXnmGui3uAd8e49DiCbMImbaJSSvkQTRpdWJhfRlxkCDNG9nPbPSJDgxmZ
GkO+zgxXSvkQTRqdHDvdxPKiSm4en0Z4SJBb72XPSGBbxQmaW3XxQqWUb9Ck0clbmw/R0mbc2jR1
lj0znsaWdooOn3T7vZRSqjdo0ujAGMPC/DImpMcxpG+02+9nz4gHdJKfUsp3aNLooKD0OAdq6pk7
sXdngJ9LSkw46QmR2q+hlPIZmjQ6WLixjOiwYL40pr/H7mnPjGdT6XFdvFAp5RNcShoiMlNEdovI
PhH5YRfPf1dEdorIdhH5UEQynMevFpGtHf41ishNzueeF5FtzmveEJEo5/EwEVnsvNcnIpLZey/3
3E6eaWHpjiPMGpdKZGiwJ24JODrDj55upvRYg8fuqZRS3XXBpCEiQcBTwHXACGCeiIzodNoWwG6M
GQO8ATwBYIxZaYwZZ4wZB1wDNADLndc8ZIwZ67ymDHjAefxu4Lgx5hLgD8BvevICXfXu1kM0trQz
z0NNU2fZMx39GtpEpZTyBa7UNCYC+4wxB4wxzcAi4MaOJziTw9mPyhuAtC7KmQ28f/Y8Y0wdgIgI
EAGcbZ+5EXjJ+fgN4FrnOW5jjGHhxnJGpsYwakCsO2/1OZckRxEbEaKd4Uopn+BK0hgAlHf4usJ5
7FzuBt7v4vhcYGHHAyLyIlAJDAP+p/P9jDGtwEkgsXNhInKviBSISEFNTY0LL+PctlecpPhIncc6
wDuy2YScjHitaSilfIIrSaOrT/ld9tqKyO2AHXiy0/H+wGhg2WcKMeYuIBUoBm67mPsZY54xxtiN
Mfbk5J7t3b0ov5yIkCBuHJfao3K6y54Zz/6aemrrmy25v1JKucqVpFEBdJzplgYc7nySiEwDfgzM
MsY0dXp6DvC2MeZzS7oaY9qAxcAtne8nIsFALOC2j+H1Ta28t/UQ14/pT0x4iLtuc166KZNSyle4
kjTygcEikiUioTiamd7reIKIjAf+jiNhVHdRxjw6NE2JwyVnHwM3ALucT78H3OF8PBv4t3HjeNR/
bj9MfXMb8zwwA/xcxqTFEhpk002ZlFJe74JjS40xrSLyAI6mpSDgBWNMkYj8HCgwxryHozkqCnjd
2WddZoyZBeAcMjsQWNWhWAFeEpEY5+NtwH3O554HXhGRfThqGHN7+iLPZ+HGcganRDEhPd6dtzmv
8JAgRg2I0U2ZlFJez6UJCcaYpcDSTsd+2uHxtPNcW0KnjnNjTDtw+TnObwRudSWuntpVWcfW8hP8
5PrhuHmA1gXlZibw4toSGlva3L5QolJKdVdAzwhftLGc0CAbN0/oaoSwZ+VkxNPc1s6OQ7p4oVLK
ewVs0mhsaeOtzRXMGNWPhD6hVodDjnPxQm2iUkp5s4BNGnmFldQ1tjLPTXuAX6zEqDCyk/uwSTvD
lVJeLGCTxsKNZWQkRjIp+3PzBi1jz4inoPQ47e26eKFSyjsFZNI4UHOaTw7WclvuQGw2azvAO7Jn
JnCioYUDR09bHYpSSnUpIJPG4vxygmzC7BzrO8A7OrspU772ayilvFTAJY3m1nbe2FTBtcNSSIkO
tzqcz8hK6kNin1DtDFdKea2ASxoriqs4Vt/s8SXQXSHiWLxQZ4YrpbxVwCWNRfnlpMaGM3VIzxY5
dJfczARKjzVQfarR6lCUUupzAipplNc2sGZvDbfaBxLkRR3gHeU4N2XapE1USikvFFBJ4/UCx7Yg
c7xkbkZXRqXGEhZso0BXvFVKeaGASRqtbe0sKajgyiHJDIiLsDqccwoNtjF2YBwFuimTUsoLBUzS
WLWnhsq6RuZ6cS3jrNzMeIoO13Gmuc3qUJRS6jMCJmks3FhOUlQY1w7va3UoF2TPSKC13bC1/ITV
oSil1GcERNKoqmtk5e5qZuekERLk/S/57N4e2kSllPI23v8O2gve2FRBW7vxiaYpgNjIEIb2jdbO
cKWU1/H7pNHebliUX8Zl2YlkJvWxOhyX5WTGs7n0OG1+vnjhrso6Dp84Y3UYSikX+X3SWLf/GOW1
Z5hr4R7g3ZGbGc+pplb2VJ2yOhS3OXq6iVl/WcuUJ1by7YVb2KZ9OEp5Pb9PGgvzy4iLDGHGyH5W
h3JR7BkJgH/3ayz8pIzm1nbm2NP4aFc1Nz61llufXseyokq/r2Ep5av8OmkcO93E8qJKvjx+gM/t
u50WH0HfmDC/7ddoaWvn1U9KmTI4iV/dPIZ1j17DY18aweETjXzzlU1c87uPeGldCfVNrVaHqpTq
wK+TxttbDtHSZrxyccILERHsGQl+u+JtXmElVXVN3Dk5E4Do8BDuviKLVd+/iqe+MoH4yFAef6+I
y371Ib9+fxdHTmq/h1LewG+ThjGGhRvLmJAex5C+0VaH0y32zHgOnTjjlx3FL60rIT0hkquGpnzm
eHCQjevH9Oed+y/nzfsmc8XgJJ5ZvZ8pv1nJQ4u3UnjopEURK6UAgq0OwF0KSo+zv6aeJ2aPsTqU
bvu0X6P0OLO8eOmTi1V46CQFpcf5yfXDz7twZE5GPDkZOZTXNvDC2oMsyS/n7S2HmJSdwPwrsrlm
WIpX7byoVCDw25rGwo1lRIcF86Ux/a0OpduG948mMjSITX7WGb5gXQkRIUHcandtRNvAhEgev2Ek
6x69lh99cRilxxqY/3IB036/ilc2lOpyK0p5kF8mjZNnWli64wizxqUSGeq7langIBvj0+P8qjP8
2Okm3tt2mFtyBhAbEXJR18ZGhHDv1EGsfuRq/jxvPFHhwTz2TiGX/fpDfrtsN9V1ugeJUu7ml0nj
3a2HaGxpZ26u73WAd5aTkUDxkTpO+8kookX55TS3tnPHZZndLiMkyMassam8e//lLPnmZUzMTOCp
j/Zx+W/+zfeWbKP4SF3vBax8yqnGFu5ekM/2Cp3z4y6++zH8HBwd4OWMTI1hdFqs1eH0WG5mPO0G
tpQdZ8pg79xt0FUtbe28sr6UKy5JYnAvDE4QESZmJTAxK4GSo/W8uPYgSwoqeHNzBVdcksTdU7K4
cnCy9nsEkOVFVXy4q5r9NadZ+p0pPt3S4K38rqax49BJio/UMdcHh9l2ZXx6PDaBfD8Yeru8qIrK
ukbucA6z7U2ZSX342Y2jWP/oNTwycyh7q09x14v5TP/jahZtLKOxRfs9AkFeUSXR4cGU1jbw30uL
rQ7HL/ld0li4sZyIkCBuHJdqdSi9IiosmOH9Y9hU6vud4S+tK2FgQgTXDEu58MndFBcZyreuuoQ1
j1zDH24bS2iQjR++tYPLf/1v/vDBHo6ebnLbvZW16ptaWb2nhlsmpHH35Vm8uqGMVXtqrA7L7/hV
0qhvauW9rYe4fkx/YsIvrpPVm9kz4tlSdoLWtnarQ+m2osMn2VhSy9cnZXpkf/bQYBtfHp/Gv/7z
ChbeM4nx6XH86cO9TP71v/nBG9v9ek2vQLVqTw1Nre3MHNWPh2cMZXBKFN9/fRsnGpqtDs2v+FXS
+Of2w9Q3tzHPxxYnvBB7ZgINzW0UH/HdN7qXnMNs57g4zLa3iAiXDUrkuTty+fB7V3JrThrvbjvE
9D+s5o4XNrJmbw3G6DpX/iCvsJLEPqHkZiYQHhLEH24bR219M4+9W2R1aH7Fr5LGwo3lDE6J+nQT
I39hz3S8nnwfna9RW9/MO1sP8+UJA4iNtK4GOCg5iv/68mjW/fBaHp4+hJ1H6vja8xuZ+cc1LCko
p6lV+z18VVNrG//eVc30kX0/rcmOGhDLd64dzD+2Hea9bYctjtB/+E3S2FVZx9byE9yWOxAR/xot
0z82ggFxEWzy0fkai/LLejzMtjcl9AnlgWsG8/EPrua3t45FBB55YzuX/3olf/5wL7X12pzha9bu
O8rpptbPrWZ931WDGJ8ex2PvFFJ5Uufx9Aa/SRqLNpYTGmTj5glpVofiFvbMePJLan2uKaW1rZ1X
15cyeVAiQ/t51xpgYcFBzM5J4/3vTOHVuy9l1IAYfv/BHi771Yc8//FBq8NTFyGvsJLosGAmD0r6
zPHgIBu/nzOO5tZ2Hnlzu8/9/Xgjv0gaxjhWtJ0xqh8JfUKtDsct7JkJVJ9qouK4by1e+MHOKg6f
dM8w294iIlwxOIkFd03kg4emYs+M54m8XRzTkVY+obWtnQ92VnHt8BRCgz//lpaV1IcffXEYq/fU
8OqGUgsi9C9+kTROnmnh5JkW5vnIHuDdYc9w9GsU+NjQ2wXrShgQF8G04X2tDsUlg/tG87NZo2hq
becVfYPxCRtLajne0MLMUefeaO32SRlMHZLMfy0t5uDReg9G53/8ImnU1jeTkRjJpOxEq0NxmyF9
o4kOC/apSX7FR+r45GAtX78swyPDbHvLJSlRTBuewsvrS3VSoA9YVlhJeIiNqUPOvWKCiPDELWMI
Cw7iocVbfXr4utVcShoiMlNEdovIPhH5YRfPf1dEdorIdhH5UEQynMevFpGtHf41ishNzudec5ZZ
KCIviEiI83isiPxDRLaJSJGI3HWh+OqbW7ktd6BfLxcRZBMmZMSzyYeSxkvrSggPsXGbD9YA50/J
pra+mTc3V1gdijqP9nbDsqIqrhySfMElQ/rFhvOLm0axtfwET6/a76EI/c8Fk4aIBAFPAdcBI4B5
IjKi02lbALsxZgzwBvAEgDFmpTFmnDFmHHAN0AAsd17zGjAMGA1EAPOdx+8HdhpjxgJXAb8TkQt2
VMz20w7wjuwZ8eyuOsXJhharQ7mg4/XNvL3lEF8eP4C4SN/rZ7o0K4ExabE8t+Yg7bpfudfaVnGC
yrrG8zZNdTRrbCo3jE3ljyv26oZe3eRKTWMisM8Yc8AY0wwsAm7seIIzOTQ4v9wAdPUOPht4/+x5
xpilxgnY2OEaA0SLY9xsFFALnHeJ19S4CFJiwl14Kb7NnunYlGlzmffXNhYXlNPU2u7VHeDnIyLc
MyWbg0frWVFcZXU46hzyCisJCRKuGeZ6n9kvbhxJYlQoDy3eqs2P3eBK0hgAlHf4usJ57FzuBt7v
4vhcYGHng85mqa8Bec5DfwGGA4eBHcB3jDHnbYBM9NMRU52NGxhHsE28fpJfq3M120nZCQzrF2N1
ON123ah+DIiL4Lk1OvzWGxljyCuqZPKgpIvamyUuMpQnZo9lb/Vpfrtstxsj9E+uJI2uOgq6rK+L
yO2AHXiy0/H+OJqhlnVx2V+B1caYNc6vZwBbgVRgHPAXEfncO4+I3CsiBSJSUFMTGIuSRYQGMXJA
rNdvyrSiuJpDJ85wp4/WMs4KDrJx9xVZbCypZYsP1O4Cza7KU5Qea3C5aaqjK4ckc/ukdJ5fe5D1
+4+5ITr/5UrSqAA69mSm4agFfIaITAN+DMwyxnQe4D4HeNsY09LpmseBZOC7HQ7fBbzlbLnaBxzE
0ffxGcaYZ4wxdmOMPTnZt/eZuBj2jHi2lZ+gudV7R38sWHfQp4bZns+c3IHEhAdrbcML5RVWIgJf
GNG937MffXE4mYl9ePj1bZxq9P5+Qm/hStLIBwaLSJazQ3ou8F7HE0RkPPB3HAmjuosy5tGpaUpE
5uOoVczr1PxUBlzrPKcvMBQ44NrL8X+5mfE0tbZTeNg7O/F2Vdax4UAtt0/KIDjI90d0R4UF85VL
M3i/8AhlxxoufIHymGVFleRmJpAUFdat6yNDg/ndnLEcOXmGn/1jZy9H578u+FdtjGkFHsDRtFQM
LDHGFInIz0VklvO0J3F0Wr/uHFr7aVIRkUwcNZVVnYp+GugLrHde81Pn8V8Ak0VkB/Ah8ANjzNHu
vkB/k5Ph6Awv8NJ+jZfWlRIWbGOuDw6zPZc7JzuWc39hrdY2vMXBo/XsqjzFzJEX3zTV0YT0eL51
1SW8samCZUWVvRSdf3NpL0RjzFJgaadjP+3weNp5ri2hi45zY0yX9zbGHAamuxJXIEqODiMjMZKC
kuPcO9XqaD7rREMzb2+p4KZxA4j3o8EJ/WLDmTV2AIvzy3lw2mCfHELsb86+wc/oRn9GZ/957WBW
7q7mR2/tICcjvts1l0Dh++0HAciekcCm0uNet/jakoJyGlt8d5jt+dwzNYszLW289kmZ1aEoHP0Z
Y9JiGRAX0eOyQoNt/OG2cZxqauWHb+7wur8rb6NJwwfZM+M5Vt/sVWvotLUbXl5fysSsBEak+u4w
23MZ1i+GqUOSWbCuRPfdsNiRk2fYWn6iW6OmzmVI32gemTGUFcVVvL5JVwE4H00aPig38+zihd4z
DPTD4ioqjvv+MNvzuXdKNjWnmnh3q27oY6VlhY6mqZ72Z3T2jcuzuDQrgZ//YyfltTro4Vw0afig
7KQo4iJDvKozfMG6EvrHhjO9m8MffcHllyQyrF80z64+oE0YFsorqmRI3yiyk6N6tVybTfjdnLEA
fO/1bbTp8jFd0qThg2w2wZ4R7zU1jT1Vp1i3/5jfDLM9FxHh3qnZ7K0+zUd7AmNCqbc5drqJjQdr
e72WcVZafCSP3zCCjQdreUE34uqS//6F+7mcjAQO1NR7xUZBL60rITTYxryJ6VaH4nZfGpNKv5hw
nl2tU4essKK4inbTO6OmzmV2ThpfGNGXJ5ftZnflKbfdx1dp0vBRZ/s1rN43/GRDC29tPsSNY1P9
dtfEjkKDbdx1eSbr9h/TVVItkFdYycCECEb0d99gCxHhVzePJjo8mIcWb/Xq1ResoEnDR40aEEto
kM3yJqrXN5VzpqXNL4fZnsu8S9OJCgvmuTVa2/CkusYW1u47xsyR/XAsgu0+SVFh/Orm0ew8Usef
Ptzj1nv5Gk0aPio8JIgxabGWdoa3tRteWl9CbmY8owbEWhaHp8WEhzA3dyD/2H6Ewyd8a892X7Zy
VzXNbe29OtT2fKaP7MetOWn87aP9ltfovYkmDR+WkxnPjkMnLdsTYOWuasprz3Dn5CxL7m+lu65w
vOYXdWkRj1lWVElKdBjjB8Z77J4/vWEE/WMj+N6SrTQ0n3dbn4ChScOH2TMSaGkzbK+wpm19wboS
+sWEM32k/w6zPZcBcRFcP7o/CzeWU6crpLpdY0sbK3fVMGNkP49u6xwdHsLv5oyltLaB//pXscfu
6800afiwnIyzk/w830S1r/oUH+87yu2T0gnx42G253PPlGxON7WyaKMuLeJuq/bUcKalzWNNUx1N
yk5k/hVZvPZJGSt3d7WId2AJzL92P5HQJ5RByX0oKPF8e+tL60oDZpjtuYxOi+Wy7EReXFtCS5uO
sHGnZYWVxEWGMDErwZL7f2/6UIb0jeIHb2zneH2zJTF4C00aPi4307F4YbsHZ6/WNbbw5uYKbhiT
SmKArwh679Rsjpxs5F/bj1gdit9qbm1nRXEV04b3taxWGx4SxO/njON4QzOPvVtoSQzeQpOGj8vJ
iOfkmRb21Zz22D1fL6igobnNr9eZctWVQ5IZnBLFM7q0iNtsOHCMusZWt80Cd9WoAbE8OG0I/9x+
hHe3HrI0Fitp0vBxuZlnN2XyTBNVe7vh5fUl5GTEMzotcIbZnovNJtwzJZudR+pYp3tNu0VeUSWR
oUFcMTjB0oDQAAAc90lEQVTJ6lD45tRsxqfH8dg7hRw5GZjDrTVp+LiMxEiSokI9Nl/joz3VlB5r
0FpGBzeOTyUpKoxndGmRXtfWblheVMXVw1IIDwmyOhyCg2z8fs44WtoMj7yxPSBrl5o0fJyIYM9I
8NjM8BfXltA3JsySUSzeKiw4iDsnZ7BqT42uVdTLNpcd5+jpJsubpjrKSurDj64fzpq9R3llQ6nV
4XicJg0/YM+Mp6y2geq6RrfeZ3/NadbsPcrtl2YE7DDbc/nqpRlEhATp0iK9LK+wktBgG1cPS7E6
lM+4/dJ0pg5J5r+XFnPAg/2J3kD/8v2A/Wy/hptrGy+vKyE0yMa8SwN3mO25xPcJZY49jXe2HnJ7
8g4UxhjyCiuZOjiJqLBgq8P5DBHhydljCAsO4qEl22gNoCHXmjT8wMjUGMJDbOS7sV/jVGMLb2yq
4Etj+pMU4MNsz+UbV2TR1m5YsK7E6lD8QuGhOg6dOMMML2qa6qhvTDi/vGkU28pP8NeP9lsdjsdo
0vADIUE2xqbFuXVRtTc2VVDfHFir2V6sjMQ+zBzVj1c3lFLfpOsU9VRe0RGCbMK04d67TM0NY1OZ
NTaVP3+4lx0WLefjaZo0/ERuZgJFh+vcsqhae7vhpXUljE+PY+zAuF4v35/Mn5JNXWMrSwrKrQ7F
5+UVVjIpO4F4L9+n5ec3jiQxKpSHlmy1bPFQT9Kk4SdyMuNpazdsLTvR62Wv2ltDiQ6zdcmE9Hjs
GfE8//HBgGrn7m37qk+xv6beq0ZNnUtcZChPzh7LvurTPLlst9XhuJ0mDT8xIT0eEfd0hi9YW0Jy
dBjXjerf62X7o3umZlNx/Ax5RZVWh+Kz8god37vpPpA0AKYOSeZrkzJ4/uODrNt/1Opw3EqThp+I
jQhhaN/oXu8MP1BzmlV7arj90gxCg/XXxRXThvclK6kPz+rSIt2WV1TJhPQ4+saEWx2Kyx794jCy
kvrw8JJtfr1cvr4L+BF7Zjxbyk7Q1ouLF768vpSQIGHepQN7rUx/F2QT7r4ii20VJ8m3YAViX1de
20DhoTqfm0AaGRrM7+eMpbKukZ+9t9PqcNxGk4YfsWckcLqplV2Vdb1S3ummVucw21RSon3nE583
uGVCGgl9QnVpkW5Y5mzWmznS95pDx6fHc//Vl/Dm5opPm9j8jSYNP2LPdGzK1FtDb9/cVMHpplYd
ZtsNEaFB3D4pgxXFVewPsBnDPbWsqJIR/WNIT4y0OpRu+fY1gxk1IIYfvb2DmlNNVofT6zRp+JEB
cRH0iwnvlSaRs8Nsxw6MY5wOs+2Wr1/m6Ad6bo3uI+6q6lONFJQe97mmqY5Cg238Yc44Tje18uhb
/reooSYNPyIi2DPj2dQLneFr9h3lwNF67tJaRrclRYVxy4Q03txcwdHT/veJ0x2WF1VhDD6dNAAG
943mkRlDWVFc7XdzdjRp+Bl7RjyHTzZy6ETP1vpfsPYgSVFhfHG077Ure5P5U7Jobm3nlfWBtxpq
dywrqiQ7qQ+DU6KsDqXHvnF5FpOyE/j5P3ZSXttgdTi9RpOGn/l08cIe1DYOHq1n5e4avnppug6z
7aFByVFMG96XVzaUcqbZ/2cL98SJhmbW7z/GjFH9EBGrw+kxm0347a1jsYnwvSXbenVUo5X0HcHP
DOsXTZ/QoB51hr+8voRgm/BVXc22V9w7NZva+mbe3FxhdShe7cPialrbjU/MAndVWnwkj88aycaS
Wv720T6rw+kVmjT8THCQjfHp8d3uDK9vauWNggquH9OfFB+aWOXNcjPjGZsWy/MfH/SbT5vukFdU
Sf/YcMb42TbCt0wYwA1jU/ndB3tYsbPK6nB6TJOGH7JnxrO7sq5bs1Lf2lzBKR1m26tEhHumZnPw
aD0rin3/TcMd6ptaWb2nhhkj/aNpqiMR4YlbxjAqNZbvLNri87s7atLwQ/aMBNoNbLnIxQuNcewF
MTYtlvE6zLZXzRzZj7T4CJ7VyX5dWrWnhqbWdq7z8VFT5xIRGsSzX7fTJyyYu1/Kp7a+2eqQuk2T
hh8alx5HkE0ueujtx/uOsr+mnjsmZ/rdpz2rBQfZuPuKLApKj7O5TJcW6SyvsJLEPqGfDuTwR/1i
w3nm63aqTzVx36ubaG71zVWQXUoaIjJTRHaLyD4R+WEXz39XRHaKyHYR+VBEMpzHrxaRrR3+NYrI
Tc7nXnOWWSgiL4hISIfyrnKeXyQiq3rrxQaKqLBghvePvuh+jQVrS0iKCuX6MTrM1h3m2AcSEx6s
+4h30tTaxr93VTN9ZF+CbP79YWXcwDieuGUMnxys5fH3Cn1y4t8Fk4aIBAFPAdcBI4B5IjKi02lb
ALsxZgzwBvAEgDFmpTFmnDFmHHAN0AAsd17zGjAMGA1EAPOd94sD/grMMsaMBG7t0SsMUPaMBLaW
n6DFxT0dSo/V8+/d1XxlYjphwUFuji4w9QkL5vZJGeQVVlJ6rN7qcLzG2n1HOd3U6rXbuva2m8YP
4L6rBrFwYzkv+eDWwK7UNCYC+4wxB4wxzcAi4MaOJziTw9nZKxuAtC7KmQ28f/Y8Y8xS4wRs7HDN
V4C3jDFlzvOqL/ZFKUdn+JmWNnYedm3xwpfXlxIkwlcnZbg5ssB25+RMgmzCCx/r0iJn5RVWEh0W
zORBSVaH4jHfnz6UacNT+MW/ilmzt8bqcC6KK0ljANBxHnyF89i53A2838XxucDCzgedzVJfA/Kc
h4YA8SLykYhsEpGvuxCj6sSe4Zzk58J8jfomx/ak143u71P7F/iilJhwbhw3gCUFFRz34c7Q3tLa
1s4HO6u4dnhKQE0ktdmEP84dzyXJUdz/2mYO+NCilq78lLpqZOyyIU5EbgfswJOdjvfH0Qy1rIvL
/gqsNsascX4dDOQA1wMzgMdEZEgX97pXRApEpKCmxrcytSf0iw0nLT7CpZnhb285xKnGVu6crLUM
T7hnSjZnWtp47RNdWmRjSS3HG1p8fq2p7ogKC+a5O+wEB9mY/3IBJ8/4xsZNriSNCqDjDjxpwOHO
J4nINODHOPoiOq/ONgd42xjT0umax4Fk4Lud7pdnjKk3xhwFVgNjO9/PGPOMMcZujLEnJye78DIC
jz0jnoLS4+ftbDPGsZrt6AGxTEiP92B0gWtov2iuHJLMgnWlNLUG9tIiyworCQ+xMXVIYP4ND0yI
5G9fnUDZsQa+vXCLT+wr70rSyAcGi0iWiITiaGZ6r+MJIjIe+DuOhNFVH8Q8OjVNich8HDWJecaY
jt+pd4EpIhIsIpHApUCxqy9I/R97ZgI1p5oorz334oXr9h9jb/VpHWbrYfdOzebo6Sbe3fK5z18B
o73dsKyoiquGpBAZGmx1OJa5NDuRX9w0itV7avjV+7usDueCLpg0jDGtwAM4mpaKgSXGmCIR+bmI
zHKe9iQQBbzuHCr7aVIRkUwcNZXOQ2efBvoC653X/NR5v2Ic/RvbcXSQP2eMKez+SwxcZzdlOt++
4S+uLSGxTyhf0mG2HjV5UCIj+sfwzJoDtAfo0iLbKk5QWdcYkE1Tnc2bmM6dkzN5/uODLM4vszqc
83IpvRtjlgJLOx37aYfH085zbQlddJwbY855b2PMk3TqF1EXb0hKNNHhwRSUHueWnM8PaCuvbeDD
XVXcf9UlhIfoMFtPEhHunZrNg4u3smpPDVcPS7E6JI/LK6wkJEgC8rV35SfXD2d/zWl+8k4h2clR
5HrpRMfAGa4QgGw2IScj/pyd4S+vL8Emwu06zNYS14/pT//Y8IDcR9wYQ15RJZMHJREbEXLhCwJA
cJCNv8ybQFp8JP/xyiav3YNDk4afy81MYG/1aU40fHZ4Z0NzK4vzy5k5qh/9YnWYrRVCgmzcdXkm
6w8co/DQSavD8ahdlacoPdagTVOdxEaG8Nwddprb2rnn5QLqm1qtDulzNGn4uZwMR79G5/013tly
mLrGVu7U1WwtNXdiOlFhwTwbYEuL5BVWIgJfGNHX6lC8zqDkKJ76ygT2VJ3iocVbva7PS5OGnxub
FkdIkHxmkp9jNduDjEyNwZ6hw2ytFBMewryJA/nn9iM93qLXlywrqiQ3M4GkqDCrQ/FKU4ck85Pr
R7B8ZxW//2CP1eF8hiYNPxcRGsTI1NjP9GusP3CMPVU6zNZb3HV5FgK8GCBLixw8Ws+uylN+tUOf
O9x1eSa32Qfyl5X7eHfrIavD+ZQmjQBgz4hnW8XJTyeSLVhbQkKfUGaNTbU4MgWQGhfBl8b0Z+HG
Mp+ZFdwTy4oqAZih/RnnJSL84qZRTMxM4JE3trOt/OL2x3EXTRoBwJ6ZQHNrO4WHTlJe28CK4irm
5g7UYbZeZP6UbOqb21i00bvH6PeGvMJKxqbFMiAuwupQvF5osI2/3T6BpKgw7nm5gKq6RqtD0qQR
CM52hheUHOfVDaWIDrP1OqMGxDJ5UCIvri3x2c15XHHk5Bm2lp/QWsZFSIwK47k77JxuauXelwto
bLF26RlNGgEgOTqMrKQ+fLzvKIvyy5kxsi+p+inP69wzNZvKukb+ud1/lxZZVuhomtL+jIszvH8M
f7xtHNsPneSRN7ZbunmTJo0AkZMRz5q9Rzl5poU7J2dZHY7qwlVDkhmcEsWzaw765I5ursgrqmRI
3yiyk6OsDsXnTB/Zj4enD+W9bYf560f7LYtDk0aAyHWuQzW8f8ynj5V3ERHumZpN8ZE61u47ZnU4
ve7Y6SY2HqzVWkYPfOuqQcwam8qTy3az3DmgwNM0aQSIy7KTCLYJ90zJ0mG2XuzGcakkR4fxjB9O
9ltRXEW70VFTPSEiPDF7DGPTYnlw8VaKj7i2M2dv0qQRINITI/nkR9dy84SuduJV3iIsOIg7J2ey
ek+NJW8I7pRXWMnAhAhG9I+xOhSfFh4SxDNftxMdHsz8lwo4drrz9kXupUkjgCTq7Fuf8NVL04kI
CeK5Nf4z2a+usYW1+44xc2Q/ren2gr4x4TzzNTtHTzdx36ubPTriTpOGUl4mLjKU23IH8t62Q14x
Lr83rNxVTXNbOzNH6b4tvWXswDiemD2GjSW1PPZOoccGT2jSUMoLfePyLNraDQvWlVgdSq9YVlRJ
SnQY4wfGWR2KX7lx3AAeuPoSFheU8+LaEo/cU5OGUl4oPTGS60b157UNpZz2wuWxL0ZjSxsrd9Uw
Y2Q/bDZtmupt3/3CEKaP6Msv/7WTVXtq3H4/TRpKean5U7Koa2xlSX651aH0yKo9NZxpadO9M9zE
ZhP+cNs4hvSN5oH/3cz+mtPuvZ9bS1dKddv49HgmZibw/McHaW3z3aVFlhVWEhcZwsQs79y+1B/0
CQvmuTvshAbZmP9SAScb3LfwpSYNpbzY/ClZHDpxhvcLrZnI1VPNre2sKK5i2vC+hATp2407pcVH
8vTXcqg43sADCze77YOG/hSV8mLThvclK6kPz6454JNLi2w4cIy6xladBe4huZkJ/PKmUazZe5Rf
/qvYLffQpKGUF7PZhPlTsthecZKNB2svfIGXySuqJDI0iCsGJ1kdSsC4LTedb1yexYJ1JSx0w1L7
mjSU8nK3TEgjoU8of1m5z6f6NtraDcuLqrh6WIru3eJhP/riMKYOSeaxdwrZcKB31zHTpKGUlwsP
CeJbVw1izd6jzH56PQfcPDqmt2wuO87R001cp6OmPC44yMb/zBtPemIk9726ifLahl4rW5OGUj5g
/pRs/vKV8Rw8Ws8X/7yGV9aXeH0fR15hJaHBNq4ammJ1KAEpNiKE575up63dMP+lgl6b76NJQykf
8aUxqSx/aCqXZiXy2LtFfP2FjVSe9M5lRowx5BVWMnVwElFhwVaHE7Cyk6N46qsT2FdzmgcXbaW9
vecfNDRpKOVD+saEs+CuXH550ygKSo4z44+reW+b9+30V3iojkMnzjBDR01ZbsrgZB67fjgriqv4
7fLdPS5Pk4ZSPubsHu9LvzOF7OQ+/OfCLXx74RZONDRbHdqn8oqOEGQTpg3va3UoCrhjcibzJqbz
14/2886WQz0qS5OGUj4qK6kPr3/zMh6ePoT3dxxhxh9Xe2TtIVfkFVYyKTuB+D6hVoeicHzQ+Nms
kUzMSuCRN7ezpex4t8vSpKGUDwsOsvHANYN55/7LiQkP4Y4XNvLTdwtpaLZukcN91afYX1OvE/q8
TGiwjadvzyElOox7X9nEkZNnulWOJg2l/MCoAbH849tXcPcVWby8vpTr//xxjz5N9kSec8mT6Zo0
vE5Cn1CevyOXhqZW7n15U7fK0KShlJ8IDwnisS+N4H/vuZTm1nZmP72e3y/fTYuHJwTmFVWSkxFP
35hwj95XuWZov2j+NHc8hYdPdut6TRpK+ZnJg5J4/8Ep3DRuAH/+9z5u/us69lWf8si9y2sbKDxU
p01TXm7aiL78cOawbl2rSUMpPxQTHsLv5ozl6dsncOjEGa7/88e88PHBXhmnfz7LihxNUzrU1vt9
88pB3bpOk4ZSfmzmqP7kPTiFKy5J4uf/3Mntz3/CoRPd6wB1xbKiSkb0jyE9MdJt91DW0qShlJ9L
iQ7nuTvs/Prm0WwrP8HMP67mrc0Vvb4MSfWpRgpKj+sOfX5Ok4ZSAUBEmDsxnfe/M5WhfaP57pJt
fOu1zdTW996EwOVFVRiDJg0/51LSEJGZIrJbRPaJyA+7eP67IrJTRLaLyIcikuE8frWIbO3wr1FE
bnI+95qzzEIReUFEQjqVmSsibSIyuzdeqFIK0hMjWfzNy/jBzGGsKK5i+h9W8+9dVb1S9rKiSrKT
+jA4JapXylPe6YJJQ0SCgKeA64ARwDwRGdHptC2A3RgzBngDeALAGLPSGDPOGDMOuAZoAJY7r3kN
GAaMBiKA+Z3u+RtgWfdfmlKqK0E24b6rBvHu/VeQFBXKNxYU8OhbO6jvwSqoJxqaWb//GDNG9UNE
ejFa5W1cqWlMBPYZYw4YY5qBRcCNHU9wJoezC7ZvANK6KGc28P7Z84wxS40TsLHTNd8G3gSqL+rV
KKVcNiI1hncfuJxvXpnNovwyrvvTGgpKurc74IfF1bS2Gx1qGwBcSRoDgPIOX1c4j53L3cD7XRyf
CyzsfNDZLPU1IM/59QDgy8DTLsSmlOqBsOAgHr1uOIvvvYx2Y5jz9/X8Jm8Xza0XNyEwr6iS1Nhw
xqTFuilS5S1cSRpd1TW7HHYhIrcDduDJTsf742iG6qq56a/AamPMGufXfwR+YIxpO29QIveKSIGI
FNTUeMcibUr5qolZCeQ9OJVbcwbyt4/2c+NTa9ld6dqEwPqmVlbvqdGmqQDhStKoAAZ2+DoN+NwC
/iIyDfgxMMsY09Tp6TnA28aYlk7XPA4kA9/tcNgOLBKREhxNWn8923nekTHmGWOM3RhjT05OduFl
KKXOJyosmN/MHsOzX7dTc6qRG/7nY55ZvZ+2C0wIXLWnhqbWdm2aChCuJI18YLCIZIlIKI5mpvc6
niAi44G/40gYXfVDzKNT05SIzAdmAPOMMZ/WhY0xWcaYTGNMJo5O9W8ZY965iNeklOqBL4zoy7IH
p3LV0GT+e+ku5j274bx7TOcVVpLYJxR7ZoIHo1RWuWDSMMa0Ag/gaFoqBpYYY4pE5OciMst52pNA
FPC6c2jtp0lFRDJx1FRWdSr6aaAvsN55zU97+mKUUr0jMSqMv38th9/eOpadh+uY+cfVLMkv/9yE
wKbWNv69q5rpI/sSZNOmqUDg0ua9xpilwNJOx37a4fG081xbQhcd58aYC97bGHOnK/EppXqfiDA7
J41J2Qk8/Po2HnlzOx8UV/Grm0eTFBUGwNp9Rznd1KprTQUQnRGulDqvtPhI/nf+JH5y/XBW7alh
xh9Ws9y5MGFeYSXRYcFMHpRkcZTKU1yqaSilApvNJsyfks2Uwck8tHgr976yiVtz0lhRXMW1w1MI
DdbPn4FCf9JKKZcN7RfNO/dfzv1XD+LNzRUcb2hh5qj+VoelPEhrGkqpixIabOP7M4ZxzbC+fLCz
iquH6ZD3QKJJQynVLTkZ8eRkxFsdhvIwbZ5SSinlMk0aSimlXKZJQymllMs0aSillHKZJg2llFIu
06ShlFLKZZo0lFJKuUyThlJKKZdJ56WOfZGInAJ2Wx1HJ0nAUauD6II3xqUxuUZjcp03xuWNMQ01
xkRfzAX+MiN8tzHGbnUQHYlIgbfFBN4Zl8bkGo3Jdd4Yl7fGdLHXaPOUUkopl2nSUEop5TJ/SRrP
WB1AF7wxJvDOuDQm12hMrvPGuPwiJr/oCFdKKeUZ/lLTUEop5QE+lzRE5AURqRaRwg7HbhWRIhFp
FxGPj044R0xPisguEdkuIm+LSJwXxPQLZzxbRWS5iKR6MqZzxdXhuYdFxIiIRzecPsf36v+JyCHn
92qriHzR6picx78tIrudv+9PWB2TiCzu8D0qEZGtXhDTOBHZ4IypQEQmejKm88Q1VkTWi8gOEfmH
iMR4MJ6BIrJSRIqdvzvfcR5PEJEPRGSv8/8Lb5BijPGpf8BUYAJQ2OHYcGAo8BFg95KYpgPBzse/
AX7jBTHFdHj8n8DT3vC9ch4fCCwDSoEkq2MC/h/wsKe/PxeI6WpgBRDm/DrF6pg6Pf874KdWxwQs
B65zPv4i8JGX/PzygSudj78B/MKD8fQHJjgfRwN7gBHAE8APncd/6Mr7lM/VNIwxq4HaTseKjTGW
Te47R0zLjTGtzi83AGleEFNdhy/7AB7v0OoqLqc/AI/gXTFZ5hwx3Qf82hjT5Dyn2gtiAkBEBJgD
LPSCmAxw9lN8LHDYkzHBOeMaCqx2Pv4AuMWD8Rwxxmx2Pj4FFAMDgBuBl5ynvQTcdKGyfC5p+Khv
AO9bHQSAiPyXiJQDXwV+anU8ACIyCzhkjNlmdSydPOBsznvBpWq7+w0BpojIJyKySkRyrQ6ogylA
lTFmr9WBAA8CTzp/z38LPGpxPGcVArOcj2/FUbv2OBHJBMYDnwB9jTFHwJFYgJQLXa9Jw81E5MdA
K/Ca1bEAGGN+bIwZiCOeB6yOR0QigR/jJQmsg78Bg4BxwBEcTS9WCwbigUnA94Elzk/43mAeHq5l
nMd9wEPO3/OHgOctjuesbwD3i8gmHE1EzZ4OQESigDeBBzu1PLhMk4YbicgdwJeArxpno6EX+V88
WD0+j0FAFrBNREpwNONtFpF+VgZljKkyxrQZY9qBZwGPd6Z2oQJ4yzhsBNpxrGdkKREJBm4GFlsd
i9MdwFvOx6/jHT87jDG7jDHTjTE5OBLsfk/eX0RCcCSM14wxZ78/VSLS3/l8f+CCTZ6aNNxERGYC
PwBmGWMarI4HQEQGd/hyFrDLqljOMsbsMMakGGMyjTGZON4YJxhjKq2M6+wfktOXcTQtWO0d4BoA
ERkChOIdC+BNA3YZYyqsDsTpMHCl8/E1gDc0mSEiKc7/bcBPgKc9eG/BUeMqNsb8vsNT7+FIsjj/
f/eChXl6VEEvjAJYiKO5oAXHG8zdOP6oK4AmoApY5gUx7QPKga3Ofx4dqXSOmN7E8ea3HfgHMMAb
fn6dni/B86OnuvpevQLscH6v3gP6e0FMocCrzp/hZuAaq2NyHl8A/Ienf5fO8326AtgEbMPRbp/j
JXF9B8eopT3Ar3FOrvZQPFfgGCCwvcN70heBROBDHIn1QyDhQmXpjHCllFIu0+YppZRSLtOkoZRS
ymWaNJRSSrlMk4ZSSimXadJQSinlMk0aSimlXKZJQymllMs0aSjlBiKS6dy74Fnn/gXLRSTC6riU
6ilNGkq5z2DgKWPMSOAE3rHWl1I9oklDKfc5aIw5u5PdJiDTwliU6hWaNJRyn6YOj9twLG2ulE/T
pKGUUsplmjSUUkq5TFe5VUop5TKtaSillHKZJg2llFIu06ShlFLKZZo0lFJKuUyThlJKKZdp0lBK
KeUyTRpKKaVcpklDKaWUy/4/U2KKUp0pjXIAAAAASUVORK5CYII=
)



{% highlight python %}
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
{% endhighlight %}


{% highlight python %}
?plot_partial_dependence
{% endhighlight %}


{% highlight python %}
names

{% endhighlight %}




    Index([u'amenity_veterinary', u'leisure_motor_track', u'median_income',
           u'ready_to_build', u'bedrooms', u'amenity_grave_yard', u'lot_size',
           u'prop_type', u'leisure_dog_park', u'size_sqft'],
          dtype='object')




{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[3],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);
{% endhighlight %}


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-1050-411788b1b4f5> in <module>()
          3 fig, axes = plot_partial_dependence(clf, X=X, features=[3],
          4                                     feature_names=names, n_cols=2,
    ----> 5                                     n_jobs=3, grid_resolution=20);
    

    /Users/dave/.pyenv/versions/anaconda2-4.1.0/envs/geo/lib/python2.7/site-packages/sklearn/ensemble/partial_dependence.pyc in plot_partial_dependence(gbrt, X, features, feature_names, label, n_cols, grid_resolution, percentiles, n_jobs, verbose, ax, line_kw, contour_kw, **fig_kw)
        263     X = check_array(X, dtype=DTYPE, order='C')
        264     if gbrt.n_features != X.shape[1]:
    --> 265         raise ValueError('X.shape[1] does not match gbrt.n_features')
        266 
        267     if line_kw is None:


    ValueError: X.shape[1] does not match gbrt.n_features



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[0],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);


{% endhighlight %}


![png](data:image/png;base64,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
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[1],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);


{% endhighlight %}


![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAagAAADQCAYAAABStPXYAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzt3XecVPW9//HXh+LSpfeuiEZQyoJYbgTFhqigNEEDEVvi
VcEbfxJzEzXJNRrvFcs14SKrEt2loyhiQcQu6lJ0lyZIEQTpSC8Ln98fc1YX3DK77OyZ3Xk/Hw8e
M3PmlDfjjB9O+37M3REREYk35cIOICIikhsVKBERiUsqUCIiEpdUoEREJC6pQImISFxSgRIRkbik
AiUiInEppgXKzEaa2WIzyzSzCWZWycxamdlnZrbCzCaZ2UmxzCAiIqVTzAqUmTUB7gKS3b0dUB4Y
BDwKjHb3NsAOYHisMoiISOlVoQTWX9nMDgNVgI3ARcDg4P3xwIPAP/NbSd26db1ly5axSylSyuw7
vI+lW5aGHUPKuLMbnk2FckUvE/Pnz9/q7vWKunzMCpS7f2dm/w18C+wH3gbmAzvdPSuYbT3QpKB1
tWzZkvT09FhFFSl17nnrHr754hteH/w6J5XXUXKJjW5Nu53Q98vM1p7I9mNWoMysFnAN0ArYCUwB
rshl1lwHAzSzW4FbAZo3bx6jlCKlz5GjR5iYOZFebXrRs3XPsOOIxEwsL5LoCax29y3ufhiYDpwH
1DSz7MLYFNiQ28LuPtbdk909uV69Iu8hipQ57615j417NjK43eCCZxYpxWJZoL4FuplZFTMz4GJg
CTAX6BfMMxSYEcMMImVOakYq1U+qTu/TeocdRSSmYlag3P0zYCqwAMgItjUWuA+4x8xWAnWAlFhl
EClrDmQdYNrSaVx7xrVUrlg57DgiMRXTq/jc/QHggeMmrwK6xnK7ImXV61+/zq6DuxjSfkjYUURi
TiNJiJQiaZlpNKjagB6teoQdRSTmVKBESomdB3by+tevM6jdoBO6N0WktFCBEiklpi+dzsEjBxnc
XlfvSWJQgRIpJdIy0jil1il0adwl7CgiJUIFSqQU2LB7A++ufpch7YcQuWtDpOxTgRIpBSZlTsJx
Hd6ThKICJVIKpGak0rlRZ9rWbRt2FJESowIlEueWb13O/I3ztfckCUcFSiTOTcicgGEMajco7Cgi
JUoFSiSOuTupGan0aNWDxtUbhx1HpESpQInEsfQN6azcvlIjl0tCUoESiWOpGamcVP4krvvFdWFH
ESlxKlAicSq7MeGVba6kZqWaYccRKXEqUCJxau6auWzau0kjl0vCUoESiVOpGanUSKrBladdGXYU
kVCoQInEof2H9zNtyTSuO+M6KlWoFHYckVCoQInEoddXvM7uQ7t1c64kNBUokTiUlpFGw2oN6dFS
jQklcalAicSZHft38PqK1xl05iDKlysfdhyR0KhAicSZ6Uunc+jIIR3ek4SnAiUSZ9Iy02hTuw3J
jZPDjiISKhUokTjy3a7vmLt6LoPbD1ZjQkl4KlAicWTSYjUmFMmmAiUSR1IzUklunMxpdU4LO4pI
6FSgROLEsq3LWLBxgUYuFwlEVaDMrIWZ9QyeVzaz6rGNJZJ40jLS1JhQJIcCC5SZ3QJMBf4vmNQU
eCWalZtZTTObambLzGypmZ1rZrXNbLaZrQgeaxU9vkjZ4O6kZaRxUauLaFS9UdhxROJCNHtQdwDn
A7sA3H0FUD/K9T8JvOnupwNnA0uBUcAcd28DzAleiyS0LzZ8wTc7vtHI5SI5RFOgDrr7oewXZlYB
8IIWMrMawC+BFAB3P+TuO4FrgPHBbOOBPoUNLVLWpH6VSlL5JK4949qwo4jEjWgK1Ptmdj9Q2cwu
AaYAr0WxXGtgC/C8mS00s3FmVhVo4O4bAYLHaPfGRMqkrKNZTFo8id6n9ebkSieHHUckbkRToEYR
KTQZwG3ALOA/o1iuAtAJ+Ke7dwT2UojDeWZ2q5mlm1n6li1bol1MpNR5d/W7bNq7Sfc+iRwnmgJV
GXjO3fu7ez/guWBaQdYD6939s+D1VCIFa5OZNQIIHjfntrC7j3X3ZHdPrlevXhSbEymd0jLSODnp
ZHq16RV2FJG4Ek2BmsOxBaky8E5BC7n798A6M2sbTLoYWAK8CgwNpg0FZkSdVqSM2X94P9OXTldj
QpFcVIhinkruvif7hbvvMbMqUa7/TiDVzE4CVgG/JlIUJ5vZcOBboH8hM4uUGTO/nqnGhCJ5iKZA
7TWzTu6+AMDMOgP7o1m5uy8CchuS+eLoI4qUXWmZaTSq1ojuLbuHHUUk7kRToEYAU8xsQ/C6ETAw
dpFEEsOO/TuYtWIWd3S5Q40JRXJRYIFy9y/M7HSgLWDAMnc/HPNkImXctKXT1JhQJB/R7EEBdAFa
BvN3NDPc/V8xSyWSAFIzUjmtzml0btQ57CgicanAAmVmLwKnAIuAI8FkB1SgRIpo/a71vL/mfR64
8AE1JhTJQzR7UMnAL9y9wOGNRCQ6kzLVmFCkINHcB5UJNIx1EJFEkpqRSpfGXWhTp03YUUTiVjR7
UHWBJWb2OXAwe6K7Xx2zVCJl2NItS1n4/UKeuOyJsKOIxLVoCtSDsQ4hkkjSMtIoZ+UYcOaAsKOI
xLVoLjN/38xaAG3c/Z1gFAndtCFSBO5OWqYaE4pEoygddZsQZUddETnWZ999xqodq9SYUCQKse6o
KyI5pGWkkVQ+ib6n9w07ikjci1lHXRE5VnZjwqvaXqXGhCJRiGVHXRHJYc6qOWzeu5nB7XTvk0g0
YtlRV0RySMtUY0KRwojmKr6jwLPBHxEpgn2H9zF96XQGnjmQpApJYccRKRXyLFBmlkE+55rc/ayY
JBIpg2Z+PZM9h/ZoaCORQshvD6p38HhH8Phi8DgE2BezRCJlUFpGGo2rN+bCFheGHUWk1MizQLn7
WgAzO9/dz8/x1igz+xj4c6zDiZQF2/dvZ9aKWdzZ9U41JhQphGgukqhqZhdkvzCz84CqsYskUrZM
WzKNw0cPM+Qs3ZwrUhjRjMU3HHjOzLJv3NgJ3BS7SCJlS2pGKm3rtKVjw45hRxEpVaK5im8+cLaZ
1QDM3X+IfSyRsmHdD+v4YO0HPNT9ITUmFCmkaDrqJgHXEbR8z/6RubvOQYkUYGLmRBzn+vbXhx1F
pNSJ5hDfDOAHYD45+kGJSMHSMtPo2qQrp9Y+NewoIqVONAWqqbtfHvMkImXMki1LWPT9Ip68/Mmw
o4iUStFcxfeJmbWPeRKRMkaNCUVOTDR7UBcAw8xsNZFDfAa4RpIQyZu7k5aRRs/WPWlYrWHYcURK
pWgK1BUnsgEzKw+kA9+5e28zawVMBGoDC4Abc7bzECkL5q2fx+qdq3ngwgfCjiJSahV4iC8YUaIZ
cFHwfF80y+VwN7A0x+tHgdHu3gbYQeQ+K5EyJS0jjUoVKtH3DDUmFCmqaFq+PwDcB/w+mFQReCma
lZtZU+BKYFzw2oCLiLSQBxgP9ClcZJH4dvjI4UhjwtOuokZSjbDjiJRa0ewJ9QWuBvYCuPsGoHqU
638C+H/A0eB1HWCnu2cFr9cDTaJOK1IKzFk9hy37tmjkcpETFE2BOuTuTtB6w8yiGofPzHoDm4OR
KH6cnMusubb0MLNbzSzdzNK3bNkSzSYlQbk7D733EPfPuZ/1u9aHHYe0jDRqVqrJFaee0OlbkYQX
TYGabGb/B9Q0s1uAd4iueeH5wNVmtobIRREXEdmjqmlm2RdnNAU25Lawu49192R3T65Xr14Um5NE
lbIwhQfff5C/ffQ3Wj3ZisHTBvPFd1+EkmXf4X28vOxl+p3RT40JRU5QNBdJ/DeRc0bTgNOAP7n7
01Es93t3b+ruLYFBwLvuPgSYC/QLZhtKZKQKkSL5atNX3PnGnVzS+hJW3rmSO7veycyvZ9J1XFcu
eO4Cpi2ZxpGjR0osz2vLX2PPoT0auVykGER7NV4G8CHwQfD8RNwH3GNmK4mck0o5wfVJgtp9cDf9
p/SnVqVavHTtS5xS+xQev+xx1t+zntGXjWbD7g30m9KPU58+lcc/fZwfDsR+nOPUjFSaVG/CL1v8
MubbEinrormK72bgc+BaIns+88ysUO023P09d+8dPF/l7l3d/VR37+/uGt9PCs3duW3mbazcvpIJ
102gftX6P75XI6kGI7qNYMWdK5g+YDrNajTjP97+D5qNbsaIN0ewaseqmGTatm8bb6x8g+vbXU85
K8ydGCKSm2h+RfcCHd19mLsPBToT2QsSCc3Y+WOZkDmBv/T4Cxe2zL2Nevly5el7Rl8++PUHpN+S
ztVtr+aZL56hzdNtuHbStXy49kMi1/8Uj6lLppJ1NEtX74kUk2gK1Hpgd47Xu4F1sYkjUrCFGxdy
95t3c9kplzHqglFRLdO5cWdeuvYl1ty9hlHnj+L9te/zyxd+SZdnu5D6VSqHjpz4YCZpmWmcXvd0
OjTscMLrEpHoCtR3wGdm9mBw0+48YKWZ3WNm98Q2nsixdh3cxYCpA6hbpS4v9n2x0IfSmtRown9d
/F+sG7mOMVeOYe/hvdzw8g20erIVf/vwb2zbt61IubIbEw5pP0SNCUWKSTS/7m+AV/jpfqUZwEYi
N+tGe8OuyAlzd2557RZW71jNxH4TqVe16LcfVKlYhduSb2Pxbxcza/Aszqx3Jve/ez/NRjfj9pm3
s2zrskKtb0LmBACub6fGhCLFxaI9Bm9mVd19b4zz5Co5OdnT09PD2LTEkX988Q/umHUHj1z8CPdd
UPynQTM3Z/LkvCd58asXOXjkIFecegUju42kZ+ueBe4VdRjTgcoVK/Pp8E+LPZdIaWVm8909uajL
R3MV37lmtoRgwFczO9vM/lHUDYoUxYKNCxj51kh6tenFveffG5NttKvfjmevfpZ1I9fx5+5/ZsHG
BVz60qWcNeYsUhakcCDrQK7LLd68mC83fcngdro4QqQ4RXOI7wngMmAbgLt/CegmDykxPxz4gf5T
+lO/an3G9xkf80u461Wtxx8v/CNrR6zlhWteoLyV5+bXbqb56Ob8ae6f+H7P98fMn5aRRnkrr8aE
IsUsql+6ux9/1V7J3ZovCc3dGf7qcNbuXMukfpOoW6VuiW07qUISQzsMZeFtC5k7dC7nNjuXv37w
V1o80YJhrwzjy++/jDQmzIw0JmxQrUGJZRNJBNE0LFxnZucBbmYnAXdxbH8nkZh55otnmLZ0Gn/v
+XfOa3ZeKBnMjO4tu9O9ZXdWbFvBU589xfOLnmf8l+Pp3Kgza3au4aHuD4WSTaQsK/AiCTOrCzwJ
9CQyGvnbwN3uXrTrcYtAF0kkpvQN6ZyXch6XnXoZMwbNiKvRGXbs38G4BeN4+vOn2XNoD2tHrKV6
ki5qFcnpRC+SiPoqvjCpQCWenQd20un/OnHEj7DwtoXUrlw77Ei5yjqaxd5Dezm50slhRxGJOyda
oPI8xGdmT5NHryYAd7+rqBsVyY+7c9OMm1i3ax0f/vrDuC1OABXKVVBxEomR/I6ZpAPzgUpAJ2BF
8KcDukhCYuipz57i5WUv82jPR+nWtFvYcUQkJHnuQbn7eAAzGwb0cPfDwesxRM5DiRS7z7/7nHtn
38vVba9mZLeRYccRkRBFc9a5MccOaVQtmCZSrHbs38GAKQNoXL0xL1zzgsa0E0lw0Vxm/giw0Mzm
Bq8vBB6MWSJJSO7Or2f8mg27N/DRTR9Rq3KtsCOJSMgKLFDu/ryZvQGcE0wa5e7f57eMSGGNnjea
Gctn8MRlT9C1Sdew44hIHIhmD4qgIM2IcRZJUPPWz+O+d+6j7+l9uescXRwqIhHxc+ejJKTt+7cz
cOpAmtVoxnPXPKfzTiLyo6j2oERi4agfZegrQ9m4eyOfDP+EmpVqhh1JROJIfjfq5nt3pLtvL/44
kkge//RxZn49k6cuf4rkxkW+2VxEyqj89qDmExlJIrdjLg60jkkiSQifrPuEUe+Mot8v+vHvXf89
7DgiEofyu1G3VUkGkcSxdd9WBk4dSIuaLRh31TiddxKRXEV1DsrMagFtiAx7BIC7fxCrUFJ2ZZ93
2rx3M58O/1Tj2IlIngosUGZ2M3A30BRYBHQDPgUuim00KYse+/gxZq2YxTO9nqFTo05hxxGROBbN
ZeZ3A12Ate7eA+gIbIlpKimTPvr2I/7w7h8YcOYAfpP8m7DjiEici6ZAHXD3AwBmluTuy4C2sY0l
Zc2WvVsYNHUQrWq14tmrntV5JxEpUDQFar2Z1QReAWab2QxgQ0ELmVkzM5trZkvNbLGZ3R1Mr21m
s81sRfCoQdfKuKN+lBtfvpGt+7Yypf8UaiTVCDuSiJQC0YzF1zd4+mAwYOzJwJtRrDsL+A93X2Bm
1YH5ZjYbGAbMcfdHzGwUMAq4r0jppVR49KNHeeubtxhz5Rg6NOwQdhwRKSXyu1G3hrvvOu6G3Yzg
sRqQ74267r4R2Bg8321mS4EmwDVA92C28cB7qECVWR+s/YD/nPufDGo3iFs73xp2HBEpRfLbg0oD
enPsDbs5H6O+UdfMWhK5uOIzoEFQvHD3jWZWP49lbgVuBWjevHm0m5I4snnvZgZNHcSptU9lbO+x
Ou8kIoWS3426vYPHE7ph18yqAdOAEcEeWVTLuftYYCxAcnKyn0iGRLN9/3beW/MeTao3oXWt1tSt
UrfEi8NRP8oN029gx4EdvHnDm1RPql7wQiIiOURzH9Qcd7+4oGl5LFuRSHFKdffpweRNZtYo2Htq
BGwuSnDJ3d5De+kxvgdfbfrqx2lVK1alda3WtKrVitY1g8darWlVsxWtarWiSsUqxZ7j4Q8fZvaq
2YztPZazGpxV7OsXkbIvv3NQlYAqQN3gSrvsf4LXIIqW7xb5J3sKsNTdH8/x1qvAUCKdeoeiPlPF
xt256dWbyNycyb/6/ItalWuxascqVu9Yzaqdkcc5q+aw9/DeY5ZrWK0hrWr+VLR+LGa1WtOkehPK
lytfqBxzV8/lgfceYEj7Idzc6ebi/CuKSALJbw/qNmAEkWI0n58K1C7gmSjWfT5wI5BhZouCafcT
KUyTzWw48C3Qvwi5JRePffIYkxdP5tGej3Lj2TfmOo+7s2XflkjR2rGK1Tt/evx43cdMzJzIET/y
4/wVy1WkRc0WeRawWpVqHXP4cNOeTQyePpjT6pzGmN5jdN5JRIrM3PM+vWNm5YH73f0vJRfp55KT
kz09PT3MCHHvzZVv0iu1FwPOHMCE6yYUuTAcPnKYdbvW/VjAji9iW/dtPWb+Gkk1aF2r9Y/F69P1
n7Jw40I+v+Vz2tVvVxx/NREppcxsvrsXuZdOvgUq2MCn7n5uUTdQHFSg8rdy+0q6PNuF5ic355Ob
PqHqSVVjtq3dB3f/VLBy2QvLOprFuKvGMbTD0JhlEJHS4UQLVDSjmb9tZtcB072gaiYlbvfB3fSZ
2IdyVo5XBr4S0+IEUD2pOmc1OCvXCx+O+lEOZB2IyUUXIpJ4oilQ9wBVgSwzO0BwH5S7a7yakLk7
w2YMY+nWpbx9w9u0qhVuC69yVk7FSUSKTTRDHekGljj18IcPM33pdB6/9HEubl3gVf8iIqWKGhaW
UjO/nskf5/6RIe2HMKLbiLDjiIgUOzUsLIWWb13OkOlD6Nioo1pXiEiZpYaFpcyug7voM6kPSeWT
eHngy1SuWDnsSCIiMRHNIb4D7n7AzH5sWGhmalgYguy+Siu2rWDOr+bQ/GQNoisiZVc0Ber4hoU7
iKJhoRS/P7//Z15d/ipPX/E0F7a8MOw4IiIxFcuGhVKMXln2Cg+9/xDDOgzjji53hB1HRCTmChos
9nbgVCKNClPc/f2SCiY/WbJlCTe+fCNdGnfhn1f+UxdFiEhCyO8iifFAMpHidAXwPyWSSI6x88BO
+kzsQ9WKVZk+cDqVKlQqeCERkTIgv0N8v3D39gBmlgJ8XjKRJNuRo0cYPG0wa3auYe7QuTSt0TTs
SCIiJSa/AnU4+4m7Z+mwUsn709w/8cbKNxhz5RjOb35+2HFEREpUfgXqbDPbFTw3oHLwWmPxlYAp
i6fw8EcPc0unW7gt+baw44iIlLg8C5S7F66NqhSbjE0ZDJsxjHObnsvTVzwddhwRkVBEM5KElKDt
+7fTZ1IfTk46mWkDppFUISnsSCIioYhqsFgpGVlHsxg0dRDrd63n/WHv06h6o7AjiYiERgUqjtw/
535mr5rNuKvG0a1pt7DjiIiESof44sSEjAk89slj/Db5twzvNDzsOCIioVOBigOLvl/E8FeH82/N
/43Rl48OO46ISFxQgQrZ1n1b6TOxD3Wq1GFK/ymcVP6ksCOJiMQFnYMKUdbRLAZMGcD3e77no5s+
okG1BmFHEhGJGypQIbr37XuZu2Yu4/uMJ7lxcthxRETiig7xheTFL1/kic+e4O5z7uZXZ/8q7Dgi
InFHBSoE6RvSueW1W+jRsgePXfJY2HFEROJSKAXKzC43s+VmttLMRoWRISyb9myi76S+NKzWkEn9
JlGxfMWwI4mIxKUSPwdlZuWBZ4BLgPXAF2b2qrsvKeksJe3wkcP0n9Kfbfu28fFNH1Ovar2wI4mI
xK0w9qC6AivdfZW7HwImAteEkKPEjXxrJB9++yEpV6fQsVHHsOOIiMS1MApUE2Bdjtfrg2ll2nML
n+OZL57hd+f+juvbXx92HBGRuBdGgcqt86H/bCazW80s3czSt2zZUgKxYuez9Z/xm9d/wyWtL+Fv
Pf8WdhwRkVIhjAK1HmiW43VTYMPxM7n7WHdPdvfkevVK77majbs3cu3ka2laoykT+02kQjndeiYi
Eo0w/m/5BdDGzFoB3wGDgMEh5Ii5g1kHuW7ydew8sJN5w+dRu3LtsCOJiJQaJb4H5e5ZwL8DbwFL
gcnuvrikc8Tasq3LuOHlG/h0/ae8cM0LtG/QPuxIIiKlSijHm9x9FjArjG3H0t5De5m8eDIpC1P4
eN3HVChXgb/2+Cv9z+wfdjQRkVJHJ0ROkLvz+Xefk7IwhYmZE9l9aDdt67Tl7z3/zq/O/pUGgBUR
KSIVqCLaum8rL331EikLU8jcnEmVilUYeOZAhnccznnNzsMst4sVRUQkWipQhXDk6BHeWfUOKQtT
mLF8BoeOHOKcJucwtvdYBrYbSI2kGmFHFBEpM1SgorB251qeX/Q8zy96nm9/+JY6lev82Jq9Xf12
eS6XvRfl/rPbvGKqQoXIf9asrKwS3a6ISHFSgcrDwayDzFg+g5SFKcz+ZjYAl5xyCY9d8hjXtL2G
pApJIScUESnbVKCOk7Epg5SFKbz01Uts27+N5ic354ELH2BYh2G0qNki7HgiIglDBQrYdXAXEzMn
krIwhc+/+5yK5SrS94y+DO84nItbXUz5cuXDjigiknAStkC5Ox+v+5iUhSlMXjyZfYf30a5+O0Zf
NpobzrqBulXqhh1RRCShJVyB2rRnE+O/HM9zC59j+bblVDupGkPaD2F4x+F0bdJVl4eLiMSJhChQ
WUezeHPlm6QsTGHm1zPJOprF+c3OZ9QFo+j/i/5UPalq2BFFROQ4Zb5AffztxwyYOoANuzdQv2p9
RnYbyU0db+L0uqeHHU1ERPJhJX2PTlGY2RZgbdg5QlAX2Bp2iFJEn1fh6PMqHH1ehVMXqOruRe6X
VCoKVKIys3R3Tw47R2mhz6tw9HkVjj6vwimOzyuMhoUiIiIFUoESEZG4pAIV38aGHaCU0edVOPq8
CkefV+Gc8Oelc1AiIhKXtAclIiJxSQUqZGbWzMzmmtlSM1tsZnfnMk93M/vBzBYFf/4URtZ4YWZr
zCwj+CzSc3nfzOwpM1tpZl+ZWacwcsYDM2ub43uzyMx2mdmI4+ZJ6O+XmT1nZpvNLDPHtNpmNtvM
VgSPtfJYdmgwzwozG1pyqcORx2f1mJktC35rL5tZzTyWzfd3m+syOsQXLjNrBDRy9wVmVh2YD/Rx
9yU55ukO/M7de4cUM66Y2Rog2d1zvSfFzHoBdwK9gHOAJ939nJJLGJ/MrDzwHXCOu6/NMb07Cfz9
MrNfAnuAf7l7u2Da34Ht7v6ImY0Carn7fcctVxtIB5IBJ/Lb7ezuO0r0L1CC8visLgXedfcsM3sU
4PjPKphvDfn8bnOjPaiQuftGd18QPN8NLAWahJuq1LuGyA/I3X0eUDP4h0Ciuxj4JmdxEnD3D4Dt
x02+BhgfPB8P9Mll0cuA2e6+PShKs4HLYxY0DuT2Wbn72+6e3R11HtC0uLanAhVHzKwl0BH4LJe3
zzWzL83sDTM7s0SDxR8H3jaz+WZ2ay7vNwHW5Xi9HhV9gEHAhDze0/frWA3cfSNE/hEJ1M9lHn3P
fu4m4I083ivod/szZX4svtLCzKoB04AR7r7ruLcXAC3cfU9w+OoVoE1JZ4wj57v7BjOrD8w2s2XB
v+yy5TYkfUIfyzazk4Crgd/n8ra+X0Wj71kOZvYHIAtIzWOWgn63P6M9qDhgZhWJFKdUd59+/Pvu
vsvd9wTPZwEVzSxhG1a5+4bgcTPwMtD1uFnWA81yvG4KbCiZdHHrCmCBu286/g19v3K1KfuwcPC4
OZd59D0LBBeI9AaGeB4XNkTxu/0ZFaiQWaQBVQqw1N0fz2OehsF8mFlXIv/dtpVcyvhhZlWDi0kw
s6rApUDmcbO9CvwquJqvG/BD9uGaBHY9eRze0/crV68C2VflDQVm5DLPW8ClZlYruMrv0mBaQjGz
y4H7gKvdfV8e80Tzu/0ZHeIL3/nAjUCGmS0Kpt0PNAdw9zFAP+A3ZpYF7AcG5fWvlATQAHg5+P9p
BSDN3d80s9vhx89rFpEr+FYC+4Bfh5Q1LphZFeAS4LYc03J+Xgn9/TKzCUB3oK6ZrQceAB4BJpvZ
cOBboH8wbzJwu7vf7O7bzewvwBfBqv7s7sdfbFGm5PFZ/R5IInLYDmCeu99uZo2Bce7eizx+twVu
L4G+hyIiUoroEJ+IiMQlFSgREYlLKlAiIhKXVKBERCQuqUCJiEhcUoESEZG4pAIlCcfM7rJIe5O8
hmTBzPYEjy1zthbIZb5hZva/hdz+/Tme57t+kUSmAiWJ6LdAL3cfEtL27y94lmOZmW6ql4SjAiUJ
xczGAK1inRC5AAACaklEQVSBV4Mmfb/L8V5mMKJ8YTUzszfNbLmZPZBjfa8EIzcvzh692cweASoH
Tduy9+DKm9mzwXxvm1nlYN73zOxhM3sfuNvMWpjZnKAx3Bwzax7Ml9f0F8zsnxZpiLnKzC60SMO5
pWb2QjBP+WC+TIs0kxtZhL+/SEyoQElCcffbiQzo2QMYXUyr7QoMAToA/YPhcABucvfORBra3WVm
ddx9FLDf3Tvk2INrAzzj7mcCO4Hrcqy7prtf6O7/A/wvkT5XZxEZMfqpYJ68pgPUAi4CRgKvBX/n
M4H2ZtYhyNzE3du5e3vg+WL6TEROmAqUyImb7e7b3H0/MB24IJh+l5l9SaSJWzPybmGx2t2zx2Gc
D7TM8d6kHM/PBdKC5y/m2E5e0wFeC8bVywA2uXuGux8FFgfbWQW0NrOng0E/j2/1IhIaFShJZFkc
+xuoVMT1HD+gpVukjXpP4Fx3PxtYmM/6D+Z4foRjB3HeW4jt5jY9e91Hj9vOUaBC0An2bOA94A5g
XD7bEylRKlCSyNYAnQDMrBPQqojrucTMagfnjvoAHwMnAzvcfZ+ZnQ50yzH/4aAHWGF9QqQrLkQO
KX5UwPQCBX2fyrn7NOCPBJ+HSDzQlUGSyKYR6Ru1iEjLhK+LuJ6PiBxaO5VIG4F0M8sAbjezr4Dl
RA7zZRsLfGVmC4A/FGI7dwHPmdm9wBZ+aiOS1/RoNAGeN7Psf6zm1nFXJBRqtyEiInFJh/hERCQu
6RCfSBTM7DLg0eMmr3b3vmHkEUkEOsQnIiJxSYf4REQkLqlAiYhIXFKBEhGRuKQCJSIicUkFSkRE
4tL/BzdIkrGaBdXfAAAAAElFTkSuQmCC
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[2],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);



{% endhighlight %}


![png](data:image/png;base64,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
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[3],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);




{% endhighlight %}


![png](data:image/png;base64,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
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[4],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);




{% endhighlight %}


![png](data:image/png;base64,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
Yv22H+sHWQAAAABJRU5ErkJggg==
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[5],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);
{% endhighlight %}


![png](data:image/png;base64,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
AAAAAElFTkSuQmCC
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[6],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);
{% endhighlight %}


![png](data:image/png;base64,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
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[7],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);
{% endhighlight %}


![png](data:image/png;base64,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
hFjmAAAAAElFTkSuQmCC
)



{% highlight python %}
clf = grid.best_estimator_
names = X.columns
fig, axes = plot_partial_dependence(clf, X=X, features=[8],
                                    feature_names=names, n_cols=2,
                                    n_jobs=3, grid_resolution=20);
{% endhighlight %}


![png](data:image/png;base64,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
RK5CYII=
)



{% highlight python %}
features = [(7,8)]
fig, axs = plot_partial_dependence(clf, X=X, features=features,
                                   feature_names=names)

#F.set_figsize_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
fig.set_figheight(2)
fig.set_figwidth(2)
{% endhighlight %}


{% highlight python %}
from mpl_toolkits.mplot3d import Axes3D


#print('Convenience plot with ``partial_dependence_plots``')

features = [0, 5, 1, 2, (5, 1)]
fig, axs = plot_partial_dependence(clf, X, features,
                                   feature_names=names,
                                   n_jobs=3, grid_resolution=50)
fig.suptitle('Partial dependence of PPSF value')
plt.subplots_adjust(top=0.9, bottom=0.1)  # tight_layout causes overlap with suptitle

#print('Custom 3d plot via ``partial_dependence``')
fig = plt.figure()

target_feature = (1, 5)
pdp, axes = partial_dependence(clf, target_feature,
                               X=X, grid_resolution=100)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of PPSF on size_sqft and '
             'correctional')
plt.subplots_adjust(top=0.85)

plt.show()
{% endhighlight %}


![png](data:image/png;base64,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
AABJRU5ErkJggg==
)



![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAasAAAE3CAYAAAAUtW5VAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzsvXeYHNWVv//equow0z1RGkkjaTSSEEkii2SyvRivvRjM
GpPB2AYvfrz+gf1lFxYwYRdj2MB6cQIDa7B3ZWycMMkmClsmCCMQiwgCIU3OuadjVZ3fH9VV0z05
9EgjqV49/Wi6u+rWreqq+tS559xzlIjg4+Pj4+Mzl9F2dQd8fHx8fHwmwhcrHx8fH585jy9WPj4+
Pj5zHl+sfHx8fHzmPL5Y+fj4+PjMeXyx8vHx8fGZ8/hi5ePj4+Mz5/HFysfHx8dnzuOLlY+Pj4/P
nMeY4vJ+ugsfHx+f2UPNRqML1AJJk57yen30/UFE/noWujRlpipWPj4+Pj67GWnSnMQJU17vUR6f
PwvdmRa+WPn4+PjsBWi7udfHFysfHx+fPRwFKDWNEcY55PjxxcrHx8dnj0f5lpWPj4+Pz9xGAdpu
blnt3lI7R7jtttu47LLLJrXspZdeyg033DCpZXfs2IFSCtM0Z9K9gnPzzTdz0UUX7epuTJrf/OY3
1NTUEI1Gef3113d1d3YJUzlHdxYiwhe+8AUqKio4+uijd/r2169fz9KlS3f6dguNUupupdQ3J1wO
bcqvYdvZoZQ6ddZ2ZAL2CrFavnw5RUVFRKNRFi5cyBe+8AVisdi02hrtBL/uuuu47777CtFVn1ng
6quv5nvf+x6xWIzDDz98xPdKKSKRCNFolCVLlvCNb3wDy7KA8c+dLVu2cNppp1FRUUF5eTlr167l
iSeeAJzzRNM0otGo9/r0pz+983Z6GHPxHN2wYQNPP/00jY2NbNy4cY8Rj9lEKXWpUmpD7mcicoWI
/MsEa6Kpqb/mEnuFWAE8+uijxGIxNm3axKuvvsqtt9465TbmmoXjMznq6upYs2bNuMts3ryZWCzG
s88+y7p167j33nu978Y6dz796U/z8Y9/nLa2Ntrb27nrrrsoLS311lu8eDGxWMx7Pfroo7Ozg7sp
dXV1LF++nEgksqu7MmuMds/YVfeRmVpWu5q51ZudwJIlS/jkJz/JW2+9BcCPf/xjDjzwQEpKSli5
ciX33HOPt6z7pHfHHXewaNEizj//fD75yU/S3NzsPS03NzePGBb73Oc+x6JFiygrK+Okk05iy5Yt
k+qbZVlcffXVzJ8/n5UrV/L444/nfd/X18eXvvQlqqurWbJkCTfccINnATzwwAMcf/zxfO1rX6Os
rIwDDjiAZ599dtLrnnDCCVx99dVUVFSwYsUKnnzySW/d7du3c/LJJ1NSUsLHP/5xOjs78/r18ssv
c9xxx1FeXs6hhx7K+vXrve9OOeUUvvnNb3L88cdTUlLCaaedlrf+hg0bvHVramp44IEHAEilUlx9
9dUsW7aMhQsXcsUVV5BIJEY9brZtc+utt1JbW8uCBQu45JJL6OvrI5VKEY1GsSyLQw89lH322WfC
3+CAAw7gxBNP9M6PXHLPnc7OTrZv387ll19OMBgkGAxy/PHHc8IJU5/L0tfXxyWXXEJVVRW1tbXc
euut2LYNTPzbDOeOO+5gyZIllJSUsP/++3vnQO45+vd///d5Fp9hGNx8880ANDc389nPfpaqqipW
rFjBXXfdNWH/N27cyJFHHklpaSkLFy7kG9/4hvfdT3/6U2pra5k3bx7f+ta3WL58Oc888wz3338/
l112GS+99BLRaJR/+Id/GPXaGs7jjz/O4YcfTmlpKTU1NV6/YWjY/MEHH2TZsmXMnz+fb33rW973
iUSCSy+9lIqKClavXs2rr7467n5t2bKFj3/841RWVrJw4UJuu+02wDk3r7rqKhYvXszixYu56qqr
SKVSwMh7xhe+8IVRPwN47LHHOOywwygvL+e4447jzTff9LatlKpRSv1aKdWhlOpSSn1PKXUgcDfw
EaVUTCnVm132AaXUrTnrXq6U+kAp1a2U+p1SarH73UPWL/jA/oDHrCf4lfUb/mK/hojjmBqQGM9Z
6/m19Vt+bf2Wl+yXAfRxD9LORESm8totqa2tlaefflpEROrr62X16tVyww03iIjIY489Jh988IHY
ti3r16+XoqIiee2110RE5Pnnnxdd1+Uf//EfJZlMSjwel+eff16WLFmS1/5NN90kF154off+/vvv
l/7+fkkmk3LllVfKoYce6n33+c9/Xq6//vpR+/nDH/5Q9t9/f6mvr5euri455ZRTBJBMJiMiImee
eaZ8+ctfllgsJm1tbXLUUUfJ3XffLSIiP/7xj0XXdbnzzjslnU7LQw89JKWlpdLV1TWpdQ3DkB/9
6Edimqb84Ac/kOrqarFtW0REjj32WPn6178uyWRSXnjhBYlGo97+NjY2SmVlpTz++ONiWZY89dRT
UllZKe3t7SIicvLJJ8vKlSvlvffek3g8LieffLJcc801IiJSV1cn0WhU1q1bJ+l0Wjo7O+X1118X
EZErr7xSPv3pT0tXV5f09/fL6aefLtdee+2ox+3++++XffbZR7Zt2yYDAwNy1llnyUUXXeR9D8j7
778/1umR9/2WLVtk4cKFct9994nI2OeObduyatUq+Zu/+Rv5zW9+I62trXltjnaejMXFF18sZ5xx
hvT398v27dtl33339bY/0W+Ty7vvvitLly6VpqYmERHZvn27fPDBByIy8hx1ef3112X+/PmyadMm
sSxLjjjiCLnlllsklUrJtm3bZMWKFfL73/9+3P4fe+yx8pOf/ERERAYGBuSll17yjmUkEpEXXnhB
ksmkfP3rXxdd173j+eMf/1iOP/74KR2z559/Xt58802xLEs2b94sCxYskN/85jfe/gJy2WWXSTwe
lzfeeEOCwaC8/fbbIiJyzTXXyAknnCBdXV1SX18va9asGXN7/f39smjRIvn3f/93SSQS0t/fLy+/
/LKIiHzzm9+UY445Rtra2qS9vV0+8pGPePeTse4Zwz977bXXpKqqSl5++WUxTVMeeOABqa2tFSCE
IxCbgf8EIkAYOEEcUbkU2CA592XgAeDW7N8fAzqBI7JtfRf4YyUVcoFxrgCyWFXL2fpZcqZ+uoQI
ySnaSXKBca58Wv+UfFQ7Wc7Vz5a/1c+UKqoEaMvZzg7gVJmaZhTstdeIVSQSkbKyMlm2bJl85Stf
kXg8PuqyZ555pnznO98REefECwQCkkgkvO8nI1a59PT0CCC9vb0iMr5YffSjH5Uf/vCH3vs//OEP
nli1trZKMBjM6/e6devklFNOERHnwh9+EzvqqKPkJz/5yaTW3WeffbzvBgcHBZCWlhapq6sTXdcl
Fot5359//vne/t5+++15wiAictppp8kDDzwgIo5Y/cu//Iv33fe//335xCc+ISIit912m3zmM58Z
cRxs25bi4mLvRisi8uKLL8ry5ctHPW4f+9jH5Pvf/773/t133xXDMDyRn4xYlZSUSHl5uaxcuVKu
v/56sSxLRMY/dxoaGuSrX/2qrFy5UpRScuKJJ8rWrVtFxDlPlFJSVlbmvX7+85+P2LZpmhIMBmXL
li3eZ3fffbecfPLJIjL+bzOc999/X6qqquTpp5+WdDqd991o52h7e7vU1tbKz372MxERefnll6Wm
piZvmdtuu00uvfTSMY+diMiJJ54oN954o3R0dOR9fsstt8i5557rvY/FYhIIBGYkVsO58sor5aqr
rhKRIbFqaGjwvj/qqKO8/VuxYoU8+eST3nf33HPPmNtbt26dHHbYYaN+t3LlSnn88ce997///e+l
trbW24fR7hnDP7viiis8gXPZb7/9BDgZ+AjQARgy/IY9sVjdD/xrzndRIFNGmVxknC+AnKafKhcZ
58tFxvmyTNXIYdqh3vvc18naiQLEc9rapWK114Su//a3v+XUU0cGsjz55JPccsstbN26Fdu2icfj
HHzwwd73VVVVhMPhSW/Hsiyuv/56Hn74YTo6OtA0Z6S1s7OTsrKycddtbm6mpqbGe19bW+v9XVdX
RyaTobq62vvMtu285ZcsWZI38a+2tpbm5uZJrbto0SLv7+LiYgBisRidnZ1UVFTk+RVqa2tpaGjw
+vXwww/n+WMymQwf/ehHx2zbDVBoaGgYdWiuo6ODeDzO2rVrvc9ExBu2HE5zc3PesaqtrcU0Tdra
2liyZMmo6wxn06ZNrFq1atTvxjp3li5dyve+9z1vX7785S9zySWX8NJLLwGOz6qxsXHc7XZ2dpJO
p0f0v6mpyXs/1m8znFWrVvGd73yHm2++mS1btvCJT3yCO++8k8WLF49YNpPJcPbZZ3PBBRdw3nnn
Ac5v2dzcTHl5ubecZVmceOKJ4+7D/fffz4033sgBBxzAihUruOmmmzj99NNHnM+RSIR58+aN29ZE
vPLKK1x77bW89dZbpNNpUqkUn/vc5/KWGet8G+/6Gs5Y56bbzvDfK3fIcrR7xvDP6urqePDBB/nu
d7/rfZZOpwEWAxZQJyLTcW4tBja5b0QkppTqsrEXuveGIlXk3ScMDExMlFIkJMmr1l9olw5MMm7U
+pzRiL3OZ5VLKpXis5/9LFdffTVtbW309vbyqU99yhvDhZGzvieaBb5u3ToeeeQRnnnmGfr6+tix
YwdAXptjUV1d7YkAQH19vfd3TU0NoVCIzs5Oent76e3tpb+/P88f1tTUlLed+vp6Fi9ePKl1x+tT
T08Pg4ODY/br4osv9trt7e1lcHCQa6+9dsK2a2pq2LZt24jP58+fT1FREVu2bPHa7OvrGzOCc/Hi
xdTV1eX1zzAMFi5cOGEfCkVNTQ1f/epXR/V1jcf8+fMJBAIj+j9ZkR3OBRdcwIYNG6irq0MpxTXX
XDPqcl/72tcoKSnJCzSqqalhxYoVeb/lwMCAF+E4Fvvuuy8/+9nPaG9v55prruHss89mcHBwxPkc
j8fp6uoas53JZFi44IILOOOMM2hoaKCvr48rrrhiUtcWjH99DWescxNGP99yHwhG24/hn9XU1HD9
9dfnHet4PI6I/AxoAJYppUYTiol2thnwlFQpFQHm6WjepGAt55/K/tPQeMPajEJxhnE6FwTO40T9
+Ak2tXPZq8XKfTKrqqrCMAyefPJJnnrqqXHXWbhwIV1dXfT19Y36/cDAAKFQiHnz5hGPx7nuuusm
3Z9zzjmHu+66i8bGRnp6erj99tu976qrqznttNP4f//v/9Hf349t22zbto0XXnjBW8aNSMtkMjz8
8MO88847fOpTn5rUumNRW1vLkUceyU033UQ6nWbDhg15VtRFF13Eo48+yh/+8AcsyyKZTLJ+/foJ
LQqACy+8kGeeeYZf/OIXmKZJV1cXb7zxBpqmcfnll/P1r3+d9vZ2wBHiP/zhD6O2c/755/Of//mf
bN++nVgsxnXXXce5556LYczeQ2FPTw833XQTH3zwAbZt09nZyX//939z7LHHTqkdXdc555xzuP76
6xkYGKCuro4777xzWvPY3nvvPZ577jlSqRThcJiioiJ0faR//J577uGFF15g3bp1nuUPcPTRR1Na
Wsodd9xBIpHAsizeeuutCQMR/ud//scbRXCtMl3XOfvss3nsscfYsGED6XSaG2+80QscGY2Jri1w
rq/KykrC4TAbN25k3bp1Ex0Wj3POOYdvf/vb9PT00NjYmGfVDOf000+ntbWV73znO6RSKQYGBnjl
lVcA53y79dZb6ejooLOzk3/+53+e8u91+eWXc/fdd/PKK68gIgwODvL444+jlCoBNgItwO1KqYhS
KqyUcpWjDViqlAqO0fQ64AtKqcOUUiHgNuAVXeleKLqm8ELTlRqaMGwqk6AKEFZBEiTYYr89pX2a
bfZqsSopKeGuu+7inHPOoaKignXr1nHGGWeMu84BBxzA+eefz8qVKykvLx8RsXTJJZdQW1vLkiVL
WL169ZRuXpdffjmf+MQnOPTQQzniiCP427/927zvf/KTn5BOp1m9ejUVFRWcffbZtLS0eN8fc8wx
vP/++8yfP5/rr7+eX/7yl96wy0Trjse6det45ZVXqKys5JZbbuGSSy7xvqupqeGRRx7htttuo6qq
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
3nrrLQQHB+PcuXO46667EBcXh4qKCuzfvx+A3dB4z549uEgmJjUAACAASURBVPHGG/HXv/6Vr6b2
7t2LdevWYf369Zg5c6bX53S50ddrFsMwzwO4EYAZQAGAewkhuvafLQewEIAVwKOEkH29HW9Ak9Xh
w4fx5ZdfYvfu3TAajdDr9bjzzjv5D4tCocC9996LF154AUD3w4nx8fH4/vvveaFGQ0MDnnvuOf5u
raGhAePGjcOHH34IuVwOk8mEu+++G6dPn0ZYWBg+++wz3tts48aNePvttyGRSLBt2zbMnj27z1+X
voawdbhgwQJeBHHs2DEcOnQIH330EWpqahwGltPT0/HPf/4TQ4cORWJiIrKzs0UTNZhMJuTn58Nk
MmHkyJG8ga5SqXRwlqDCDWH0O63QgoKCXFJJ1tbWoqCgALGxsT67JmG0/OHDh/s0Wt5dq6S3334b
1113XbeP9bRj0Vn0YDQacdttt8FsNuPs2bMoKyvDq6++il27dmHZsmUoKCjAihUrUF1djejoaDz+
+OMYPnw4gA6VYGpqKn744Qex96j6DJfh9vprAMsJIRzDMJsALAewjGGYEQDmAxgJIBbANwzDpBFC
rD0dbEDvWW3cuBHl5eUoLi7Gjh07MH36dHz00UeoqqoC0OGHNmrUKADA3Llz8cEHH4AQgmPHjiEo
KAgxMTF8j3vjxo1ISUlBbW0tZs+ejWXLlmHJkiXQarUICQnB22+/DcD+gQwJCUF+fj6WLFmCZcuW
AbD3y3fs2IHz589j7969+H//7//Bau3x/RmQoAPLM2fOxLp16/D111/jzJkzWL58OWQyGdauXYvk
5GS88847+P7775Gfnw+DweC0wvUGdC7r7NmziIiIQGZmZrdO7/ScExISMHr0aF7Gr1arUVdXhzNn
zuD48eP47bffUFVVhdbWVofzbWtrw08//YSamhpkZmYiMTHRa6IqLy/HggUL8K9//Qt79uzBokWL
+pSoAPeskj766COcOnUKS5cudfuxPUHoGnHp0iU0NDRAqVTij3/8I06fPg2O45CQkIAJEyagrKwM
J0+exI8//sh7V7711ls8UQnPQUxBTV+DAfp8z4oQsp8QQjesjwGIb///mwDsIISYCCFFAPIB9NoK
GNCVVXe44447UFtbC0IIMjIy8NprrwEA5syZg927d/MxGO+++y4AIDQ0FA8//DDWrFmDkJAQfsbm
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
)



{% highlight python %}

{% endhighlight %}


{% highlight python %}

{% endhighlight %}


{% highlight python %}
X = X_all[['amenity_swingerclub', 'amenity_veterinary', 'leisure_motor_track', 'median_income', 'ready_to_build', 'bedrooms', 'amenity_grave_yard', 'amenity_correctional_center', 'lot_size', 'prop_type', 'leisure_dog_park', 'size_sqft']].assign(prop_type = lambda x: x['prop_type'].cat.codes)

for c in X.select_dtypes(['object']).columns:
    X.ix[:,c] = X[c].astype('category').cat.codes

backward_features = util.backward_elimination(model, X, y, get_score=get_score_gini)
{% endhighlight %}

    100%|██████████| 11/11 [00:36<00:00,  1.91s/it]



{% highlight python %}
print 'best score: {:.4f}'.format(np.max(backward_features.score))
print 'best features: {}'.format(backward_features.ix[np.argmax(backward_features.score),:]['features'])
backward_features.sort_values('n').plot(x='n', y='score');
{% endhighlight %}

    best score: 0.2730
    best features: ['amenity_veterinary', 'leisure_motor_track', 'median_income', 'ready_to_build', 'bedrooms', 'amenity_grave_yard', 'lot_size', 'prop_type', 'leisure_dog_park', 'size_sqft']



![png](data:image/png;base64,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
v7eSylUAAAAASUVORK5CYII=
)



{% highlight python %}
backward_features
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n</th>
      <th>score</th>
      <th>features</th>
      <th>drop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14</td>
      <td>0.272717</td>
      <td>[amenity_swingerclub, amenity_hospital, amenit...</td>
      <td>amenity_car_wash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>0.272337</td>
      <td>[amenity_hospital, amenity_veterinary, leisure...</td>
      <td>amenity_swingerclub</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>0.272299</td>
      <td>[amenity_hospital, amenity_veterinary, leisure...</td>
      <td>half_bathrooms</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>0.272506</td>
      <td>[amenity_veterinary, leisure_motor_track, medi...</td>
      <td>amenity_hospital</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>0.272986</td>
      <td>[amenity_veterinary, leisure_motor_track, medi...</td>
      <td>leisure_hockey</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9</td>
      <td>0.272279</td>
      <td>[amenity_veterinary, leisure_motor_track, medi...</td>
      <td>bedrooms</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.272230</td>
      <td>[leisure_motor_track, median_income, ready_to_...</td>
      <td>amenity_veterinary</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.272467</td>
      <td>[leisure_motor_track, median_income, ready_to_...</td>
      <td>leisure_dog_park</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6</td>
      <td>0.271860</td>
      <td>[leisure_motor_track, size_sqft, amenity_grave...</td>
      <td>ready_to_build</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>0.270564</td>
      <td>[amenity_grave_yard, lot_size, leisure_motor_t...</td>
      <td>prop_type</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>0.268477</td>
      <td>[lot_size, leisure_motor_track, median_income,...</td>
      <td>amenity_grave_yard</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>0.266906</td>
      <td>[lot_size, leisure_motor_track, median_income]</td>
      <td>size_sqft</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>0.251186</td>
      <td>[lot_size, median_income]</td>
      <td>leisure_motor_track</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0.214415</td>
      <td>[median_income]</td>
      <td>lot_size</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python %}

{% endhighlight %}
