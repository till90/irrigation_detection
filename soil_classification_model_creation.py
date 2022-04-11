def filter_dataframe(df, column_name, condition, arg1, index=False):
    import pandas as gpd
    
    # Filter rows with digits in cell
    if condition == 'is_number':
        df = df[df[column_name].astype(str).str.isdigit()]
    elif condition == 'isin' and arg1 is not None:
        if index == True: 
            df = df[df.index.isin(arg1)] 
        else:
            df = df[df[column_name].isin(arg1)]
    elif condition == 'equal' and arg1 is not None:
        df = df[df[column_name] == arg1]
    
    return df

def rename_columns(df, rename_cols):
    import pandas as pd
    
    # Rename columns names
    df.rename(rename_cols, inplace=True, axis=1) 

    return df

def merge_lts_data(df_09, df_15):
    import pandas
    
    # Update missing landcover information from the 2015 survey
    df_09.update(df_15)
    
    return df_09

def load_gdf(filepath, convert_date,convert_temperature_to_celvin, create_doy, dummies_columns, add_na_columns,columns, dropna, index_column, p_dtypes, p_head):
    from glob import glob
    import geopandas as gpd
    import pandas as pd
    
    files = glob(filepath + '*')
    gdf = pd.concat([gpd.read_file(x, ignore_index=True) for x in files])
    initial_rows = len(gdf)
    
    # Conversation
    if convert_date == True: gdf['date'] = gdf.loc[:,'date'].astype('datetime64[ns]')
    if convert_temperature_to_celvin is not None:
        for i, col in enumerate(convert_temperature_to_celvin):
                    gdf[convert_temperature_to_celvin[i]] = gdf.loc[:,convert_temperature_to_celvin[i]] -273.15
    # Column Creation
    if create_doy == True: gdf['day_of_year'] = gdf.loc[:,'date'].dt.day_of_year
    if dummies_columns is not None:
        gdf = pd.get_dummies(gdf, columns = dummies_columns)
    if add_na_columns is not None:
        gdf[add_na_columns] = pd.NA
    # Column Selection
    gdf = gdf.loc[:,columns]
    #Set Index
    if index_column is not None: gdf.set_index('POINT_ID', inplace=True)
    # Clear Rows
    if dropna == True: gdf.dropna(inplace = True)
    final_rows = len(gdf)
    # Prints
    print(f'Total rows: {final_rows} \nRemoved rows: {initial_rows - final_rows}')
    if p_dtypes == True: print(f'Column dtypes: \n\n{gdf.dtypes}')
    if p_head is not None: 
          gdf.head(p_head)
    
    return gdf

def load_df(filepath, rename_cols, filter_rows, index_column, add_na_columns):
    import pandas as pd
    
    #Check filetype
    filetype = filepath.split('.')[-1]
    if filetype == 'xlsx':
        df = pd.read_excel(filepath)
    elif filetype == 'csv':
        df = pd.read_csv(filepath)
    
    # Rename columns
    if rename_cols is not None: 
        df = rename_columns(df, rename_cols)
        
    # Filter rows with digits in cell
    if filter_rows is not None:
        df = filter_dataframe(df = df,
                     column_name = filter_rows['column_name'],
                     condition = filter_rows['condition'],
                     arg1 = None)
        
    # Set index
    if index_column is not None: df.set_index(index_column, inplace=True)

    # Add columns from 2015 survey 
    if add_na_columns is not None:
        df[add_na_columns] = pd.NA
        
    return df

def load_LTS_combined(filepath_09, filepath_15, subset, landcover_filter):
    # LUCAS TOPSOIL V1 2009
    df_09 = load_df(
        filepath = filepath_09, 
        rename_cols = None, 
        filter_rows = {'column_name' : 'POINT_ID', 'condition' : 'is_number'},
        index_column = 'POINT_ID', 
        # Add columns from 2015 survey 
        add_na_columns = ['Elevation','LC1', 'LU1', 'Soil_Stones', "NUTS_0","NUTS_1","NUTS_2","NUTS_3", "LC1_Desc","LU1_Desc"])
    len_09 = len(df_09)
    
    # LUCAS TOPSOIL 2015
    df_15 = load_df(
        filepath = filepath_15, 
        rename_cols = {'Point_ID' : 'POINT_ID'}, 
        filter_rows = {'column_name' : 'POINT_ID', 'condition' : 'is_number'},
        index_column = 'POINT_ID', 
        add_na_columns = None,)
    len_15 = len(df_15)
    
    # Update missing data in LTS 2009 V1 with data from LTS 2015
    df = merge_lts_data(df_09, df_15)
    len_09_update = len(df)
    
    # Subset data to specific Country by POINT ID 
    if subset is not None:
        df_DE = load_df(filepath = subset,
                     rename_cols = {'Point_ID' : 'POINT_ID'},
                     index_column = 'POINT_ID',
                     filter_rows = None,
                     add_na_columns = None)
    
        POINT_ID_DE = list(df_DE.index)
        df = filter_dataframe(df, column_name = None, index = True, condition = 'isin', arg1 = POINT_ID_DE)
    len_09_subset = len(df)
    
    landcover_units = list(df['LC1_Desc'].unique())
    column_names = list(df.columns)
    if landcover_filter is not None:
        df = filter_dataframe(df, column_name ='LC1_Desc', condition = 'equal', arg1 = landcover_filter)
    len_09_landcover = len(df)
    if POINT_ID_DE is not None: print(f' Subset rows: {len(POINT_ID_DE)}')
    print(f'Initial rows LTS 09: {len_09}, LTS 15: {len_15} \nAfter updating: {len_09_update} \nAfter subsetting: {len_09_subset} \nAfter Landcover filtering: {len_09_landcover}\nRemoved rows: {len_09 - len_09_landcover}')
    print(f'Landcover Units: \n{landcover_units}')
    print(f'Column Names: \n{column_names}')
    return df

def gdf_sjoin_LTS(df, gdf):
    import geopandas as gpd
    import pandas as pd
    
    # Load LUCAS TOPSOIL DATABASE
    gdf_lts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = 'epsg:4326')
    # Join LTS data to S1, S2, Era5 Data
    gdf = gpd.sjoin_nearest(gdf, gdf_lts, how='left', distance_col='distance_lts')
    
    print(f'Length after mergin Satellite Data with LTS stuff: {len(gdf)}')
    print(f' Maximal distance between Downloaded Satellite data and LTS: {gdf.distance_lts.max()}')
        
    return gdf

def add_soil_classes(gdf, filepath_lts_image):

    import geopandas as gpd
    import soiltexture
    import contextily as cx
    import matplotlib.pyplot as plt
    USDA_class, FAO_class, INTERNATIONAL_class, ISSS_class = list(), list(), list(), list()

    for index, row in gdf.iterrows():
        USDA_class.append(soiltexture.getTexture(row.sand, row.clay, classification='USDA'))
        FAO_class.append(soiltexture.getTexture(row.sand, row.clay, classification='FAO'))
        INTERNATIONAL_class.append(soiltexture.getTexture(row.sand, row.clay, classification='INTERNATIONAL'))
        ISSS_class.append(soiltexture.getTexture(row.sand, row.clay, classification='ISSS'))

    gdf['USDA'] = USDA_class
    gdf['FAO'] = FAO_class
    gdf['INTERNATIONAL'] = INTERNATIONAL_class
    gdf['ISSS'] = ISSS_class
    
    print(gdf['USDA'].value_counts(),gdf['FAO'].value_counts(),gdf['INTERNATIONAL'].value_counts(),gdf['ISSS'].value_counts())
    
    # for all rows and lts sample points
    FAO_to_numerical = {'FAO':     {'medium': int(0), 'coarse': int(1), 'fine' : int(2)}}
    gdf['FAO_nr'] = gdf.replace(FAO_to_numerical)['FAO']

    USDA_to_numerical = {'USDA':     {'silt loam': int(0), 'sandy loam': int(1), 'loam' : int(2),'silty clay loam': int(3), 
                                    'loamy sand': int(4), 'silty clay' : int(5),'clay loam': int(6), 'sand': int(7), 
                                    'clay' : int(8), 'sandy clay loam' : int(9), 'silt' : int(10)}}
    gdf['USDA_nr'] = gdf.replace(USDA_to_numerical)['USDA']

    INTERNATIONAL_to_numerical = {'INTERNATIONAL':     {'silty loam': int(0), 'silty clay loam': int(1), 'loamy sand' : int(2),
                                                        'silty clay': int(3), 'loam': int(4), 'sand' : int(5),'clay': int(6),
                                                        'sandy loam': int(7)}}
    gdf['INTERNATIONAL_nr'] = gdf.replace(INTERNATIONAL_to_numerical)['INTERNATIONAL']

    ISSS_to_numerical = {'ISSS':     {'silty clay': int(0), 'silty clay loam': int(1), 'clay loam' : int(2),'sandy loam': int(3),
                                    'light clay': int(4), 'loam' : int(5),'heavy clay': int(6), 'loamy sand': int(7), 
                                    'silt loam': int(8), 'sand': int(9), 'sandy clay loam': int(10)}}
    gdf['ISSS_nr'] = gdf.replace(ISSS_to_numerical)['ISSS']
    
    print(f'Length after adding soil classes: {len(gdf)}')
    print(gdf['USDA_nr'].value_counts(),gdf['FAO_nr'].value_counts(),gdf['INTERNATIONAL_nr'].value_counts(),gdf['ISSS_nr'].value_counts())
    
    if filepath_lts_image is not None:
        # Visualize Points
        fig, ax = plt.subplots(1, dpi=500)
        gdf_plot = gpd.GeoDataFrame(geometry = gdf.geometry.unique()).plot(ax=ax, markersize=2, color='red')
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world_DE = world.loc[world['name'] == 'Germany'].boundary
        world_DE.plot(ax=ax)
        cx.add_basemap(ax=ax, source = cx.providers.Esri.WorldImagery, crs=gdf.crs, zoom = 'auto')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.text(x=0.5, y=0.92, s=f'{len(gdf.geometry.unique())} LUCAS Topsoil Data Locations', fontsize=12, ha="center", transform=fig.transFigure)
        plt.text(x=0.5, y=0.88, s=f"Landcover Class: {gdf.LC1_Desc.iloc[1]}", fontsize=8, ha="center", transform=fig.transFigure)
        plt.subplots_adjust(top=0.865, wspace=0.3)
        fig.savefig(filepath_lts_image)
        
    return gdf

def add_radolan(gdf, filepath_radolan):
    import xarray as xr
    import geopandas as gpd
    import pandas as pd
    import datetime as dt
    
    RADOLAN = xr.open_dataset(filepath_radolan)
                
    pp_24h, pp_48h, pp_72h = list(), list(), list()
    midday = dt.time(12,0)

    for index, row in gdf.iterrows():
        pp_loc = RADOLAN.sel(latitude=row.geometry.y, longitude=row.geometry.x, method='nearest').sel(time=slice(row.date.date() - pd.Timedelta('3d'), row.date.date())).precipitation_1km
        if row.date.time() > midday:
            try:
                pp_24h.append(pp_loc.isel(time=3).values)
                pp_48h.append(pp_loc.isel(time=slice(2,4)).sum().values)  
                pp_72h.append(pp_loc.isel(time=slice(1,4)).sum().values)  
            except:
                pp_24h.append(pd.NA)
                pp_48h.append(pd.NA)
                pp_72h.append(pd.NA)
        else:
            try:
                pp_24h.append(pp_loc.isel(time=2).values)
                pp_48h.append(pp_loc.isel(time=slice(1,3)).sum().values)  
                pp_72h.append(pp_loc.isel(time=slice(0,3)).sum().values)  
            except:
                pp_24h.append(pd.NA)
                pp_48h.append(pd.NA)
                pp_72h.append(pd.NA)

    gdf['pp_24h'] = pp_24h
    gdf['pp_48h'] = pp_48h
    gdf['pp_72h'] = pp_72h
    print('Add Radolan data')   
    return gdf

def add_soil_moisture(gdf, filepath_ssm_model, filepath_df):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    import geopandas as gpd

    gdf_predict = gdf
    # Load pre trained model
    cnn_basic_model = load_model(filepath_ssm_model)

    features = ['VV', 'VH', 'angle', 'NDVI', 'platform_A', 'platform_B', 'orbit_ASCENDING',
                    'orbit_DESCENDING', 'day_of_year','mean_2m_air_temperature','maximum_2m_air_temperature',
                    'minimum_2m_air_temperature', 'Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 
                    'RedEdge4', 'NIR', 'WaterVapor', 'SWIR1', 'SWIR2']

    #Normalization
    scaler = MinMaxScaler(feature_range=(0, 1), copy = False)
    gdf_predict[features] = scaler.fit_transform(X = gdf_predict[features])

    # Predict Surface Soil Moisture ValuesValues
    predict_ssm_scale = cnn_basic_model.predict(gdf_predict[features])

    # Add predicted smm values to initial dataframe
    gdf['ssm'] = predict_ssm_scale
    print('Add soil moisture data')
    
    if filepath_df is not None: gdf.to_csv(filepath_df)
           
    return gdf

def create_pipe(gdf, load_from_file, features_normalization, plot_normalize_to, features_ml, drop_frozen_grounds, filter_months, shuffle, apply_bootstrapping, soil_class, test_size,random_state):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import geopandas as gpd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    gdf_p = gdf.copy()
    if load_from_file == False: gdf_p.reset_index(inplace=True)
    print(f'Initial Rows before preparing data for machine learning: {len(gdf)}')
    gdf_p.dropna(how='any', subset=['FAO', 'FAO_nr', 'ISSS_nr', 'ISSS', 'pp_24h', 'pp_48h', 'pp_72h'], inplace=True)
    

    # Dtype Transformation
    gdf_p['date'] = gdf_p.loc[:,'date'].astype('datetime64[ns]')
    gdf_p['FAO_nr'] = gdf_p.loc[:,['FAO_nr']].astype('int64')
    gdf_p['USDA_nr'] = gdf_p.loc[:,['USDA_nr']].astype('int64')
    gdf_p['INTERNATIONAL_nr'] = gdf_p.loc[:,['INTERNATIONAL_nr']].astype('int64')
    gdf_p['ISSS_nr'] = gdf_p.loc[:,['ISSS_nr']].astype('int64')
    gdf_p[['pp_24h', 'pp_48h', 'pp_72h']] = gdf_p.loc[:,['pp_24h', 'pp_48h', 'pp_72h']].astype('float64')
    gdf_p['platform_A'] = gdf_p.loc[:,'platform_A'].astype('int64')
    gdf_p['platform_B'] = gdf_p.loc[:,['platform_B']].astype('int64')
    gdf_p['orbit_ASCENDING'] = gdf_p.loc[:,['orbit_ASCENDING']].astype('int64')
    gdf_p['orbit_DESCENDING'] = gdf_p.loc[:,['orbit_DESCENDING']].astype('int64')
    gdf_p['day_of_year'] = gdf_p.loc[:,['day_of_year']].astype('int64')

    #Normalization
    scaler = MinMaxScaler() 
    gdf_p.loc[:,features_normalization] = scaler.fit_transform(gdf_p.loc[:,features_normalization]) 
    
    if plot_normalize_to is not None:
        df = pd.DataFrame(gdf_p.iloc[:,2:6])
        df_std = (df.subtract(df.mean())) / df.std()
        df_std = df_std.melt(var_name='Column', value_name='Normalized')
        fig, ax = plt.subplots(1, dpi=300)
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(df.keys(), rotation=90)
        plt.text(x=0.5, y=0.92, s=f'{len(gdf_p.geometry.unique())} LUCAS Topsoil Normalized Distribution', fontsize=12, ha="center", transform=fig.transFigure)
        plt.text(x=0.5, y=0.88, s=f"Landcover Class: {gdf_p.LC1_Desc.iloc[1]}", fontsize=8, ha="center", transform=fig.transFigure)
        plt.subplots_adjust(top=0.865, wspace=0.3)
        fig.savefig(plot_normalize_to)
    
    # Drop frozen grounds
    if drop_frozen_grounds == True:
        gdf_p = gdf_p[gdf_p['minimum_2m_air_temperature'] > 0]
    
    # Filter Months
    if filter_months is not None:
        gdf_p = gdf_p[gdf_p['date'].dt.month.isin(filter_months)]
        
    # Shuffle dataset
    if shuffle == True:
        gdf_p = gdf_p.sample(frac=1, random_state = random_state)
        
    # Create equal soil classes entries 
    print(gdf_p[soil_class].value_counts())
    if apply_bootstrapping == True:
        for x in dict(gdf_p[soil_class].value_counts()).keys():
            if len(gdf_p[gdf_p[soil_class] == x]) < 20000:
                gdf_p = gdf_p.append(gdf_p[gdf_p[soil_class] == x].sample(n=(20000 - len(gdf_p[gdf_p[soil_class] == x])) , replace=True, random_state=random_state))
            elif len(gdf_p[gdf_p[soil_class] == x]) > 20000:
                index = gdf_p[gdf_p[soil_class] == x].sample(n=(len(gdf_p[gdf_p[soil_class] == x]) - 20000), random_state = random_state).index
                gdf_p.drop(labels = index, inplace=True)
    print(gdf_p[soil_class].value_counts())
    gdf_p.dropna(how='any', subset=features_ml + [soil_class], inplace=True)

    X = gdf_p[features_ml].values
    y = gdf_p[soil_class].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
    
    FAO_to_numerical = {'FAO':     {'medium': 0, 'coarse': 1, 'fine' : 2}}
    USDA_to_numerical = {'USDA':     {'silt loam': 0, 'sandy loam': 1, 'loam' : 2,'silty clay loam': 3, 
                                    'loamy sand': 4, 'silty clay' : 5,'clay loam': 6, 'sand': 7, 
                                    'clay' : 8, 'sandy clay loam' : 9, 'silt' : 10}}
    INTERNATIONAL_to_numerical = {'INTERNATIONAL':     {'silty loam': 0, 'silty clay loam': 1, 'loamy sand' : 2,
                                                        'silty clay': 3, 'loam': 4, 'sand' : 5,'clay': 6,
                                                        'sandy loam': 7}}
    ISSS_to_numerical = {'ISSS':     {'silty clay': 0, 'silty clay loam': 1, 'clay loam' : 2,'sandy loam': 3,
                                    'light clay': 4, 'loam' : 5,'heavy clay': 6, 'loamy sand': 7, 
                                    'silt loam': 8, 'sand': 9, 'sandy clay loam': 10}}
    
    label_list = {'USDA' : list(USDA_to_numerical['USDA'].values()), 
                  'ISSS' : list(ISSS_to_numerical['ISSS'].values()),
                  'FAO' : list(FAO_to_numerical['FAO'].values()),
                  'INTERNATIONAL' : list(INTERNATIONAL_to_numerical['INTERNATIONAL'].values())}
    key_list = {'USDA' : list(USDA_to_numerical['USDA'].keys()), 
              'ISSS' : list(ISSS_to_numerical['ISSS'].keys()),
              'FAO' : list(FAO_to_numerical['FAO'].keys()),
              'INTERNATIONAL' : list(INTERNATIONAL_to_numerical['INTERNATIONAL'].keys())}

    labels = label_list[soil_class]
    target_names = key_list[soil_class]
    
    print(f'Results Rows after preparing data for machine learning: {len(gdf_p)}')
    
    return X, y, X_train, X_test, y_train, y_test, labels, target_names, gdf_p

