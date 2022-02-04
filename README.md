# Irrigation Detection 
This repository is part of a master-thesis. It contains scripts for downloading, preprocessing, classification and visualisation for irrigation detection purpose. Main part will be implementing of irrigation detection method from [Irrigation Events Detection over Intensively Irrigated Grassland Plots Using Sentinel-1 Data](https://doi.org/10.3390/rs12244058) in V1. and from [Near Real-Time Irrigation Detection at Plot Scale Using Sentinel-1 Data](https://www.mdpi.com/2072-4292/12/9/1456) in V2.

This repository is under construction and functions may not work or be whack explained.

## To use some functions additional Accounts are requiered:    
### Google Earth Engine mainly for processing and downloading Satellite data. You can apply [here](http://code.earthengine.google.com/)   
### International Soil Moisture Network to get in-situ soil moisture time series data. You can apply [here](https://ismn.geo.tuwien.ac.at/en/accounts/signup/)    
    

Most functions used inside jupyter notebooks are stored in irrigation_detection.py .
### irrigation_detection.py Functions so far (alphabetical):

#### add_ndvi(image, dateoffset=15)
#### download_NDVI_max(user, passwd, pathFTP, download_path)
#### download_evapo(from_year, to_year, download_path, real=True)
#### download_radolan_SF(from_year, to_year, download_path)
#### export_values(latitude, longitude, ds, name)
#### extract_evapo(path)
#### get_ERA5_ts(lon, lat, ismn_idx, start, end, red, scale, crs)
#### get_ismn_data(filepath, variable, min_depth, max_depth, landcover, network)
#### get_s1_grd_mean(path, start, end, outname, with_ndvi, dateoffset)
#### get_s1_ts(lon, lat, ismn_idx, start, end, pol, mode, res, red, scale, crs)
#### get_s2_ts(lon, lat, ismn_idx, start, end, red, scale, crs)
#### mask_by_landcover(image)
#### merge_s1_s2_era5(gdf_s1, gdf_s2, gdf_era5, driver, filepath)
#### merge_sentinel_ismn(files, ismn_path, driver, out)
#### open_NDVI(path)
#### open_evapo(path, real=True)
#### open_radolan_SF(path)
#### roi_ds(path, ds)
#### search_files(path, filetype)
#### subset_ds(path, ds)
#### unpack_radolan_SF(path)
#### unzip_ndvi(pathZip, pathData)


## Notebooks
### 01_SSM_1km_V1 [Product Source | World](https://land.copernicus.eu/global/products/ssm)
Load daily .nc files in nested directories as xarray. Subset and mask out unrelevant regions. Apply conversation from digital to physical values.

### 02_RADOLAN_1km [Product Source | Germany](https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/)
Download Radolan SF product from FTP server. Exctract files. Load extracted files to xarray. Subset data. Apply conversation from digital to physical values.

### 03_NDVI_10Dmax_1km [Product Source | Europe](https://land.copernicus.eu/global/products/ndvi)
Download data from ftp server after place an free order. Extract archieves. Load extracted files to xarray. Subset data. Apply conversation from digital to physical values. Find irrigation period.

### 04_LUCAS_TOPSOIL [Product Source | Europe](https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data)
Preprocess LUCAS TOPSOIL 2015 and 2009 Datasets. Merge to Geopandas GeoDataFrame. Subset and Export data.

### 05_BDFL5 [Product Source | Hessen](https://www.hlnug.de/fileadmin/dokumente/boden/BFD5L/methoden/m46.html) - [Product Source | RLP](https://www.geoportal.rlp.de/linked_open_data/)
Preprocess Soil Information dataset. Spatial join with LUCAS TOPSOIL data. Subset and Export data.

### 06_Evapo_rel_pot[Product Source | Germany](https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/)
Download potentiel and real Evaporation. Transform projection with Gdal. Load extracted files to xarray. Subset data. Apply conversation from digital to physical values

### 07_Sentinel_1_grd[Product Source | World](https://developers.google.com/s/results/earth-engine/datasets?q=sentinel%20)
Use Earth Engine Cloudcomputing. Get Sentinel - 1 GRD VV Mean for geometry while masking high NDVI values. Construct Geopandas GeoDataFrame and add all Information from filename.SWI map expors. Extract data for plot and grid size

### Export_ImageCollection_fromGee
Query data with GEE and geemap to get 10m plot time series and 50m grid(10x10km) time series data for different Obversation modes of Sentinel 1. Get NDVI, SWI, VV, Angle. Download as tif.

### 10_surface_soil_moisture Not Working!!
PYSMM python module to extract point or grid soil moisture information with GEE. 

### 11_Irrigation_Detection_V1
Irrigation detection with data from 07_Sentinel_1_grd notebook

#### 12_query_data
Acces all exported datasets from previous notebooks. Actual: LTS, BDFL5, NDVI10Dmax, RADOLAN, SSM, EVAPO_R, EVAPO_P, ... . Subset data. Preprocess and merge

### 13_Visualize
ISMN Dashboard for interactive visualization of Sentinel 1&2 and soil moisture data. Load data from Irrigation_Detection_V1 notebook. Split into equal satellite groups. Matplotlib and hvplot plotting.

### Ground_Trouth_data
Extract TimeSeries data from ismn stations. Find corresponding Sentinel 1, Sentinel 2 and ERA5 pixels and extract all information to geojson file. Merge data and add in-situ soil moisture values closest to satellite derived data. 

### CNN & CNN_MLP_Regression_soil_moisture
Use Keras and Tensorflow to train a MLP regression model with Sentinel 1, Sentinel 2 and ERA 5 data. Visualize and Analyse trained model with test data.

### Irrigation_map_whr Not working!!
Extract features from pdf file

### old_notebook_01
GEE stuff from first project. Visualizations

### old_notebook_02
Statistical and machine learning Stuff from first project. Seasonal Mean, Standart Deviation, Histogram, Correlation, Covariance, Spearman, Pearson, Cross correlation, Windowed time lagged cross correlation, rolling window, Dynamic Time Warping, Clustering, precipitation analysis, Seasonal Patterns, show measurements on map, univariate description, bivariate description, Preprocessing, geographical fescription 

### old_notebook_03
clustering images to get soil data from image

### old_notebook_04
Matrix Profile, Clustering, Biclustering, Clustering Time Series, Stump, statistical stuff 

# Usefull Links

## Computing, Visualize, Generate

[Google Earth Engine | Geopsatial big data computing with many datasets](https://code.earthengine.google.com/)

[geojson.io | tool for creating, viewing, and sharing maps](http://geojson.io/#map=2/20.0/0.0)

## Data

### Portals

[Geoportal Hessen | raumbezogenen Daten der Geodateninfrastruktur Hessen (GDI-HE)](https://www.geoportal.hessen.de/)

[DWD Opendata FTP Server | climate enviromentl data Germany](https://opendata.dwd.de/)

[Earth Engine Data Catalog | Climate, Weather, Imagery and Geophysical datasets](https://developers.google.com/earth-engine/datasets)

[Copernicus Open Access Hub | Sentinel Data Download](https://scihub.copernicus.eu/)

[THEIA | Products and Algorithms for Land Surfaces](https://www.theia-land.fr/en/homepage-en/)

[irpi hydrologie | Satellite and Ground Data, Code](http://hydrology.irpi.cnr.it/download-area/)

### Specific Envrioment Data

[Bodeninformationen | Hessen - HLNUG](https://www.hlnug.de/themen/boden/information)

[Soil Moisture Network ISMN | global in-situ soil moisture database](https://ismn.geo.tuwien.ac.at/en/)

[THEIA IRSTEA SOIL MOISTURE CATALOG | radar signal inversion algorithm uses neural networks(https://thisme.cines.teledetection.fr/home)

### Remote Sensing

[IDB Index DataBase | A database for remote sensing indices](https://www.indexdatabase.de/)

[Sentinel Online | Official Sentinel Missions Homepage](https://sentinels.copernicus.eu/web/sentinel/home)

[Irrigation+ | quantitative, accurate and routinely monitoring of irrigation](https://esairrigationplus.org/)

## Python modules

### Geospatial Data Storage and Processing

[geemap | interactive mapping with Google Earth Engine, ipyleaflet and ipywidgets](https://geemap.org/)

[xarray | N-D labeled arrays and datasets in Python](http://xarray.pydata.org/en/stable/)

[pandas | data analysis and manipulation tool](https://pandas.pydata.org/)

[Geopandas | geospatial data handling](https://geopandas.org/en/stable/)

### Machine learning, Classification, Statistics

[tslearn | machine learning tools for time series data](https://tslearn.readthedocs.io/en/stable/)

[tsfresh | automatically calculates a large number of time series characteristic features(https://tsfresh.readthedocs.io/en/latest/)


### Soil Moisture

[PYSMM | Sentinel-1 soil-Moisture Mapping Toolbox](https://pysmm.readthedocs.io/en/latest/)

[ismn | Reader for the data from the Soil Moisture Database(ISMN)](https://ismn.readthedocs.io/en/latest/)

[Cosmic-Ray Sensor PYthon tool (crspy) | Cosmic Ray Sensor data into soil moisture estimates](https://github.com/danpower101/crspy)

### Plotting

[colorbars | matplotlib library for colorbars](https://matplotlib.org/stable/tutorials/colors/colormaps.html#qualitative)

[matplotlib | creating static, animated, and interactive visualizations in Python](https://matplotlib.org/stable/index.html)

[hvPlot | A high-level plotting API for the PyData ecosystem built on HoloViews](https://hvplot.holoviz.org/)

