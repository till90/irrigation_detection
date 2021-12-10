# Irrigation Detection 
This repository is part of a master-thesis. It contains scripts for downloading, preprocessing, classification and visualisation for irrigation detection purpose. Most functions used inside jupyter notebooks are stored in irrigation_detection.py . This repository is under construction and functions may not be appropiate declared.

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

08_vegetation_indizes

09_corine_land_Cover

10_surface_soil_moisture

# Usefull Links
## Data
### Portals
[Geoportal Hessen | raumbezogenen Daten der Geodateninfrastruktur Hessen (GDI-HE)](https://www.geoportal.hessen.de/)

[DWD Opendata FTP Server | climate enviromentl data Germany](https://opendata.dwd.de/)

[Earth Engine Data Catalog | Climate, Weather, Imagery and Geophysical datasets](https://developers.google.com/earth-engine/datasets)
