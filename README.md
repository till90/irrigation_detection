# Irrigation Detection 
This repository is part of a master-thesis. It contains scripts for downloading, preprocessing, classification and visualisation for irrigation detection purpose. Most functions used inside jupyter notebooks are stored in irrigation_detection.py . This repository is under construction and functions may not be appropiate declared.

## Notebooks
### 01_SSM_1km_V1 [Source](https://land.copernicus.eu/global/products/ssm)
Load daily .nc files in nested directories as xarray. Subset and mask out unrelevant regions. Apply conversation from digital to physical values.

### 02_RADOLAN_1km [Source](https://www.dwd.de/DE/leistungen/radolan/radolan_info/home_freie_radolan_kartendaten.html;jsessionid=71EBC7059819330A48470BECD8A5A175.live31093?nn=16102&lsbId=617848)
Download Radolan SF product from FTP server. Exctract files. Load extracted files to xarray. Subset data. Apply conversation from digital to physical values.

### 03_NDVI_10Dmax_1km [Source](https://land.copernicus.eu/global/products/ndvi)
Download data from ftp server after place an free order. Extract archieves. Load extracted files to xarray. Subset data. Apply conversation from digital to physical values. Find irrigation period.

04_LUCAS_TOPSOIL

05_BDFL5

06_Evapo_rel_pot

07_Sentinel_1_grd

08_vegetation_indizes

09_corine_land_Cover

10_surface_soil_moisture
