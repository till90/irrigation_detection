﻿def search_files(path, filetype):
    """
    Arguments: path to tar files, file extension
    """

    import os

    path_list = []  # empty list where file pahts will be added
    # create and iterate through os.walk() generator
    for root, dirs, files in os.walk(path):
        for file in files:  # loop through founded files in each dir
            # if any of these files endswith "filetype"...
            if file.endswith(filetype):
                # ...then append this file to path_list while joining root dir
                path_list.append(os.path.join(root, file))

    return path_list  # return list with full file paths


def subset_ds(path, ds):
    """
    Return subsetted xarray dataset. Arguments path to shape)
    """

    import salem

    # load shapefile with salem
    aoi = salem.read_shapefile(path)
    # subset xarray dataset
    ds_subset = ds.salem.subset(shape=aoi, margin=10)

    return ds_subset


def roi_ds(path, ds):
    """
    Return roi subsetting xarray dataset. Arguments path to shape)
    """

    import salem
    # load shapefile with salem
    aoi = salem.read_shapefile(path)

    # mask out unrelevant data
    ds_subset = ds_subset.salem.roi(shape=aoi)

    return ds_subset


def export_values(latitude, longitude, ds, name):
    """
    Find nearest cell and export it to csv. Arguments: lat, lon, dataset, filename
    """
    import pandas as pd

    # select data
    point_1 = ds.sel(lat=latitude, lon=longitude, method='nearest')
    # export with pandas
    return point_1.to_pandas().round(2).to_csv(name)


def download_radolan_SF(from_year, to_year, download_path):
    """
        Download radolan SF product for desired years to local drive
        """

    #import modules
    import ftplib
    import os

    # directory where files will be saved
    radolanSF_download = download_path

    # create folder in local space when not already there
    if not os.path.exists(radolanSF_download):
        os.makedirs(radolanSF_download)

    # path to historic files on ftp server
    path_historical = 'climate_environment/CDC/grids_germany/daily/radolan/historical/bin/'

    # ftp server
    ip = "141.38.2.26"  # domain "https://opendata.dwd.de/"
    # create range of years to download
    years = [str(year) for year in list(range(from_year, to_year + 1))]
    print("Try to download Radolan SF product for " + str(years))

    # create Connection
    with ftplib.FTP(ip) as ftp:
        ftp.login()  # Login to ftp server
        ftp.cwd(path_historical)  # Change path to historic data

        # download data
        for year in years:  # loop through years list
            if year in ftp.nlst():  # check if year is in historic folder
                ftp.cwd(year)  # enter childfolder
                file_list = ftp.nlst()  # list all files inside folder
                for file in file_list:  # loop through files inside file_list
                    # open data on local drive
                    with open(os.path.join(radolanSF_download, file), 'wb') as data:
                        # write data to file
                        ftp.retrbinary('RETR %s' % file, data.write)
                    print(file + " downloaded succsefully!")
                ftp.cwd('..')  # go back to parent folder
            else:  # year is not in historic folder must be recent year
                ftp.cwd('../..')  # move to parent folder
                ftp.cwd('recent/bin')  # enter recent folder
                file_list = ftp.nlst()  # list all files inside recent folder
                for file in file_list:  # loop through file_list
                    # open file on local drive
                    with open(os.path.join(radolanSF_download, file), 'wb') as data:
                        # write data to file
                        ftp.retrbinary('RETR %s' % file, data.write)
                    print(file + " downloaded succsefully!")

    return print("Download completed")


def unpack_radolan_SF(path):
    """
    Arguments: path to tar files
    """

    from irrigation_detection import search_files
    import os
    import tarfile as tar
    import gzip
    import shutil

    files = search_files(path, ".tar.gz")
    for file in files:
        tardude = tar.open(file)
        members = tardude.getmembers()
        members = [x for x in members if x.name.endswith("2350-dwd---bin")]
        for member in members:
            tardude.extract(member, path=path)
            with open(os.path.join(path, member.name), 'rb') as f_in:
                with gzip.open(os.path.join(path, member.name + ".gz"), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    return print("Job Done")


def open_radolan_SF(path):
    """Return Xarray dataset Timeseries, Arguments (path)   """

    #import modules
    import xarray as xr
    import wradlib as wrl
    import pandas as pd

    # xarray dataset from SF 23:50 Product
    radolan_SF = xr.open_mfdataset(path, engine="radolan", decode_cf=False)

    # Konvert float64 to datetime64
    radolan_SF['time'] = pd.to_datetime(radolan_SF.time.values, unit='s')

    # Replace coordinates with projected wgs84 lan lot predefined for radolan 900x900 cells
    radolan_grid_ll = wrl.georef.get_radolan_grid(900, 900, wgs84=True)
    lat = radolan_grid_ll[:, 0, 1]
    lon = radolan_grid_ll[0, :, 0]
    radolan_SF = radolan_SF.assign_coords({'lat': (lat), 'lon': lon})
    radolan_SF = radolan_SF.drop_vars(
        ['x', 'y']).rename({'y': 'lat', 'x': 'lon'})

    return radolan_SF


def download_evapo(from_year, to_year, download_path, real=True):
    """
    Download radolan SF product for desired years to local drive
    """

    #import modules
    import ftplib
    import os
    from tqdm import tqdm

    # create folder in local space when not already there
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # path to parent directory with files on ftp server
    ftp_path = 'climate_environment/CDC/grids_germany/daily/evapo_r/'
    if real:
        ftp_path = 'climate_environment/CDC/grids_germany/daily/evapo_r/'
    else:
        ftp_path = 'climate_environment/CDC/grids_germany/daily/evapo_p/'
    # ftp server
    ip = "141.38.2.26"  # domain "https://opendata.dwd.de/"

    # create range of years to download
    years = [str(year) for year in list(range(from_year, to_year + 1))]
    print("Try to download files from %s for " % (ftp_path) + str(years))

    # create Connection
    with ftplib.FTP(ip) as ftp:
        ftp.login()  # Login to ftp server
        ftp.cwd(ftp_path)  # Change path to historic data

        # Get list of avaiable files
        files = ftp.nlst()

        # Get list of wanted files
        files_w = [x for x in files if any(sx in x for sx in years)]

        # Download chain
        for i, file in zip(tqdm(range(len(files_w))), files_w):

            with open(os.path.join(download_path, file), 'wb') as data:
                ftp.retrbinary('RETR %s' % file, data.write)

    return print("Download completed")


def extract_evapo(path):
    """
    Arguments:
    """
    # Files in one Folder without checking for existant files (work stable!)
    from osgeo import gdal
    import tarfile
    from tqdm import tqdm
    import os
    import os.path
    from pathlib import Path
    from irrigation_detection import search_files

       # create folder in local space when not already there
    if not os.path.exists(path + 'data/'):
        os.makedirs(path + 'data/')
        # Search files with .tgz ending
    files = search_files(path, '.tgz')

    # Iterate through folders get tar archiv names loop through them and extract only with specified time
    tar_files = [x for x in files if "prop" not in x]

    # create list of lists with filenames for each month
    days = []
    for file in tar_files:
        with tarfile.open(file) as tar:
            days.append(tar.getnames())

        # set Driver for output  AAIGrid – Arc/Info ASCII Grid to GTiff – GeoTIFF File Format
        driver = gdal.GetDriverByName('Gtiff')

        # Loop through files and get tif for every elemtn inside the tarball
    for i, month, days in zip(tqdm(range(len(tar_files))), tar_files, days):
        for single_day in days:
            ds = gdal.Warp(path + 'data/' + single_day[:-4] + ".tif", "/vsitar/{%s}/%s" % (month, single_day), srcSRS='EPSG:31467', dstSRS='EPSG:4326')
            #src_ds = gdal.Open("/vsitar/{%s}/%s" % (month, single_day))
            #dst_ds = driver.CreateCopy(
                #path + 'data/' + single_day[:-4] + ".tif", src_ds, 0)
            #del src_ds
            #del dst_ds
            del ds
    return print('Job Done')

def open_evapo(path, real=True):
    """
    Arguments: path
    """

    import xarray as xr
    from irrigation_detection import search_files

    def get_dates(files, real=True):
        """
        get dates from attribute and convert to datetime64 object
        Arguments: files
        """
        import datetime
        import numpy as np

        dates = [datetime.datetime.strptime(
            x.split('_')[-1], '%Y%m%d.tif') for x in files]
        dates = [np.datetime64(x, 'D') for x in dates]

        return dates

    files = search_files(path, '.tif')

    # Load in and concatenate all individual GeoTIFFs
    evapo = xr.concat([xr.open_rasterio(i) for i in files],
                      dim=xr.Variable('time', get_dates(files)))
    if real:
        # Covert our xarray.DataArray into a xarray.Dataset
        evapo = evapo.to_dataset('band').rename(
            {1: 'evapo_r', 'x': 'lon', 'y': 'lat'})
    else:
        # Covert our xarray.DataArray into a xarray.Dataset
        evapo = evapo.to_dataset('band').rename(
            {1: 'evapo_p', 'x': 'lon', 'y': 'lat'})

    return evapo

def download_NDVI_max(user, passwd,pathFTP,download_path):
    """
    Arguments: user,passwd, pathFTP, download_path
    """
    import ftplib
    import os

    # create folder in local space when not already there
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # ftp server
    ip = 'ftp.copernicus.vgt.vito.be'

    # create Connection
    with ftplib.FTP('ftp.copernicus.vgt.vito.be') as ftp:
        ftp.login(user=user, passwd=passwd)  # Login to ftp server
        ftp.cwd(pathFTP)  # Change path to historic data
        folders = ftp.nlst()
        for folder in folders:
            ftp.cwd(folder)
            files = ftp.nlst()
            for file in files:
                print('Download %s' % file)
                with open(os.path.join(download_path, file), 'wb') as fobj:
                    ftp.retrbinary('RETR %s' % file, fobj.write)
            ftp.cwd('..')
        ftp.quit()

    return print('Job Done!')


def unzip_ndvi(pathZip, pathData):
    """
    Arguments: Path to zipfile parent folder
    """
    import os
    from irrigation_detection import search_files
    import zipfile

    # create folder in local space when not already there
    if not os.path.exists(pathData):
        os.makedirs(pathData)

    files = search_files(pathZip, '.zip')
    for file in files:
        with zipfile.ZipFile(file, 'r') as zipObject:
            zipObject.extractall(pathData)

    return print('Job done!')


def open_NDVI(path):
    """
    Arguments: path
    """

    import xarray as xr

    def get_dates(files):
        """
        get dates from attribute and convert to datetime64 object
        Arguments: files
        """
        import datetime
        import numpy as np

        dates = [datetime.datetime.strptime(
            x.split('\\')[1], '%Y%m%d') for x in files]
        dates = [np.datetime64(x, 'D') for x in dates]

        return dates

    files = search_files(path, '.tiff')

    NDVI = [x for x in files if 'NDVI-NDVI_' in x]
    NDVI_unc = [x for x in files if 'NDVI-unc' in x]
    NDVI_nobs = [x for x in files if 'NDVI-NOBS' in x]
    NDVI_Qflag = [x for x in files if 'NDVI-QFLAG' in x]
    NDVI_TGrid = [x for x in files if 'NDVI-TIME' in x]

    # Load in and concatenate all individual GeoTIFFs
    NDVI = xr.concat([xr.open_rasterio(i) for i in NDVI],
                     dim=xr.Variable('time', get_dates(NDVI)))
    # Covert our xarray.DataArray into a xarray.Dataset
    NDVI = NDVI.to_dataset('band').rename({1: 'NDVI', 'x': 'lon', 'y':'lat'})

    # Load in and concatenate all individual GeoTIFFs
    NDVI_unc = xr.concat([xr.open_rasterio(i) for i in NDVI_unc],
                         dim=xr.Variable('time', get_dates(NDVI_unc)))
    # Covert our xarray.DataArray into a xarray.Dataset
    NDVI_unc = NDVI_unc.to_dataset('band').rename({1: 'NDVI_unc', 'x': 'lon', 'y':'lat'})	

    # Load in and concatenate all individual GeoTIFFs
    NDVI_nobs = xr.concat([xr.open_rasterio(i) for i in NDVI_nobs],
                          dim=xr.Variable('time', get_dates(NDVI_nobs)))
    # Covert our xarray.DataArray into a xarray.Dataset
    NDVI_nobs = NDVI_nobs.to_dataset('band').rename({1: 'NDVI_nobs', 'x': 'lon', 'y':'lat'})	

    # Load in and concatenate all individual GeoTIFFs
    NDVI_Qflag = xr.concat([xr.open_rasterio(
        i) for i in NDVI_Qflag], dim=xr.Variable('time', get_dates(NDVI_Qflag)))
    # Covert our xarray.DataArray into a xarray.Dataset
    NDVI_Qflag = NDVI_Qflag.to_dataset('band').rename({1: 'NDVI_Qflag', 'x': 'lon', 'y':'lat'})	

    # Load in and concatenate all individual GeoTIFFs
    NDVI_TGrid = xr.concat([xr.open_rasterio(
        i) for i in NDVI_TGrid], dim=xr.Variable('time', get_dates(NDVI_TGrid)))
    # Covert our xarray.DataArray into a xarray.Dataset
    NDVI_TGrid = NDVI_TGrid.to_dataset('band').rename({1: 'NDVI_TGrid', 'x': 'lon', 'y':'lat'})	

       # merge all datasets
    ds_merge = xr.merge([NDVI, NDVI_unc, NDVI_nobs, NDVI_Qflag, NDVI_TGrid])

    return ds_merge

def get_s1_grd_mean(path, start, end, outname, with_ndvi, dateoffset):
    """
    Save a gejson to drive 
    Arguments: path to gejson featurecollection, start date, end date, outname, with_ndvi 'yes' or 'no', dateoffset (int) while finding correspnding ndvi values to s1 images
    """
    # Import modules.
    import ee

    try:
        # Initialize the library.
        ee.Initialize()
    except:
        # Trigger the authentication flow.
        ee.Authenticate()
        # Initialize the library.
        ee.Initialize()
    import geojson
    import geopandas as gpd
    import pandas as pd
    from glob import glob
    import os
    from datetime import datetime, timedelta
    import geemap.eefolium as geemap
    from tqdm import tqdm
    import geemap
    import time
    
    # Functions.
    # Calculate coverage in km²
    def get_area(image):
        # Count the non zero/null pixels in the image within the aoi
        actPixels = ee.Number(image.select('VV').reduceRegion(reducer= ee.Reducer.count(),scale= 10,geometry= fc_aoi.union().geometry(), maxPixels= 999999999).values().get(0))
        # calculate the perc of cover
        pcPix = actPixels.multiply(100).divide(1000000)
        return image.set('area', pcPix)
    
    #NDVI
    def add_ndvi(image):
        """
        Arguments: 
        """
        def maskS2clouds(image):
            qa = image.select('QA60')
            #Bits 10 and 11 are clouds and cirrus, respectively.
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            #Both flags should be set to zero, indicating clear conditions.
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)

        def NDVI(image):
            ndvi = image.normalizedDifference(['nir','red']).rename('NDVI') #(first − second) / (first + second)
            return image.addBands(ndvi)
        
        # Sentinel 2 image collection with corresponding named bands
        bandNamesOut_s2 = ['Aerosols','blue','green','red','red edge 1','red edge 2','red edge 3','nir','red edge 4','water vapor','cirrus','swir1','swir2','QA60']
        bandNamesS2 = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA60']
        s2_1c = ee.ImageCollection('COPERNICUS/S2').select(bandNamesS2,bandNamesOut_s2)
        s2_1c = s2_1c.filterDate(ee.Date(image.date().advance(-dateoffset,'days')), ee.Date(image.date().advance(+dateoffset,'days'))).filterBounds(image.geometry()).map(maskS2clouds).map(NDVI)
        ndvi = ee.Image(s2_1c.qualityMosaic('NDVI').select('NDVI'))

        return image.addBands(ndvi)
    
    def mask_by_ndvi(image):
        mask = image.select('NDVI').lte(0.6)
        return image.updateMask(mask)

    # Paths to initial polygon(s) and outdir for ts data.
    p_i = path
    p_o = os.path.dirname(path) + '/ts_data/'
    
    # create folder in local space when not already there.
    if not os.path.exists(p_o):
        os.makedirs(p_o)
        
    # Load aoi features from file.
    with open(p_i) as f:
        data = geojson.load(f)

    # Create GEE FeatureCollection from geojson file.
    fc_aoi = ee.FeatureCollection(data)
    area = fc_aoi.geometry().area().getInfo()
    
    # Sentinel 1 GRD image collection their dates and coverage over aoi
    ic_s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(fc_aoi).filterDate(ee.Date(start), ee.Date(end)).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    
    s1_dates = [datetime(1970, 1, 1) + timedelta(milliseconds=x) for x in ic_s1.aggregate_array("system:time_start").getInfo()]
    s1_dates = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in s1_dates]    
    s1_coverd = ic_s1.map(get_area).aggregate_array('area').getInfo()
    
    # Drop low coverage by metadata filter
    s1_valid = [x for x,y in zip(s1_dates,s1_coverd) if y > area*0.25]
    s1_valid_dates = ee.List(s1_valid).map(lambda x: ee.Date(x).millis())
    ic_s1 = ic_s1.filter(ee.Filter.inList("system:time_start", s1_valid_dates))

    print(ic_s1.size().getInfo(),'(%s)' %(len(s1_dates)), 'images between %s and %s' %(start,end), 'within %s km²' %(area/1000000),'\n') #s1_plot.aggregate_array("system:time_start").getInfo()
    
    if with_ndvi == 'yes':
        # Add ndvi band
        ic_s1 = ic_s1.map(add_ndvi)

        # Mask areas with ndvi > 0.6
        ic_s1 = ic_s1.map(mask_by_ndvi)
        
        # Map reducer function over imagecollection to get mean for multipolygon geometries
        fc_s1 = ic_s1.map(lambda x: x.reduceRegions(collection=fc_aoi ,reducer='mean', crs='EPSG:4326',scale=10)).flatten()
    else:
        # Map reducer function over imagecollection to get mean for multipolygon geometries
        fc_s1 = ic_s1.map(lambda x: x.reduceRegions(collection=fc_aoi ,reducer='mean', crs='EPSG:4326',scale=10)).flatten()
    
    # Export the FeatureCollection to a KML file.
    task = ee.batch.Export.table.toDrive(collection = fc_s1,description='vectorsToDrive',folder='idm_gee_export', fileFormat= 'GeoJSON', fileNamePrefix=outname)
    task.start()
    
    while task.active():
      print('Polling for task (id: {}).'.format(task.id))
      time.sleep(15)

    return print("finished")
def add_ndvi(image, dateoffset = 15):
        """
        Arguments: Filter S2 TOA Collection to roi, mask cloudy pixels, calculate NDVI values, Make mosaic from +- 15 days from s1 image
        """
        import ee
        def maskS2clouds(image):
            qa = image.select('QA60')
            #Bits 10 and 11 are clouds and cirrus, respectively.
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            #Both flags should be set to zero, indicating clear conditions.
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)

        def NDVI(image):
            ndvi = image.normalizedDifference(['nir','red']).rename('NDVI') #(first − second) / (first + second)
            return image.addBands(ndvi)
        
        # Sentinel 2 image collection with corresponding named bands
        bandNamesOut_s2 = ['Aerosols','blue','green','red','red edge 1','red edge 2','red edge 3','nir','red edge 4','water vapor','cirrus','swir1','swir2','QA60']
        bandNamesS2 = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA60']
        
        s2_1c = ee.ImageCollection('COPERNICUS/S2').select(bandNamesS2,bandNamesOut_s2)
        s2_1c = s2_1c.filterDate(ee.Date(image.date().advance(-dateoffset,'days')), ee.Date(image.date().advance(+dateoffset,'days'))).filterBounds(image.geometry()).map(maskS2clouds).map(NDVI)
        
        ndvi = ee.Image(s2_1c.qualityMosaic('NDVI').select('NDVI'))

        return image.addBands(ndvi)

def mask_by_landcover(image):
    image = image.select('Map')
    mask = image.eq(40).Or(image.eq(30))
    return image.updateMask(mask)

def get_s1_ts(lon, lat, ismn_idx, start, end, pol, mode, res, red, scale, crs, get_grid_scale, idx_name):
	"""
	Arguments: lon=longitude, lat = latitude, ismn_idx = ismn_id, start=start date, end= end date, pol=polarizaion(VV, VH, [VV, VH, HV,...], mode= [IW, SW], res=resolution[10,20,30], red=reducer['first, mean, median'], scale=scale for reducer, crs=crs for reducer, must be same as for lat/lon
	Get Sentinel 1 GRD Time Series for lat/lon with Metadata as GeopandasGeoDataFrame
	"""
	
	#import modules
	import ee
	import geopandas as gpd
	from datetime import datetime
	from shapely.geometry import Point
	
	# Authenticate Google Earth Engine
	try:
		ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com') # High Volume Endpoint
	except:
		# Trigger the authentication flow
		ee.Authenticate()
		# Initialize the library
		ee.Initialize()
	
	
	# Create Point Geometry (Longitude/Latitude)
	lon = lon
	lat = lat
	ismn_idx = ismn_idx #ISMN ID
	poi = ee.Geometry.Point([lon,lat]) #GEE Geometry Object
	poi_fc = ee.FeatureCollection(poi)
	
	if get_grid_scale == True:
		poi_fc = ee.FeatureCollection(poi_fc.geometry().buffer(5000))	 

	
	
	# Sentinel 1 Collection
	# Filter Collection by Location
	sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(poi_fc)
	# Filter by Date
	sentinel1 = sentinel1.filterDate(start, end)
	# Filter by Polarization
	sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', pol))
	# Filter by Swath Mode
	sentinel1 = sentinel1.filter(ee.Filter.eq('instrumentMode', mode))
	# Filter by Resolution
	sentinel1 = sentinel1.filter(ee.Filter.eq('resolution_meters', res))
	
	
	# Apply reducer
	ts = sentinel1.map(lambda x: x.reduceRegions(
		reducer = red,
		collection = poi_fc,
		scale = scale,
		crs = crs
		)).flatten().getInfo()
	
	
	# Aggregate data
	orbit = sentinel1.aggregate_array('orbitProperties_pass').getInfo()
	platform = sentinel1.aggregate_array('platform_number').getInfo()
	img_id = [x['id'] for x in ts['features']]
	ismn_id = [ismn_idx] * len(img_id)
	date = [datetime.strptime(x['id'].split('_')[4][:15], '%Y%m%dT%H%M%S') for x in ts['features']]
	geometry = [Point(x['geometry']['coordinates']) for x in ts['features']]
	VH = [x['properties']['VH'] for x in ts['features']]
	VV = [x['properties']['VV'] for x in ts['features']]
	angle = [x['properties']['angle'] for x in ts['features']]
	
	
	# Create GeopandasDataFrame
	gdf = gpd.GeoDataFrame({idx_name : ismn_id, 'date' : date, 'platform' : platform, 'orbit' : orbit, 'VV' : VV, 'VH' : VH, 'angle' : angle, 'img_id' : img_id, 'geometry' : geometry})
	
	print('S1 data collection succseed!')
	return gdf

def get_s2_ts(lon, lat, ismn_idx, start, end, red, scale, crs, idx_name):
    """
    Arguments:
    bla
    """
    
    # Import modules
    import ee
    from datetime import datetime
    from shapely.geometry import Point
    import geopandas as gpd
    import pandas as pd


    #Authenticate to Google Earth Engine
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com') #High Volume Endpoint
    except:
        ee.Authenticate()
        ee.Initialize()

    # Mask Clouds
    def maskS2clouds(image):
        qa = image.select('QA60')
        # Bits 10 and 11 are clouds and cirrus, respectively
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Set both to zero to have clear conditions
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        # update image by masking pixel with cloud or cirrus and copy properties
        return image.updateMask(mask).divide(10000).copyProperties(source=image)

    # Create Point Geometry (Longitude/Latitude)
    lon = lon
    lat = lat
    ismn_idx = ismn_idx #ISMN ID
    poi = ee.Geometry.Point([lon,lat]) #GEE Geometry Object
    poi_fc = ee.FeatureCollection(poi) #GEE FeatureCollection Object

    # Sentinel 2 Collection
    # Filter Collection by Location
    sentinel2 = ee.ImageCollection('COPERNICUS/S2').filterBounds(poi_fc)

    #Split daterange by half to overcome computation timeout
    end_half = (int(end.split('-')[0]) - int(start.split('-')[0])) / 2
    end_half = int(start[:4]) + end_half
    end_half = end.replace(end[:4],str(end_half)[:4])

    # Filter by Date Half 1/2
    sentinel2_1 = sentinel2.filterDate(start, end_half)
    sentinel2_1 = sentinel2_1.map(maskS2clouds)

    # Apply Reducer
    ts_half_1 = sentinel2_1.map(lambda x: x.reduceRegions(
        reducer='mean',
        collection=poi_fc,
        scale=scale,
        crs='EPSG:4326'
    )).flatten().getInfo()
    
    # Filter by Date Half 2/2
    sentinel2_2 = sentinel2.filterDate(end_half, end)
    sentinel2_2 = sentinel2_2.map(maskS2clouds)

    # Apply Reducer
    ts_half_2 = sentinel2_2.map(lambda x: x.reduceRegions(
        reducer='mean',
        collection=poi_fc,
        scale=scale,
        crs='EPSG:4326'
    )).flatten().getInfo()
    
    #merge both featurecollections
    ts_half_1['features'].extend(ts_half_2['features'])
    ts = ts_half_1
	
    #Aggregate data
    img_id = [x['id'] for x in ts['features']]
    ismn_id = [ismn_idx] * len(img_id)
    date = [datetime.strptime(x['id'][:15], '%Y%m%dT%H%M%S') for x in ts['features']]
    geometry = [Point(x['geometry']['coordinates']) for x in ts['features']]
    Aerosols = [x['properties']['B1'] for x in ts['features']]
    Blue = [x['properties']['B2'] for x in ts['features']]
    Green = [x['properties']['B3'] for x in ts['features']]
    Red = [x['properties']['B4'] for x in ts['features']]
    RedEdge1 = [x['properties']['B5'] for x in ts['features']]
    RedEdge2 = [x['properties']['B6'] for x in ts['features']]
    RedEdge3 = [x['properties']['B7'] for x in ts['features']]
    NIR = [x['properties']['B8'] for x in ts['features']]
    RedEdge4 = [x['properties']['B8A'] for x in ts['features']]
    WaterVapor = [x['properties']['B9'] for x in ts['features']]
    Cirrus = [x['properties']['B10'] for x in ts['features']]
    SWIR1 = [x['properties']['B11'] for x in ts['features']]
    SWIR2 = [x['properties']['B12'] for x in ts['features']]
    CloudMask = [x['properties']['QA60'] for x in ts['features']]
    
    
    #Create Geopandas DataFrame
    gdf = gpd.GeoDataFrame({idx_name : ismn_id, 'date' : date, 'Aerosols' : Aerosols ,'Blue' : Blue, 'Green' : Green, 'Red' : Red, 'RedEdge1' : RedEdge1, 'RedEdge2' : RedEdge2, 'RedEdge3' : RedEdge3, 'RedEdge4' : RedEdge4, 'NIR' : NIR, 'WaterVapor' : WaterVapor, 'Cirrus' : Cirrus, 'SWIR1' : SWIR1, 'SWIR2' : SWIR2, 'CloudMask' : CloudMask, 'img_id' : img_id, 'geometry' : geometry})

    
    # Tidy up
    #delete raws with nan values and get mean for multiple granularies with same date
    gdf = gdf.groupby(pd.Grouper(key='date',freq='d')).mean().dropna().reset_index()
    # In Previous step geometry and ismn_id was dropped because the mean of geometry is not possible and ismn_id was float so add here again
    gdf[idx_name] = [ismn_idx] * len(gdf)
    gdf['geometry'] = [geometry[0]] * len(gdf)
    
    # Add Indices
    gdf['NDVI'] = (gdf['NIR'] - gdf['Red']) / (gdf['NIR'] + gdf['Red'])
    print('S2 data collection sucseed!')
    return gdf
def get_ERA5_ts(lon, lat, ismn_idx, start, end, red, scale, crs, idx_name):
    """
    Arguments:
    bla
    """

    # Import modules
    import ee
    from datetime import datetime
    from shapely.geometry import Point
    import geopandas as gpd
    import pandas as pd

    # Authenticate to Google Earth Engine 
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com') #High volume endpoint
    except:
        ee.Authenticate()
        ee.Initialize()

    # Create Point Geometry (Longitude/Latitude)
    lon = lon
    lat = lat
    ismn_idx = ismn_idx  # ISMN ID
    poi = ee.Geometry.Point([lon, lat])  # GEE Geometry Object
    poi_fc = ee.FeatureCollection(poi)  # GEE FeatureCollection Object

    # ERA5 Collection
    # Filter Collection by Location
    era5 = ee.ImageCollection('ECMWF/ERA5/DAILY').filterBounds(poi_fc)
	
	#Split daterange by half to overcome computation timeout
    end_half = (int(end.split('-')[0]) - int(start.split('-')[0])) / 2
    end_half = int(start[:4]) + end_half
    end_half = end.replace(end[:4],str(end_half)[:4])
	
    # Filter by Date Half 1/2
    era5_1 = era5.filterDate(start, end_half)

    # Apply Reducer
    ts_half_1 = era5_1.map(lambda x: x.reduceRegions(
        reducer='first',
        collection=poi_fc,
        scale=scale,
        crs='EPSG:4326'
    )).flatten().getInfo()
    
    # Filter by Date Half 2/2
    era5_2 = era5.filterDate(end_half, end)

    # Apply Reducer
    ts_half_2 = era5_2.map(lambda x: x.reduceRegions(
        reducer='mean',
        collection=poi_fc,
        scale=scale,
        crs='EPSG:4326'
    )).flatten().getInfo()
    
    #merge both featurecollections
    ts_half_1['features'].extend(ts_half_2['features'])
    ts = ts_half_1
	
	# Aggregate data
    img_id = [x['id'] for x in ts['features']]
    ismn_id = [ismn_idx] * len(img_id)
    date = [datetime.strptime(x['id'].split('/')[-1].split('_')[0], '%Y%m%d')
            for x in ts['features']]
    geometry = [Point(x['geometry']['coordinates']) for x in ts['features']]
    dewpoint_2m_temperature = [x['properties']['dewpoint_2m_temperature'] for x in ts['features']]
    maximum_2m_air_temperature = [x['properties']['maximum_2m_air_temperature'] for x in ts['features']]
    mean_2m_air_temperature = [x['properties']['mean_2m_air_temperature'] for x in ts['features']]
    minimum_2m_air_temperature = [x['properties']['minimum_2m_air_temperature'] for x in ts['features']]
    surface_pressure = [x['properties']['surface_pressure'] for x in ts['features']]
    total_precipitation = [x['properties']['total_precipitation'] for x in ts['features']]
    u_component_of_wind_10m = [x['properties']['u_component_of_wind_10m'] for x in ts['features']]
    v_component_of_wind_10m = [x['properties']['v_component_of_wind_10m'] for x in ts['features']]

    # Create Geopandas DataFrame
    gdf = gpd.GeoDataFrame({idx_name: ismn_id, 
                            'date': date, 
                            'dewpoint_2m_temperature': dewpoint_2m_temperature, 
                            'maximum_2m_air_temperature': maximum_2m_air_temperature, 
                            'mean_2m_air_temperature': mean_2m_air_temperature, 
                            'minimum_2m_air_temperature': minimum_2m_air_temperature, 
                            'surface_pressure': surface_pressure, 
                            'total_precipitation': total_precipitation, 
                            'u_component_of_wind_10m': u_component_of_wind_10m,
                            'u_component_of_wind_10m': u_component_of_wind_10m, 
                            'v_component_of_wind_10m': v_component_of_wind_10m, 
                            'img_id': img_id, 
                            'geometry': geometry})

    # Tidy up
    print('ERA5 data collection sucseed!')
    return gdf
def get_ismn_data(filepath, variable, min_depth, max_depth, landcover, network, station):
	"""
	Arguments
	output: ts, ismn_loi, ismn_loi_unique
	"""
	
	# Import modules
	from ismn.interface import ISMN_Interface
	import pandas as pd
	
	# Read data
	# Path to data downloaded from ismn network in header+value composite
	file_name_in = filepath
	
	# Either a .zip file or one folder that contains all networks, here we read from .zip
	ismn_data = ISMN_Interface(filepath, parallel=True)
	
	if landcover is None:
	    landcover = list(ismn_data.landcover)
		
	if network is None:
	    network = ismn_data.list_networks()
	
	if station is None:
	    station = ismn_data.list_stations()

	# Select specific stations or networks
	ids = ismn_data.get_dataset_ids(variable = variable, min_depth = min_depth, max_depth = max_depth ,filter_meta_dict={'lc_2005': landcover, 'network' : network, 'station' : station} ) 
	
	# Read Station data for selected stations 
	ts = [ismn_data.read(x ,return_meta=True) for x in ids]
	
	
	# Extract lat/lon/id from data
	ismn_loi = list()
	for (data, meta), ismn_id in zip(ts, ids):
		ismn_loi.append([meta.longitude.values[0],meta.latitude.values[0], ismn_id])
	ismn_loi_unique = pd.DataFrame(ismn_loi).drop_duplicates(subset=[0,1]).values
	
	return ts, ismn_loi, ismn_loi_unique, ismn_data


def merge_s1_s2_era5(gdf_s1, gdf_s2, gdf_era5, driver, filepath, idx_name):
    """
    Arguments:
    """
    
    # Import modules
    import pandas as pd
    
    
    # Workaround for getting date from gdf_2 to calculate ndvi offsett in days
    gdf_s2['date_y'] = gdf_s2.date
    
    # Merge ndvi value to closest s1 date
    gdf = pd.merge_asof(gdf_s1.sort_values('date'), gdf_s2, suffixes=('', '_y'), on='date', direction='nearest', tolerance=pd.Timedelta('31d')).drop([idx_name + '_y','geometry_y' ], axis=1)
    gdf = pd.merge_asof(gdf.sort_values('date'), gdf_era5, suffixes=('', '_y'), on='date', direction='nearest', tolerance=pd.Timedelta('1d')).drop([idx_name + '_y', 'geometry_y', 'img_id_y'], axis=1)

    # Calculate time difference between s1 date and ndvi date
    gdf['s2_distance'] = gdf['date'] - gdf['date_y']
    gdf['s2_distance'] = gdf['s2_distance'].dt.days
    
    #Write GeodataFrame to file as geojson
    name = idx_name + '_' + str(gdf.iloc[0][idx_name]) + '_' + str(gdf.iloc[0].geometry.x) + '_' + str(gdf.iloc[0].geometry.y) + '.geojson'
    gdf.to_file(filename = filepath + name, driver = driver)
    
	
    return print('Write :', filepath + name, ' succesfully to disk')

def merge_sentinel_ismn(files, ismn_path, driver, out):
    """
    Arguments: 
    """

    # Import modules
    import geopandas as gpd
    import pandas as pd
    from ismn.interface import ISMN_Interface
    from glob import glob

	
    # Path to data downloaded from ismn network in header+value composite
    path_ismn_zip = ismn_path

    # Either a .zip file or one folder that contains all networks, here we read from .zip
    ismn_data = ISMN_Interface(path_ismn_zip, parallel=True)

    for file in files:
        # ismn id from filename
        idx = int(file.split('\\')[-1].split('_')[0])
        # metadata from ismn id
        metadata = ismn_data.read_metadata(idx)
        
        try:
            network = metadata.network.values[0]
            station = metadata.station.values[0]
            depth = metadata.variable.depth_from
            clay = metadata.clay_fraction.val
            sand = metadata.sand_fraction.val
            silt = metadata.silt_fraction.val
            oc = metadata.organic_carbon.val
            climate = metadata.climate_KG.values[0]
            elevation = metadata.elevation.values[0]
            instrument = metadata.instrument.val
        except:
            print(idx,' cant get all metadata')
            continue
        
        # Time-Series data for ismn id #try to get soil, air temp. precipitation as well
        ## In first case only implement stations with one sensor 
        gdf_sm = gpd.GeoDataFrame(ismn_data.read_ts(idx)).reset_index().rename({'date_time' : 'date'}, axis=1)
        gdf_sm['network'] = [network] * len(gdf_sm)
        gdf_sm['station'] = [station] * len(gdf_sm)
        gdf_sm['clay'] = [clay] * len(gdf_sm)
        gdf_sm['sand'] = [sand] * len(gdf_sm)
        gdf_sm['silt'] = [silt] * len(gdf_sm)
        gdf_sm['oc'] = [oc] * len(gdf_sm)
        gdf_sm['climate'] = [climate] * len(gdf_sm)
        gdf_sm['elevation'] = [elevation] * len(gdf_sm)
        gdf_sm['instrument'] = [instrument] * len(gdf_sm)
        gdf_sentinel = gpd.read_file(file)
        gdf_sentinel['date'] = gdf_sentinel.date.astype('datetime64[ns]')
        #gdf_ismn = gpd.GeoDataFrame(ismn_ts)
        try:
            gdf = pd.merge_asof(gdf_sentinel, gdf_sm, on='date', tolerance=pd.Timedelta("3h"), direction='nearest')
            gdf.to_file(out + file.split('\\')[-1], driver = driver)
        except:
            continue


    return print('Added soil moisture data to gdf')
	
def get_s1_s2_era5_df(longitudes, latitudes, ids, start, end, filepath, scale_s1, scale_s2, scale_era5, idx_name, get_grid_scale):
    """
    Arguments: 
    """
    
    from irrigation_detection import get_s1_ts
    from irrigation_detection import get_s2_ts
    from irrigation_detection import get_ERA5_ts
    from irrigation_detection import merge_s1_s2_era5
    from glob import glob
    failed_ids = list()
    for num, (lon,lat, idx) in enumerate(zip(longitudes, latitudes, ids)):
        print(f'Download Sentinel 1 & Sentinel 2 & ERA5 data for longitude: {lon}, latitude: {lat}, Item: {num}/{len(latitudes) - 1}')
        try:
            gdf_s1 = get_s1_ts(
                lon = lon, 
                lat = lat, 
                ismn_idx = int(idx), 
                start = start, 
                end = end, 
                pol = 'VV', 
                mode = 'IW', 
                res = 10, 
                red = 'mean',
                scale = scale_s1,
                crs = 'EPSG:4326',
                idx_name = idx_name,
                get_grid_scale = get_grid_scale
            )

            gdf_s2 = get_s2_ts(
                lon = lon, 
                lat = lat, 
                ismn_idx = int(idx), 
                start = start, 
                end = end, 
                red = 'mean',
                scale = scale_s2,
                crs = 'EPSG:4326',
                idx_name = idx_name
            )

            gdf_era5 = get_ERA5_ts(
                lon = lon, 
                lat = lat, 
                ismn_idx = int(idx), 
                start = start, 
                end = end, 
                red = 'first',
                scale = scale_era5,
                crs = 'EPSG:4326',
                idx_name = idx_name
            )

            merge_s1_s2_era5(
                gdf_s1 = gdf_s1,
                gdf_s2 = gdf_s2,
                gdf_era5 = gdf_era5,
                driver = 'GeoJSON',
                filepath = filepath,
                idx_name = idx_name
            )
        except Exception as e:
            print(e)
            print('Failed to download!')
            failed_ids.append(zip(longitudes, latitudes))
    
    return print(f'Finish downloads... failed to download {failed_ids}')
        
def get_s1_plot_grid_scale(path, start, end, outname, with_ndvi, ndvi_threshold, dateoffset, grid_to_dataframe, lon, lat):
    """
    Save a gejson to drive 
    Arguments: path to gejson featurecollection, start date, end date, outname, with_ndvi 'yes' or 'no', dateoffset (int) while finding correspnding ndvi values to s1 images
    """
    # Import modules.
    import ee

    try:
        # Initialize the library.
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except:
        # Trigger the authentication flow.
        ee.Authenticate()
        # Initialize the library.
        ee.Initialize()
    import geojson
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    from glob import glob
    import os
    from datetime import datetime, timedelta
    import geemap.eefolium as geemap
    from tqdm import tqdm
    import geemap
    import time
    
    # Functions.
    # Calculate coverage in km²
    def get_area(image):
        # Count the non zero/null pixels in the image within the aoi
        actPixels = ee.Number(image.select('VV').reduceRegion(reducer= ee.Reducer.count(),scale= 10,geometry= fc_aoi.union().geometry(), maxPixels= 999999999).values().get(0))
        # calculate the perc of cover
        pcPix = actPixels.multiply(100).divide(1000000)
        return image.set('area', pcPix)
    
    #NDVI
    def add_ndvi(image):
        """
        Arguments: 
        """
        def maskS2clouds(image):
            qa = image.select('QA60')
            #Bits 10 and 11 are clouds and cirrus, respectively.
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11
            #Both flags should be set to zero, indicating clear conditions.
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            return image.updateMask(mask).divide(10000)

        def NDVI(image):
            ndvi = image.normalizedDifference(['nir','red']).rename('NDVI') #(first − second) / (first + second)
            return image.addBands(ndvi)
        
        # Sentinel 2 image collection with corresponding named bands
        bandNamesOut_s2 = ['Aerosols','blue','green','red','red edge 1','red edge 2','red edge 3','nir','red edge 4','water vapor','cirrus','swir1','swir2','QA60']
        bandNamesS2 = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA60']
        s2_1c = ee.ImageCollection('COPERNICUS/S2').select(bandNamesS2,bandNamesOut_s2)
        s2_1c = s2_1c.filterDate(ee.Date(image.date().advance(-dateoffset,'days')), ee.Date(image.date().advance(+dateoffset,'days'))).filterBounds(image.geometry()).map(maskS2clouds).map(NDVI)
        ndvi = ee.Image(s2_1c.qualityMosaic('NDVI').select('NDVI'))

        return image.addBands(ndvi)

    def add_landcover(image):
        landcover = ee.Image(lc.select('Map'))
        return image.addBands(landcover)

    def mask_by_ndvi(image):
        mask = image.select('NDVI').lte(ndvi_threshold)
        return image.updateMask(mask)
    
    def mask_by_landcover(image):
        mask = image.select('Map').eq(40).Or(image.select('Map').eq(30))
        return image.updateMask(mask)
    
    

    if lon is not None:
        # Create Point Geometry (Longitude/Latitude)
        lon = lon
        lat = lat
        poi = ee.Geometry.Point([lon, lat])  # GEE Geometry Object
        fc_aoi = ee.FeatureCollection(poi)  # GEE FeatureCollection Object
        fcg_aoi = ee.FeatureCollection(fc_aoi.geometry().buffer(5000))
    else:
        # Paths to initial polygon(s) and outdir for ts data.
        p_i = path
        p_o = os.path.dirname(path) + '/ts_data/'

        # create folder in local space when not already there.
        if not os.path.exists(p_o):
            os.makedirs(p_o)
            
        # Load aoi features from file.
        with open(p_i) as f:
            data = geojson.load(f)

        # Create GEE FeatureCollection from geojson file.
        fc_aoi = ee.FeatureCollection(data)
    
        fcg_aoi = ee.FeatureCollection(fc_aoi.geometry().buffer(5000))
        
    area = fc_aoi.geometry().area().getInfo()
    areag = fcg_aoi.geometry().area().getInfo()

    # Sentinel 1 GRD image collection their dates and coverage over aoi
    ic_s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(fc_aoi).filterDate(ee.Date(start), ee.Date(end)).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    icg_s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(fcg_aoi).filterDate(ee.Date(start), ee.Date(end)).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

    s1_dates = [datetime(1970, 1, 1) + timedelta(milliseconds=x) for x in ic_s1.aggregate_array("system:time_start").getInfo()]
    s1_dates = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in s1_dates]    
    s1_coverd = ic_s1.map(get_area).aggregate_array('area').getInfo()
    
    s1g_dates = [datetime(1970, 1, 1) + timedelta(milliseconds=x) for x in ic_s1.aggregate_array("system:time_start").getInfo()]
    s1g_dates = [x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in s1g_dates]    
    s1g_coverd = icg_s1.map(get_area).aggregate_array('area').getInfo()
    
    # Drop low coverage by metadata filter
    s1_valid = [x for x,y in zip(s1_dates,s1_coverd) if y > area*0.25]
    s1_valid_dates = ee.List(s1_valid).map(lambda x: ee.Date(x).millis())
    #ic_s1 = ic_s1.filter(ee.Filter.inList("system:time_start", s1_valid_dates))

    # Drop low coverage by metadata filter
    s1g_valid = [x for x,y in zip(s1g_dates,s1g_coverd) if y > areag*0.25]
    s1g_valid_dates = ee.List(s1g_valid).map(lambda x: ee.Date(x).millis())
    #icg_s1 = icg_s1.filter(ee.Filter.inList("system:time_start", s1g_valid_dates))
    
    #print(ic_s1.size().getInfo(),'(%s)' %(len(s1_dates)), 'images between %s and %s' %(start,end), 'within %s km²' %(area/1000000),'\n') #s1_plot.aggregate_array("system:time_start").getInfo()
    print(icg_s1.size().getInfo(),'(%s)' %(len(s1g_dates)), 'images between %s and %s' %(start,end), 'within %s km²' %(areag/1000000),'\n') #s1_plot.aggregate_array("system:time_start").getInfo()
    
    #Landcover map
    lc = ee.ImageCollection("ESA/WorldCover/v100").first().clip(fcg_aoi.union().geometry())
    
    if with_ndvi == 'yes':
        # Add ndvi band
        ic_s1 = ic_s1.map(add_ndvi)

        # Mask areas with ndvi > 0.6
        #ic_s1 = ic_s1.map(mask_by_ndvi)
        
        # Map reducer function over imagecollection to get mean for multipolygon geometries
        fc_s1 = ic_s1.map(lambda x: x.reduceRegions(collection=fc_aoi ,reducer='mean', crs='EPSG:4326',scale=10)).flatten()
    else:
        # Map reducer function over imagecollection to get mean for multipolygon geometries
        fc_s1 = ic_s1.map(lambda x: x.reduceRegions(collection=fc_aoi ,reducer='mean', crs='EPSG:4326',scale=10)).flatten()
    
    if with_ndvi == 'yes':
        # Add ndvi band
        icg_s1 = icg_s1.map(add_ndvi)
        icg_s1 = icg_s1.map(add_landcover)
        # Mask areas with ndvi > 0.4 and landcover != 30,40
        icg_s1 = icg_s1.map(mask_by_ndvi)
        icg_s1 = icg_s1.map(mask_by_landcover)
        # Map reducer function over imagecollection to get mean for multipolygon geometries
        fcg_s1 = icg_s1.map(lambda x: x.reduceRegions(collection=fcg_aoi ,reducer='mean', crs='EPSG:4326',scale=10)).flatten()
    else:
        # Map reducer function over imagecollection to get mean for multipolygon geometries
        fcg_s1 = icg_s1.map(lambda x: x.reduceRegions(collection=fcg_aoi ,reducer='mean', crs='EPSG:4326',scale=10)).flatten()
    
    if grid_to_dataframe == True:
        ts = fcg_s1.getInfo()
        # Aggregate data
        #orbit = icg_s1.aggregate_array('orbitProperties_pass').getInfo()
        #platform = icg_s1.aggregate_array('platform_number').getInfo()
        img_id = [x['id'] for x in ts['features']]
        date = [datetime.strptime(x['id'].split('_')[4][:15], '%Y%m%dT%H%M%S') for x in ts['features']]
        geometry = [Point(x['geometry']['coordinates'][0][0]) for x in ts['features']]
        VH = [x['properties']['VH'] for x in ts['features']]
        VV = [x['properties']['VV'] for x in ts['features']]
        angle = [x['properties']['angle'] for x in ts['features']]
        ndvi = [x['properties']['NDVI'] for x in ts['features']]
        
        # Create GeopandasDataFrame
        gdf = gpd.GeoDataFrame({'date' : date, 'VV' : VV, 'VH' : VH, 'angle' : angle, 'NDVI' : ndvi, 'img_id' : img_id, 'geometry' : geometry}) #
        
        gdf.to_file(filename = path + outname + '_grid' + '.geojson', driver = 'GeoJSON')
        print('Saved' + path + outname + '_grid')
    else:
        # Export the FeatureCollection to a KML file.
        task1 = ee.batch.Export.table.toDrive(collection = fc_s1, description='vectorsToDrive',folder='idm_gee_export', fileFormat= 'GeoJSON', fileNamePrefix=outname + '_plot')
        task1.start()

        while task1.active():
          print('Polling for task (id: {}).'.format(task1.id))
          time.sleep(15)

        # Export the FeatureCollection to a KML file.
        task2 = ee.batch.Export.table.toDrive(collection = fcg_s1,description='vectorsGToDrive',folder='idm_gee_export', fileFormat= 'GeoJSON', fileNamePrefix=outname + '_grid')
        task2.start()

        while task2.active():
          print('Polling for task (id: {}).'.format(task2.id))
          time.sleep(15)

    return print('Finished')
if __name__ == "__main__":
    print("hea")