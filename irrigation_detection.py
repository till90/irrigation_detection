def search_files(path, filetype):
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
if __name__ == "__main__":
    print("hea")