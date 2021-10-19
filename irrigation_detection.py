def search_files(path,filetype):
    """
    Arguments: path to tar files, file extension
    """
	
    import os
	
    path_list = [] #empty list where file pahts will be added
    for root, dirs, files in os.walk(path): #create and iterate through os.walk() generator 
        for file in files: #loop through founded files in each dir
            if file.endswith(filetype): #if any of these files endswith "filetype"...
                path_list.append(os.path.join(root,file)) #...then append this file to path_list while joining root dir
	
    return path_list #return list with full file paths 

def subset_ds(path,ds):
    """
    Return subsetted xarray dataset. Arguments path to shape)
    """
	
    import salem
	
    #load shapefile with salem
    aoi = salem.read_shapefile(path)
    #subset xarray dataset 
    ds_subset = ds.salem.subset(shape=aoi, margin=10)
    #mask out unrelevant data 
    ds_subset = ds_subset.salem.roi(shape=aoi)
    
    return ds_subset

def export_values(latitude,longitude,ds,name):
    """
    Find nearest cell and export it to csv. Arguments: lat, lon, dataset, filename
    """
    import pandas as pd
    
    #select data
    point_1 = ds.sel(lat=latitude, lon=longitude, method='nearest')
    #export with pandas
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
    
    #import modules
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
    
    #xarray dataset from SF 23:50 Product
    radolan_SF = xr.open_mfdataset(path, engine="radolan", decode_cf=False)
    
    #Konvert float64 to datetime64 
    radolan_SF['time'] = pd.to_datetime(radolan_SF.time.values, unit='s')
    
    #Replace coordinates with projected wgs84 lan lot predefined for radolan 900x900 cells
    radolan_grid_ll = wrl.georef.get_radolan_grid(900, 900, wgs84=True) 
    lat = radolan_grid_ll[:, 0, 1] 
    lon = radolan_grid_ll[0, :, 0]
    radolan_SF = radolan_SF.assign_coords({'lat': (lat), 'lon': lon})
    radolan_SF = radolan_SF.drop_vars(['x', 'y']).rename({'y': 'lat', 'x': 'lon'})
    
    return radolan_SF

if __name__ == "__main__":
	print("hea")