def group_s1_orbit_platform(df):
    """
    Return list of 4 dataframes seperated by orbit & platform values
    """

    #Import Modules
    import pandas as pd

    # Indexing data by platform and orbit 
    ms1 = df[(df.platform == 'A') & (df.orbit == 'DESCENDING')].copy()
    ms2 = df[(df.platform == 'B') & (df.orbit == 'DESCENDING')].copy()
    as1 = df[(df.platform == 'A') & (df.orbit == 'ASCENDING')].copy()
    as2 = df[(df.platform == 'B') & (df.orbit == 'ASCENDING')].copy()
    
    return [ms1, ms2, as1, as2]

def group_features(df, feature_label):
    """
    Return list of i dataframes seperated by feature label
    """

    #Import Modules
    import pandas as pd

    # List comprehesnion for splitting into seperated dfs according to feature label
    return [df[df[feature_label] == x] for x in df[feature_label]]

if __name__ == "__main__":
    print("hea")