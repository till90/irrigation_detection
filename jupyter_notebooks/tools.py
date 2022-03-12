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


def add_group_label(df, col_name):
    """Adds group label according of orbit and platform

    Args:
        df (_type_): _description_
        col_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    import pandas as pd
    group = list()

    for row in df.itertuples():
        
        
        if row.platform == 'A' and row.orbit == 'DESCENDING':
            group.append(1)
        elif row.platform == 'B' and row.orbit == 'DESCENDING':
            group.append(2)
        elif row.platform == 'A' and row.orbit == 'ASCENDING':
            group.append(3)
        else:# row.platform == 'B' and row.orbit == 'ASCENDING':
            group.append(4)
        
    df[col_name] = group
    return df


if __name__ == "__main__":
    print("hea")
    