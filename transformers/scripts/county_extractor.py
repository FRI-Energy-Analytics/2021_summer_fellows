import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import lasio
from tqdm import tqdm
from textwrap import wrap  # for making pretty well names
from multiprocessing import Pool, Queue
from functools import partial


def get_well(well_log):
    try:
        return well_log.well["WELL"]
    except:
        return well_log.well["WELL:1"]


def get_county(well_log):
    if "CNTY" in well_log.well:
        return well_log.well["CNTY"]
    if "CNTY." in well_log.well:
        return well_log.well["CNTY."]
    if "CNTY ." in well_log.well:
        return well_log.well["CNTY ."]
    if "API" in well_log.well:
        print(well_log.well["API"])
    return f"NA"


def add_log(file):
    try:
        return lasio.read(file)
    except:
        return None


if __name__ == "__main__":
    wells = []
    year = 2016
    counties = []
    well_logs = []

    for name in glob.glob(f"logs/{year}/*.las"):
        wells.append(name)

    pool = Pool()
    print(f"Queue 'em up")
    well_logs = list(pool.imap(add_log, wells))
    pool.close()
    pool.join()

    well_logs = filter(lambda x: x is not None, well_logs)  # Remove nulls

    for log in well_logs:
        counties.append(get_county(log))

    print(f"Contains {len(counties)}")

    for i, county in enumerate(counties):
        if type(county) != type(""):
            counties[i] = county.value

        counties[i] = counties[i].upper()

    # Extract all the counties into a dataframe
    kwargs = dict(County=counties)
    final_df = pd.DataFrame(kwargs)

    # There were a bunch of errors and typos in this county data
    # Time to fix the typos
    corrections = {
        "ANDERSON   SEC. 22   TWP. 20S   RGE. 20E": "ANDERSON",
        "STATON": "STANTON",
        "KEARNEY": "KEARNY",
        "ELI WIRELINE": "NA",
        "LGAN": "LOGAN",
        "SALINA": "SALINE",
        "HARPER CO": "HARPER",
        "HARPER CO.": "HARPER",
        "SUMMER": "SUMNER",
        "SEDGWICH": "SEDGWICK",
        "SEDOWICK": "SEDGWICK",
        "SEDGEWICK": "SEDGWICK",
        "LORRAINE": "ELLSWORTH",  # Lorrained is a city in Ellsworth CO.
        "HASKEL": "HASKELL",
        "DECTAUR": "DECATUR",
        "TRGO": "TREGO",
        "ELLS": "ELLIS",
        "NESS CO.": "NESS",
        "OSBOURNE": "OSBORNE",
        "": "NA",
        "HODGMAN": "HODGEMAN",
        "USA": "NA",
        "KANSAS": "NA",
        "RUSSEL": "RUSSELL",
        "PRATT COUNTY": "PRATT",
        "WITCHITA": "WICHITA",
        "RUCH": "RUSH",
        "RAWLINGS": "RAWLINS",
        "RENO CO": "RENO",
        "RENO CO.": "RENO",
    }
    # Apply colrrections
    for key, value in tqdm(corrections.items(), desc="Corrections"):
        final_df.loc[final_df["County"] == key] = value

    freq_count = final_df["County"].value_counts()
    freq_df = pd.DataFrame(
        {"County": freq_count.keys(), "Frequency": freq_count.values}
    )
    freq_df = freq_df.sort_values(by="Frequency", ascending=False)
    freq_df = freq_df.reset_index(drop=True)
    print(freq_df.head())  # Lets see the Frequencies in order

    freq_df["Percent"] = freq_df["Frequency"] / freq_df["Frequency"].sum() * 100
    print(f"Number of NA's {freq_df[freq_df['County'] == 'NA']}")

    fig = plt.figure(figsize=(6, 17))
    plt.yticks(range(len(freq_df)), freq_df["County"])
    # Lets plot the frequency
    plt.barh(range(len(freq_df)), freq_df["Frequency"])

    import geopandas as gpd

    kansas_map = gpd.read_file("kansas.zip")

    # To merge the county data with the kansas map we must use fips numbers
    fpis = pd.read_csv("fips.csv")
    freq_df["COUNTYFP"] = 0

    for index, county in freq_df["County"].iteritems():
        name = county.capitalize() + " County"
        if name == "Mcpherson County":  # Only county with speical capitalization
            name = "McPherson County"
        q1 = fpis[fpis["name"] == name]
        try:
            freq_df.loc[index, "COUNTYFP"] = str(
                q1[q1["state"] == "KS"]["fips"].iloc[0]
            )[2:]
        except Exception as e:
            # Print out any thing that is not marked as existing
            print("None - " + county)

    density_map = kansas_map
    density_map = density_map.merge(freq_df, on="COUNTYFP", how="left")

    density_map = density_map.fillna(0)
    plt.show()
    density_map.plot(column="Frequency", cmap="inferno", legend=True)
    plt.title(f"Kansas Well County Distribution - {year}")
    plt.savefig(f"maps/kansas_map {year}.png")
    plt.show()
    plt.close(fig)
