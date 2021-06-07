import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import lasio
from tqdm import tqdm
from textwrap import wrap  # for making pretty well names
from multiprocessing import Pool, Queue
from functools import partial

def get_depth(well_log):
    try:
        return well_log["DEPT"]
    except:
        return well_log["DEPTH"]

def get_gamma(well_log):
    try:
        return well_log["GR"]
    except:
        try:
            return well_log["GRGC"]
        except:
            try:
                return well_log["GAMMA"]
            except:
                return well_log["GAMMA:1"]

year = 2019
counties = []
well_logs = []
errors = 0

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
    return f"NA"

def add_log(file):
    global errors
    try:
        return lasio.read(file)
    except Exception as e:
        print(f"{e} Error number: {errors} on file {file}")
        errors += 1
        return None

wells = []
for name in glob.glob(f"logs/{year}/*.las"):
    wells.append(name)

pool = Pool()
print(f"Queue 'em up")
well_logs = list(pool.imap(add_log, wells));
pool.close()
pool.join()
print(f"Finished reading all well values with {errors} errors.")

well_logs = filter(lambda x: x is not None, well_logs)

for log in well_logs:
    counties.append(get_county(log))

print(f"Contains {len(counties)}")

for i, county in tqdm(enumerate(counties), desc="Reading countries in"):
    if type(county) != type(""):
        counties[i] = county.value
    
    counties[i] = counties[i].upper()

kwargs = dict(County=counties)

final_df = pd.DataFrame(kwargs)

corrections = {
    "ANDERSON   SEC. 22   TWP. 20S   RGE. 20E" : "ANDERSON",
    "STATON": "STANTON",
    "KEARNEY": "KEARNY",
    "ELI WIRELINE": "NA",
    "LGAN": "LOGAN",
    "SALINA": "SALINE",
    "HARPER CO": "HARPER",
    "HARPER CO.": "HARPER",
    "SUMMER": "SUMNER",
    "SEDOWICK": "SEDGWICK",
    "ELLS": "ELLIS",
    "NESS CO.": "NESS",
    '': "NA",
    "HODGMAN": "HODGEMAN",
    "USA" : "NA",
    "KANSAS" : "NA",
    "RUSSEL" : "RUSSELL",
    "PRATT COUNTY" : "PRATT",
    "WITCHITA" : "WICHITA",
    "RUCH" : "RUSH",
    "RAWLINGS" : "RAWLINS",
}
for key, value in tqdm(corrections.items(), desc="Corrections"):
    final_df.loc[final_df["County"] == key] = value

freq_count = final_df["County"].value_counts()
freq_df = pd.DataFrame({"County": freq_count.keys(), "Frequency": freq_count.values})
freq_df = freq_df.sort_values(by="Frequency", ascending=False)
freq_df = freq_df.reset_index(drop=True)

freq_df["Percent"] = freq_df['Frequency'] /freq_df["Frequency"].sum() * 100
print(f"Number of NA's {freq_df[freq_df['County'] == 'NA']}")

fig = plt.figure(figsize=(6,17))
plt.yticks(range(len(freq_df)), freq_df["County"])
plt.barh(range(len(freq_df)), freq_df["Percent"])

import geopandas as gpd
kansas_map = gpd.read_file("kansas.zip")
kansas_map.plot()


fpis = pd.read_csv("fips.csv")

freq_df["COUNTYFP"] = 0

for index, county in freq_df["County"].iteritems():
    name = county.capitalize() + " County"
    if name == "Mcpherson County":
        name = "McPherson County"
    q1 = fpis[fpis["name"] == name]
    try:
        freq_df.loc[index, "COUNTYFP"] = str(q1[q1["state"] == "KS"]["fips"].iloc[0])[2:]
    except Exception as e:
        print("None - " + county)

print(freq_df.head())
density_map = kansas_map
density_map = density_map.merge(freq_df, on = "COUNTYFP", how = "left")

density_map = density_map.fillna(0)
density_map.plot(column="Frequency", cmap = 'inferno', legend=True)
plt.show()
plt.title(f"Kansas Well County Distribution - {year}")
plt.savefig(f"kansas_map {year}.png")
plt.close(fig)

