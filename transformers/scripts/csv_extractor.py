""" CSV extractpr
TODO describe
"""

import glob
import pandas as pd
import lasio
from tqdm import tqdm
from multiprocessing.pool import Pool


def get_depth(well_log):
    """ get well depth
    TODO describe
    """
    if well_log is None:
        return None
    if "DEPT" in well_log.keys():
        return well_log["DEPT"]
    if "DEPTH" in well_log.keys():
        return well_log["DEPTH"]
    return None


def get_gamma(well_log):
    """ get well gamma
    TODO describe
    """
    if well_log is None:
        return None
    if "GR" in well_log.keys():
        return well_log["GR"]
    if "GR." in well_log.keys():
        return well_log["GR."]
    if "GRX" in well_log.keys():
        return well_log["GRX"]
    if "GRGC" in well_log.keys():
        return well_log["GRGC"]
    if "GAMMA" in well_log.keys():
        return well_log["GAMMA"]
    if "GAMMA:1" in well_log.keys():
        return well_log["GAMMA:1"]
    if "GR.GAPI" in well_log.keys():
        return well_log["GR.GAPI"]
    if "GREV" in well_log.keys():
        return well_log["GR."]
    print(well_log.keys())
    return None


def get_well(well_log):
    """ get well
    TODO describe
    """
    try:
        return well_log.well["WELL"]
    except:
        return well_log.well["WELL:1"]


def get_county(well_log):
    """ get well county
    TODO describe
    """
    if well_log is None:
        return "NA"
    if "CNTY" in well_log.well:
        return well_log.well["CNTY"].value
    if "CNTY." in well_log.well:
        return well_log.well["CNTY."].value
    if "CNTY ." in well_log.well:
        return well_log.well["CNTY ."].value
    return f"NA"


def add_log(file):
    """ add well log
    TODO describe
    """
    try:
        return lasio.read(file)
    except:
        return None


def extract_all(well_log, uid) -> pd.DataFrame:
    """ extract all data
    TODO describe
    """
    depth = get_depth(well_log)
    gamma = get_gamma(well_log)
    if depth is None or gamma is None:
        return (
            pd.DataFrame()
        )  # If no time series or gamma is avaliable, we can't use it
    index_df = pd.DataFrame(dict(depth=depth, well_id=uid))
    index = pd.MultiIndex.from_frame(index_df)
    county = get_county(well_log)

    return pd.DataFrame(dict(gamma=gamma, County=county), index)


if __name__ == "__main__":
    wells = []
    year = 2015
    counties = []
    well_logs = []

    for name in glob.glob(f"logs/{year}/*.las"):
        wells.append(name)

    pool = Pool()
    print(f"Queue 'em up")
    try:
        well_logs = list(pool.imap(add_log, wells))
    finally:
        pool.close()
        pool.join()

    print("Flushed queue")

    final_df = pd.DataFrame()
    for uid, well_log in enumerate(tqdm(well_logs, desc="Merging")):
        final_df = pd.concat(
            [extract_all(well_log, uid), final_df]
        )  # Slow but saves memory

    corrections = {
        "ELI WIRELINE": "NA",
        "LGAN": "LOGAN",
        "SALINA": "SALINE",
        "HARPER CO": "HARPER",
        "HARPER CO.": "HARPER",
        "SUMMER": "SUMNER",
        "SEDOWICK": "SEDGWICK",
        "ELLS": "ELLIS",
        "NESS CO.": "NESS",
        "": "NA",
        "HODGMAN": "HODGEMAN",
        "USA": "NA",
        "KANSAS": "NA",
        "RUSSEL": "RUSSELL",
        "PRATT COUNTY": "PRATT",
        "WITCHITA": "WICHITA",
        "RUCH": "RUSH",
        "RAWLINGS": "RAWLINS",
    }

    # Spelling corrections
    for key, value in corrections.items():
        final_df.loc[final_df["County"] == key] = value

    final_df.to_csv(f"export_csv/{year}.csv")
    print(f"Exported Data to export_csv/{year}.csv")
