"""
Dataloader 
"""
import pandas as pd
from utils import Config
import numpy as np


class WellLogDataset:
    """Well logs dataset."""

    def __init__(self, root_dir, cnf: Config):
        csv_file = f"{root_dir}/formated_{cnf.data.year}.csv"
        self.wells = pd.read_csv(csv_file, index_col=['well_id'])
        self.T = cnf.input_length
        self.S = cnf.forecast_window
        self.year = cnf.data.year

    def __len__(self):
        return len(self.wells.groupby(by=["well_id"])) #type:ignore

    def __getitem__(self, index):
        well = self.wells.loc[int(str(index) + str(self.year))] #type:ignore
        start = np.random.randint(0, len(well) - self.T - self.S) # Pick a random starting location
        # start = 0

        well_id = str(well[start:start+1].index.values.item())# type:ignore

        # index_in = np.array([i for i in range(start, start+self.T)])
        # index_tar = np.array([i for i in range(start + self.T, start + self.T + self.S)])# type:ignore

        input_values = [
            "Gamma",
            # "Density Por",
            # "Resist",
            # "Shallow",
            # "SP",
            # "Density Corr",
            # "Caliper"
        ]

        _input = np.array(well[input_values][start : start + self.T].values)
        target = np.array(well[["Gamma"]][start+self.T: start + self.T + self.S].values)

        _input[:,0] = np.squeeze(np.expand_dims(_input[:,0], -1))
        target[:,0] = np.squeeze(np.expand_dims(target[:,0], -1))

        return _input, target

