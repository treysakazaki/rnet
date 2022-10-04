from enum import Enum
import pandas as pd


class PointTypes(Enum):
    
    VERTEX = 0
    NODE = 1
    BORDERNODE = 2
    PLACE = 3


def merge_point_dfs(*dfs, assume_unique: bool = True) -> pd.DataFrame:
    df = pd.concat(dfs)
    df.reset_index()
    return df
