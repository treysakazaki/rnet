from abc import ABC, abstractmethod
import pandas as pd
from rnet.core.layer import Layer


class Dataset(ABC):
    '''
    Base class for datasets.
    
    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Frame summarizing dataset.
    layer : :class:`Layer`, optional
        Layer for visualizing dataset.
    '''
    
    def __init__(self, df: pd.DataFrame, layer: Layer = None):
        self.df = df
        self.layer = layer
    
    def __contains__(self, id_) -> bool:
        return id_ in self.df.index
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __iter__(self):
        self._rows = self.df.iterrows()
    
    def __next__(self):
        return next(self._rows)
    
    @classmethod
    def _from_layer(self, layer: Layer):
        pass
    
    @classmethod
    def from_gpkg(self, path_to_gpkg: str, layername: str):
        pass
    
    @classmethod
    def from_ml(self, layername: str):
        pass
    
    @abstractmethod
    def generate(self):
        pass
    
    @abstractmethod
    def info(self) -> None:
        pass
    
    @abstractmethod
    def render(self):
        pass


def PointDataset(Dataset):
    '''
    Base class for datasets containing point data.
    '''
    
    def __init__(self, df: pd.DataFrame, layer: Layer = None):
        super().__init__(df, layer)
    
    @property
    def crs(self):
        '''
        int: EPSG code of CRS in which coordinates are represented.
        '''
        return self.df.attrs['crs']
    
    @property
    def dims(self):
        '''
        int: Number of coordinate dimensions, either 2 or 3.
        '''
        return len(self.df.columns)
