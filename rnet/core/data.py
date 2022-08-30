from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from rnet.core.layer import Layer


@dataclass
class Data(ABC):
    '''
    Base class for representing datasets.
    '''
    
    df: pd.DataFrame
    layer: Layer = None
    
    def __contains__(self, id_) -> bool:
        return id_ in self.df.index
    
    def __len__(self) -> int:
        return len(self.df)
    
    @abstractmethod
    def generate(self):
        pass
    
    @abstractmethod
    def info(self) -> None:
        pass
    
    @abstractmethod
    def render(self):
        pass
