from abc import ABC
from dataclasses import dataclass
import os
from rnet.utils import abspath


@dataclass
class Source(ABC):
    '''
    Base class for representing a data source.
    
    Parameters
    ----------
    fp : str
        Path to data source.
    
    Example
    -------
    Subclass by defining the class variable ``ext``, representing the expected
    file extension::
        
        import pandas as pd
        
        class CsvSource(Source):
            
            ext = '.csv'
            
            def __post_init__(self):
                super().__post_init__()
                self.source = pd.read_csv(self.fp)
    '''
    
    fp: str
    
    def __post_init__(self):
        self.fp = abspath(self.fp)
        _, ext = os.path.splitext(self.fp)
        if ext != self.ext:
            raise TypeError(f'expected extension {self.ext!r}')
    
    def info(self) -> None:
        print(self)
