from dataclasses import dataclass
from typing import List
import numpy as np

from rnet.core.element import Element
from rnet.utils import line_geometry, single_line_renderer
from rnet.core.layer import Layer


@dataclass
class Path(Element):
    
    sequence: List[int]
    length: float
    coords: np.ndarray
    S: int = None
    G: int = None
    
    def __post_init__(self):
        self.S, self.G = self.sequence[0], self.sequence[-1]

    def geometry(self):
        return line_geometry(self.coords)


class PathLayer(Layer):
    
    @classmethod
    def create(cls, crs: int, layername: str = 'path') -> 'PathLayer':
        return super().create('linestring', crs, layername, Path.fields())

    @staticmethod
    def renderer(**kwargs):
        return single_line_renderer(**kwargs)
