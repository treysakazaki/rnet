from dataclasses import dataclass, field

import numpy as np

try:
    from qgis.core import QgsFeature
except:
    pass

from rnet.core.geometry import edge_length
from rnet.core.layer import (
    Layer
    )
from rnet.utils import (
    polygon_geometry,
    categorized_renderer,
    fill_category
    )


@dataclass
class Buffer:
    
    coords: np.ndarray
    crs: int
    heading: float = None
    length: float = field(init=False)
    
    def __post_init__(self):
        self.length = edge_length(self.coords)
    
    def attributes(self):
        return [self.length, self.heading]
    
    def geometry(self):
        return polygon_geometry(self.coords)


class BufferDataSet:
    
    __slots__ = ['crs', 'data', 'layer']
    
    def __init__(self, crs, *data):
        self.crs = crs
        self.data = list(data)
    
    def generate(self, report=None):
        N = self.N
        for i, buffer in enumerate(self.data, 1):
            if report is not None:
                report(i/N*100)
            feat = QgsFeature()
            feat.setAttributes([i] + buffer.attributes())
            feat.setGeometry(buffer.geometry())
            yield feat
    
    @property
    def N(self):
        return len(self.data)
    
    def render(self, groupname='', index=0):
        self.layer = BufferLayer.create(self.crs)
        self.layer.populate(self.generate)
        self.layer.render()
        for i in range(self.N):
            self.layer.add_category(str(i+1))
        self.layer.add(groupname, index)


class BufferLayer(Layer):
    
    __slots__ = []
    
    fields = [('length', 'double'), ('heading', 'double')]
    
    def add_category(self, label, **kwargs):
        '''
        Adds a category to the renderer.
        
        Parameters
        ----------
        label : str
            Category label.
        **kwargs : dict, optional
            See keyword arguments for :func:`rnet.utils.symbols.fill_symbol`.
        '''
        self.vl.renderer().addCategory(fill_category(label, **kwargs))
    
    @classmethod
    def create(cls, crs, layername='buffers'):
        return super().create('polygon', crs, layername)

    @classmethod
    def renderer(cls):
        return categorized_renderer('fid')
