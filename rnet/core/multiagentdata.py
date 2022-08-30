import pandas as pd

from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsPointXY
    )

from rnet.core.layer import Layer
from rnet.core.crs import CRS
from rnet.utils import categorized_renderer, point_category


class MultiAgentData:
    
    __slots__ = ['data', '_crs', 'layer']
    
    def __init__(self):
        self.data = []
    
    def add(self, agent_data):
        if len(self.data) == 0:
            self._crs = CRS(agent_data.crs)
        else:
            assert agent_data.crs == self.crs
        self.data.append(agent_data)
    
    @property
    def crs(self):
        return self._crs.epsg
    
    @property
    def df(self):
        return self.out()
    
    def generate(self, report=None):
        df = self.df.reset_index()
        N = len(df)
        for i, row in df.iterrows():
            if report is not None:
                report(i/N*100)
            row = list(row)
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*row[2:4])))
            f.setAttributes([i+1] + row)
            yield f
    
    def render(self, groupname='', index=0):
        self.layer = MultiAgentLayer.create(self.crs)
        self.layer.populate(self.generate)
        self.layer.render()
        for label in set(self.df['id']):
            self.layer.add_category(label)
        self.layer.add(groupname, index)
    
    def to_gpkg(self):
        #TODO
        pass
    
    def out(self, include_headings=True):
        dfs = []
        for data in self.data:
            df = pd.concat([data.df.copy(), data.headings()], axis=1)
            df.insert(0, 'id', data.name)
            dfs.append(df)
        return pd.concat(dfs)


class MultiAgentLayer(Layer):
    
    __slots__ = []
    
    fields = [('timestamp', 'time'), ('id', 'str'), ('x', 'double'),
              ('y', 'double'), ('heading', 'double')]
    
    def add_category(self, label, **kwargs):
        '''
        Adds a new category to the renderer.
        
        Parameters
        ----------
        label : str
            The label for the new category.
        **kwargs : dict, optional
            Keyword arguments passed to :func:`rnet.utils.symbols.marker_symbol`.
        '''
        kwargs.setdefault('shape', 'half_square')
        kwargs.setdefault('anglename', 'heading')
        self.vl.renderer().addCategory(point_category(label, **kwargs))
    
    @classmethod
    def create(cls, crs, layername='agents'):
        return super().create('point', crs, layername)
    
    @classmethod
    def renderer(cls):
        return categorized_renderer('id')
    