import os

import numpy as np
import pandas as pd

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor

from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPointXY,
    QgsProperty,
    QgsSimpleMarkerSymbolLayerBase,
    QgsSingleSymbolRenderer
    )

from rnet.core.crs import CRS
from rnet.core.layer import Layer
from rnet.utils import abspath


class AgentData:
    '''
    Class for representing agent data.
    
    
    
    Parameters:
        path_to_csv (str): Path to CSV file.
        crs (:obj:`int`, optional): EPSG code of CRS in which agent coordinates
            are represented. Default: 4326.
        name (:obj:`str`, optional): Agent name. If None, then the CSV file
            name is used. Default: None.
    
    Keyword arguments:
        columns (:obj:`List[str]`, optional): List of column names that
            contain timestamps, `x`/-, and `y`/-coordinates. Default:
            ['timestamp', 'x', 'y'].
        period (:obj:`str`, optional): `'minute'`, `'second'`, or
            `'microsecond'`. Default: `'second'`.
        include_dates (:obj:`bool`, optional): Whether to include dates in the
            index. Default: False.
        xshift (:obj:`float`, optional): Shift in the `x`-direction. Default:
            0.0.
        yshift (:obj:`float`, optional): Shift in the `y`-direction. Default:
            0.0.
    
    Attributes:
        df
        crs
        name
    '''
    
    __slots__ = ['df', '_crs', 'name', 'layer']
    
    def __init__(self, path_to_csv, crs=4326, name=None, *,
                 columns=['timestamp', 'x', 'y'], period='second',
                 include_dates=False, xshift=0.0, yshift=0.0):
        # Import data
        path_to_csv = abspath(path_to_csv)
        df = pd.read_csv(path_to_csv, index_col=columns[0], usecols=columns)
        df = df[columns[1:3]]
        df = df.rename(columns={columns[1]: 'x', columns[2]: 'y'})
        self._crs = CRS(crs)
        if name is None:
            name = os.path.splitext(os.path.basename(path_to_csv))[0]
        self.name = name
        # Filter by time
        if period == 'second':
            mask = ['.' not in timestamp for timestamp in df.index]
        df = df.loc[mask]
        df.index.name = 'timestamp'
        # Fix offset
        df['x'] += xshift
        df['y'] += yshift
        self.df = df
    
    @property
    def crs(self):
        '''int: EPSG code of CRS.'''
        return self._crs.epsg
    
    def generate(self, report=None):
        df = pd.concat([self.df, self.headings()], axis=1).reset_index()
        N = self.N
        for i, row in df.iterrows():
            if report is not None:
                report(i/N*100)
            row = list(row)
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*row[1:3])))
            feat.setAttributes([row[3]])
            yield feat
    
    def headings(self):
        # TODO: If coordinates do not change, maintain heading
        coords = self.df.to_numpy()
        coords = np.column_stack([coords[:-1], coords[1:]])
        dx, dy = coords[:,2] - coords[:,0], coords[:,3] - coords[:,1]
        headings = np.mod(90 - np.degrees(np.arctan2(dy, dx)), 360)
        headings = np.append(headings, headings[-1])
        return pd.DataFrame(headings, index=self.df.index, columns=['heading'])
    
    @property
    def N(self):
        '''int: Number of rows.'''
        return len(self.df)
    
    def render(self, groupname='', index=0, **kwargs):
        self.layer = AgentLayer.create(self.crs, self.name)
        self.layer.populate(self.generate)
        self.layer.render(**kwargs)
        self.layer.add(groupname, index)
    
    def to_gpkg(self, gpkg, layername):
        pass
    
    def transform(self, dst):
        if dst == self.crs:
            return
        else:
            coords = self._crs.transform(self.df.to_numpy(), dst)
            self.df['x'] = coords[:,0]
            self.df['y'] = coords[:,1]
            self._crs = CRS(dst)


class AgentLayer(Layer):
    
    __slots__ = []
    
    fields = [('heading', 'double')]
    
    @classmethod
    def create(cls, crs, layername='agents'):
        return super().create('point', crs, layername, cls.fields)

    @classmethod
    def renderer(cls, **kwargs):
        '''
        Returns renderer for the vertex layer.
        
        Keyword arguments:
            **kwargs: Keyword arguments which are passed to the :meth:`symbol`
                method.
        
        Returns:
            qgis.core.QgsSingleSymbolRenderer:
        '''
        return QgsSingleSymbolRenderer(cls.symbol(**kwargs))
    
    @staticmethod
    def symbol(*, color=None, size=1.6):
        '''
        Returns the default symbol used for rendering agents.
        
        Keyword arguments:
            color (Tuple[int, int, int]): RGB color definition. If None, then
                a random color is assigned. Default: None.
            size (float): Marker size. Default: 1.6.
        
        Returns:
            qgis.core.QgsMarkerSymbol:
        '''
        if color is None:
            color = tuple(np.random.choice(range(256), size=3))
        symbol = QgsMarkerSymbol()
        symbol.setColor(QColor.fromRgb(*color))
        symbol.setSize(size)
        symbol.setDataDefinedAngle(QgsProperty.fromExpression('"heading"'))
        symbol.symbolLayer(0).setShape(QgsSimpleMarkerSymbolLayerBase.HalfSquare)
        symbol.symbolLayer(0).setOffset(QPointF(size/4, 0))
        return symbol

