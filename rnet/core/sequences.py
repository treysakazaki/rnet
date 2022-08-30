from itertools import count

import numpy as np

from PyQt5.QtGui import QColor

try:
    from qgis.core import (
        QgsFeature,
        QgsGeometry,
        QgsLineSymbol,
        QgsPointXY,
        QgsRuleBasedRenderer,
        QgsWkbTypes
        )
    from qgis.utils import iface
except:
    pass

from rnet.core.crs import CRS
from rnet.core.layer import Layer, LayerGroup
from rnet.core.geometry import buffer


class Path:
    
    __slots__ = ['name', 'graph_data', 'sequence', 'layer']
    counter = count(0)
    
    def __init__(self, graph_data, sequence):
        self.name = f'path_{next(self.counter):03d}'
        self.graph_data = graph_data
        self.sequence = sequence
    
    def __repr__(self):
        return f'<Path (S={self.S}, G={self.G})>'
    
    @property
    def crs(self):
        '''``crs`` property of the ``graph_data`` attribute.'''
        return self.graph_data.crs
    
    @property
    def coords(self):
        '''Array of shape (N, 2) containing node coordinates.'''
        return self.graph_data.ndata.df.loc[self.sequence].to_numpy()
    
    def densify(self, d):
        '''
        Returns coordinates of densified path. Vertices and nodes are
        retained.
        
        Parameters:
            d (float): Minimum distance between consecutive points.
        
        Returns:
            numpy.ndarray:
        '''
        vcoords = self.vertex_coords
        pseq = np.array([vcoords[0]])
        
        for k in range(1, len(vcoords)):
            pseq = np.append(pseq, [vcoords[k]], axis=0)
            dx, dy = pseq[-1] - pseq[-2]
            s = np.linalg.norm([dx, dy])  # distance to next vertex
            if s <= d:
                continue
            else:
                N = s // d  # number of points to add
                r = s / (N+1)  # interval between points
                t = np.arctan2(dy, dx)  # heading
                pseq = np.insert(pseq, -1, pseq[-2] + np.stack(
                    np.arange(1, N + 1) * r * np.vstack(
                        [np.cos(t), np.sin(t)]), axis=1), axis=0)
        '''
        dists = np.array([np.linalg.norm(vcoords[k+1] - vcoords[k])
                          for k in range(len(vcoords) - 1)])
        cdists = np.concatenate([[0], np.cumsum(dists)])
        indices = np.searchsorted(cdists, np.arange(d, cdists[-1], d))
        for k in range(len(indices)):
            i = indices[k]
            r = d * (k + 1) - cdists[i-1]
            dx, dy = vcoords[i] - vcoords[i-1]
            t = np.arctan2(dy, dx)
            pseq = np.append(
                pseq, [vcoords[i-1] + r*np.array([np.cos(t), np.sin(t)])],
                axis=0)
        '''
        return pseq
    
    def elevations(self, elevation_data, d=None):
        '''
        Returns three-dimensional points along densified path.
        
        Parameters:
            elevation_data (:class:`ElevationData`): Elevation data.
            d (int): Minimum distance between consecutive points.
        
        Returns:
            numpy.ndarray:
        '''
        if d is None:
            points = self.vertex_coords
        else:
            points = self.densify(d)
        
        if elevation_data.crs == self.crs:
            elevations = elevation_data.query(points)
        else:
            crs = CRS(self.crs)
            elevations = elevation_data.query(
                crs.transform(points, elevation_data.crs))
        
        return np.column_stack([points, list(elevations)])
    
    @property
    def G(self):
        '''Goal node.'''
        return self.seq[-1]
    
    def generate(self, report=lambda _: None):
        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromPolylineXY(
            [QgsPointXY(x, y) for (x, y) in self.vertex_coords]))
        f.setAttributes([self.name, self.length])
        report(1)
        yield f
    
    @property
    def length(self):
        '''Path length.'''
        return float(np.sum(
            self.graph_data.edge_lengths().loc[self.pairs]['length']))
    
    @property
    def N(self):
        '''Number of nodes in the sequence.'''
        return len(self.seq)
    
    @property
    def pairs(self):
        '''List of :math:`(i, j)` pairs.'''
        return list(map(
            tuple, np.sort(np.column_stack([self.seq[:-1], self.seq[1:]]))
            ))
    
    @property
    def S(self):
        '''Start node.'''
        return self.seq[0]
    
    @property
    def seq(self):
        '''Alias for the ``sequence`` attribute.'''
        return self.sequence
    
    @property
    def vertex_coords(self):
        '''Array containing vertex coordinates that define path geometry.'''
        return self.graph_data.vdata.df.loc[self.vertex_sequence].to_numpy()
    
    @property
    def vertex_sequence(self):
        '''List of vertex IDs that define path geometry.'''
        df = self.graph_data.edata.df.loc[self.pairs]
        vertex_sequence = [self.S]
        for (i, _), row in df.iterrows():
            vsequence = row[0]
            if i == vertex_sequence[-1]:
                pass
            else:
                vsequence = list(reversed(vsequence))
            vertex_sequence.extend(vsequence[1:])
        return vertex_sequence


class PathContainer:
    
    __slots__ = ['members', 'layer', '_crs']
    
    def __init__(self, crs, render=True):
        self.members = []
        self.layer = PathsLayer.create(crs, 'paths')
        self.layer.vl.setRenderer(PathsLayer.renderer())
        if render:
            self.render()
        self._crs = crs
    
    def add(self, path, color=None, width=1.0):
        self.members.append(path)
        self.layer.populate(path.generate, False)
        if color is None:
            color = tuple(np.random.choice(range(256), size=3))
        self.layer.add_rule(path, color, width)
    
    @property
    def crs(self):
        return self._crs
    
    @property
    def N(self):
        '''Member count.'''
        return len(self.members)
    
    def render(self, groupname='', index=0):
        self.layer.render(groupname, index)


class PathsLayer(Layer):
    
    def add_rule(self, P, color, width):
        symbol = QgsLineSymbol()
        symbol.setColor(QColor.fromRgb(*color))
        symbol.setWidth(width)
        self.vl.renderer().rootRule().appendChild(
            QgsRuleBasedRenderer.Rule(symbol,
                                      filterExp=f'"name" = {P.name!r}',
                                      label=P.name,
                                      description=P.name)
            )
        iface.layerTreeView().refreshLayerSymbology(self.id)
    
    @classmethod
    def create(cls, crs, layername):
        return super().create('linestring', crs, layername,
                              [('name', 'str'), ('length', 'double')])
    
    def render(self, groupname, index):
        group = LayerGroup(groupname)
        group.insert(self, index)
        self._location = (group, index)
    
    @staticmethod
    def renderer():
        return QgsRuleBasedRenderer(QgsRuleBasedRenderer.Rule(None))


class PointSequence:
    '''
    Class for representing a point sequence. A point sequence is a sequence of
    two-dimensional coordinate pairs.
    
    Parameters:
        sequence (numpy.ndarray): An array of shape (N, 2), where N is the
            number of points in the point sequence.
    '''
    
    __slots__ = ['sequence', 'name']
    counter = count(1)
    
    def __init__(self, sequence):
        N, M = sequence.shape
        assert N >= 2
        assert M == 2
        self.sequence = sequence
        self.name = f'pseq_{next(self.counter):03d}'
    
    def buffers(self, dist, *indices):
        return buffer(self.seq, dist, *indices)
    
    @classmethod
    def from_geometry(cls, geometry):
        '''
        Instantiates :class:`PointSequence` from a geometry.
        
        Parameters:
            geometry (qgis.core.QgsGeometry): A linestring geometry.
        
        Returns:
            :class:`PointSequence`:
                An instance of :class:`PointSequence`.
        '''
        assert geometry.type() == QgsWkbTypes.LineGeometry
        return cls(np.array(list(map(tuple, geometry.asPolyline()))))
    
    @property
    def geometry(self):
        '''qgis.core.QgsGeometry: The linestring geometry.'''
        return QgsGeometry.fromPolylineXY([QgsPointXY(*xy) for xy in self.seq])
    
    @property
    def length(self):
        '''float: Length of the point sequence, found by summing the length
        of all links.'''
        link_coords = self.link_coords
        dx = link_coords[:,2] - link_coords[:,0]
        dy = link_coords[:,3] - link_coords[:,1]
        return np.sum(np.linalg.norm(np.column_stack([dx, dy]), axis=1))
    
    @property
    def link_coords(self):
        '''numpy.ndarray: An array of shape :math:`(N-1, 4)`, where each row
        contains the starting and ending :math:`(x, y)` coordinates of the
        links in the point sequence.'''
        return np.column_stack([self.seq[:-1], self.seq[1:]])
    
    @property
    def N(self):
        '''int: Number of points in the point sequence.'''
        return len(self.seq)
    
    @property
    def seq(self):
        '''Alias for the :attr:`sequence` attribute.'''
        return self.sequence
