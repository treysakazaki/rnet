import os
from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    from qgis.core import QgsTask
    from osgeo import ogr
except:
    pass

from rnet.core.vertexdata import VertexData
from rnet.core.linkdata import LinkData
from rnet.core.source import Source


__all__ = ['OsmSource']


class OsmSource(Source):
    '''
    Class for representing OSM data.
    
    Parameters
    ----------
    fp : str
        Path to OSM file.
    
    Keyword arguments
    -----------------
    include : :obj:`str` or :obj:`List[str]`
        A tag, list of tags, or 'all', indicating the roads from which
        vertices and links are extracted.
    
    References
    ----------
    https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html
    
    Example
    -------
    Instantiate from a single OSM file:
        
        >>> data = rn.OsmData(<path/to/osm>)
    '''
    
    ext = '.osm'
    
    HIERARCHY = {'living_street' : 0,
                 'residential'   : 1,
                 'unclassified'  : 2,
                 'tertiary_link' : 3,
                 'tertiary'      : 4,
                 'secondary_link': 5,
                 'secondary'     : 6,
                 'primary_link'  : 7,
                 'primary'       : 8,
                 'trunk_link'    : 9,
                 'trunk'         : 10,
                 'motorway_link' : 11,
                 'motorway'      : 12}
    
    def __post_init__(self):
        super().__post_init__()
        
        #create_and_queue(OsmLoadTask, self, self.fp)
        
        # Read features from OSM source file
        driver = ogr.GetDriverByName('OSM')
        self.source = driver.Open(self.fp, 0)
        layer = self.source.GetLayer('lines')
        _ids = []
        _roads = []
        _tags = []
        for feat in layer:
            _ids.append(feat.GetFID())
            _roads.append(feat.GetGeometryRef().GetPoints())
            _tags.append(feat.GetField('highway'))
        self._ids = np.array(_ids)
        self._roads = np.array(_roads)
        self._tags = np.array(_tags)
        
        # Set roads to include
        self.include = 'all'
    
    def ids(self):
        '''
        Returns an array of shape (N,) containing the OSM IDs of features in
        the OSM layer whose tags are included in the :attr:`include` property.
        
        Returns
        -------
        :class:`numpy.ndarray`, shape (N,)
            Array of OSM IDs.
        '''
        return self._ids[self.indices()]
    
    @property
    def include(self):
        '''
        str or List[str]: Tag(s) of the features to include when extracting
        map elements. If 'all', then all features are included.
        '''
        return self._include
    
    @include.setter
    def include(self, include):
        if include == 'all':
            self._include = list(self.HIERARCHY)
        elif type(include) is str:
            self._include = [include]
        elif type(include) is list:
            self._include = include
        else:
            raise TypeError
        self._dirty = [True, True]
    
    def indices(self):
        '''
        Returns indices of features in the OSM layer whose tags are included
        in the :attr:`include` property.
        
        Returns
        -------
        :class:`numpy.ndarray`
        '''
        return np.nonzero(self.mask())[0]
    
    def links(self):
        '''
        Returns frame containing link data.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and column 'tag'.
        '''
        points, tags = self.roads(), self.tags()
        _, inv = np.unique(np.concatenate(points), return_inverse=True, axis=0)
        lengths = [len(sequence) for sequence in points]
        cumsum = [0] + np.cumsum(lengths).tolist()
        link_data = []
        for i, n in enumerate(lengths):
            indices = inv[cumsum[i]:cumsum[i+1]]
            link_data.extend(np.column_stack([
                indices[:-1], indices[1:], np.full(n-1, tags[i])
                ]).tolist())
        link_data = np.array(link_data)
        ijpairs = np.sort(link_data[:,[0,1]].astype(int))
        tags = link_data[:,2].astype(str)
        df = pd.DataFrame(
            tags,
            index=pd.MultiIndex.from_arrays(ijpairs.T, names=['i', 'j']),
            columns=['tag'])
        _, indices = np.unique(ijpairs, axis=0, return_index=True)
        df = df.iloc[indices]
        return LinkData(df)
    
    def mask(self):
        '''
        Returns a boolean array of shape (N,) whose ``i``\ th element is True
        if the tag of the ``i``\ th feature is included in the :attr:`include`
        property, and False otherwise.
        
        Returns
        -------
        :class:`numpy.ndarray`, shape (N,)
        '''
        return np.isin(self._tags, self.include)
    
    @property
    def name(self):
        '''
        str: OSM source file name.
        '''
        return os.path.basename(self.fp)
    
    @property
    def N(self):
        '''
        int: Number of features in the OSM layer whose tags are included in
        the :attr:`include` property.
        '''
        return len(self._ids)

    def roads(self):
        '''
        Returns an array of shape (N,) containing the points that define each
        included road.
        
        Returns
        -------
        :class:`numpy.ndarray`, shape (N,)
            Array whose ``i``\ th element is a list of 2-tuples defining the
            geometry of the ``i``\ th road.
        '''
        return self._roads[self.indices()]
    
    @classmethod
    def sort_tags(cls, tags: Iterable[str]) -> List[str]:
        '''
        Sort tags in order from lowest to highest rank.
        
        Parameters
        ----------
        tags : Iterable
            Tags to be sorted.
        
        Returns
        -------
        List[str]
        '''
        return list(sorted(tags, key=lambda tag: cls.HIERARCHY[tag]))
    
    def tags(self):
        '''
        Returns an array of shape (N,) containing the tag of each included
        road.
        
        Returns
        -------
        :class:`numpy.ndarray`, shape (N,)
            Array whose ``i``\ th element is the tag of the ``i``\ th road.
        '''
        return self._tags[self.indices()]
    
    def vertices(self, dst: int = 4326) -> VertexData:
        '''
        Returns frame containing vertex data.
        
        Parameters
        ----------
        dst : :obj:`int`, optional
            EPSG code of CRS in which vertex coordinates are represented. If
            other than 4326, then vertex coordinates are transformed. The
            default is 4326.
        
        Returns
        -------
        :class:`VertexData`
        
        Raises
        ------
        EPSGError
            If `dst` EPSG code is invalid.
        '''
        coords = np.unique(np.concatenate(self.roads()), axis=0)
        df = pd.DataFrame(coords, columns=['x', 'y'])
        df.index.name = 'id'
        vdata = VertexData(df, 4326)
        if dst != 4326:
            vdata.transform(dst)
        return vdata


class OsmLoadTask(QgsTask):
    '''
    Task for loading features from an OSM file.
    
    Parameters
    ----------
    src : :class:`OsmSource`
        OSM source to which features are loaded.
    fp : str
        Path to OSM file.
    '''
    
    def __init__(self, src: OsmSource, fp: str) -> None:
        super().__init__('Loading OSM file')
        self.src, self.fp = src, fp
    
    def run(self) -> bool:
        driver = ogr.GetDriverByName('OSM')
        source = driver.Open(self.fp, 0)
        
        self.result = np.array([
            [f.GetFID(), f.GetGeometryRef().GetPoints(), f.GetField('highway')]
            for f in source.GetLayer('lines')
            ])
        return True
    
    def finished(self, success: bool) -> None:
        if success:
            self.src._ids = self.result[:,0]
            self.src._roads = np.array(self.result[:,1], dtype='object')
            self.src._tags = self.result[:,2]
