from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Generator, List, Tuple, Union

import numpy as np
import pandas as pd

try:
    from qgis.core import QgsFeatureRequest, QgsProject, QgsGeometry, QgsFeature
except:
    pass

import rnet as rn
from rnet.core.element import Element
from rnet.core.field import Field
from rnet.core.layer import Layer, GpkgData
from rnet.core.vertexdata import VertexData
from rnet.utils import (
    categorized_road_renderer,
    line_geometry
    )


__all__ = ['Link', 'LinkData', 'LinkLayer']


@dataclass
class Link(Element):
    '''
    Data class representing a link.
    
    Parameters
    ----------
    i : int
        Start vertex ID.
    j : int
        End vertex ID.
    tag : str
        Link tag.
    x0 : float
        Start vertex `x`-coordinate.
    y0 : float
        Start vertex `y`-coordinate.
    xf : float
        End vertex `x`-coordinate.
    yf : float
        End vertex `y`-coordinate.
    '''
    
    i: int
    j: int
    tag: str
    x0: float = field(repr=False)
    y0: float = field(repr=False)
    xf: float = field(repr=False)
    yf: float = field(repr=False)
    
    def geometry(self) -> QgsGeometry:
        '''
        Returns the link geometry.
        
        Returns
        -------
        qgis.core.QgsGeometry
        '''
        return line_geometry([(self.x0, self.y0), (self.xf, self.yf)])
    
    def reverse(self):
        '''
        Reverses the start and end points.
        '''
        self.i, self.j = self.j, self.i
        self.x0, self.xf = self.xf, self.x0
        self.y0, self.yf = self.yf, self.y0


@dataclass
class LinkData:
    '''
    Class for representing undirected link data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing link data with multi-index ['i', 'j'] and column
        'tag'.
    layer : :class:`LinkLayer`, optional
        Layer for rendering link features.
    '''
    
    DEFAULT_NAME = 'links'
    
    def __init__(self, df: pd.DataFrame, layer: 'LinkLayer' = None):
        self.df = df
        self.layer = layer
    
    def __contains__(self, ij: Tuple[int, int]) -> bool:
        return ij in self.df.index
    
    def coords(self, vdata: VertexData) -> pd.DataFrame:
        '''
        Returns frame containing link coordiantes.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the start and end
            points of each link.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and columns ['x0', 'y0', 'xf',
            'yf'].
        '''
        coords = vdata.df[['x', 'y']].to_numpy()[self.pairs.flatten()]
        coords = coords.reshape(-1,4)
        return pd.DataFrame(coords, index=self.df.index,
                            columns=['x0', 'y0', 'xf', 'yf'])
    
    @classmethod
    def from_gpkg(cls, gpkg: Union[str, GpkgData], layername: str = 'links'
                  ) -> 'LinkData':
        '''
        Instantiates :class:`LinkData` from a GeoPackage layer.
        
        Parameters
        ----------
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GeoPackage
            containing the vertex data.
        layername : str, optional
            Name of the layer containing the vertex data. The default is
            'links'.
        
        Returns
        -------
        :class:`LinkData`
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected type 'str' or 'GpkgData' for argument 'gpkg'")
        return cls.from_vl(gpkg.sublayer(layername))
    
    @classmethod
    def from_ml(cls, layername: str = 'links'):
        '''
        Instantiates :class:`LinkData` from map layer with specified name.
        
        Parameters
        ----------
        layername : str, optional
            Layer name. The default is 'links'.
        
        Returns
        -------
        :class:`LinkData`
        
        Raises
        ------
        ValueError
            If the project contains multiple layers with the specified name.
        '''
        vl = QgsProject.instance().mapLayersByName(f'{layername}')
        if len(vl) == 1:
            return cls.from_vl(vl[0])
        elif len(vl) == 0:
            raise ValueError(f'no map layers named {layername!r}')
        else:
            raise ValueError(f'found multiple map layers named {layername!r}')
    
    @classmethod
    def from_vl(cls, vl):
        '''
        Instantiates :class:`LinkData` from a vector layer.
        
        Parameters
        ----------
        vl : qgis.core.QgsVectorLayer
            Vector layer containing link features.
        
        Returns
        -------
        :class:`LinkData`
        '''
        req = QgsFeatureRequest()
        req.setFlags(QgsFeatureRequest.NoGeometry)
        req.setSubsetOfAttributes([1,2,3])
        attrs = np.array([f.attributes() for f in vl.getFeatures(req)])
        index = pd.MultiIndex.from_arrays(attrs[:,[1,2]].T, names=['i', 'j'])
        df = pd.DataFrame(attrs[:,3], index=index, columns=['tag'])
        return cls(df, LinkLayer(vl))
    
    def generate(self, vdata: VertexData,
                 report: Callable[[float], None] = lambda x: None
                 ) -> Generator[QgsFeature, None, None]:
        '''
        Yields link features with line geometry and attributes 'fid', 'i', 'j',
        and 'tag'.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data that provides the coordinates of the start and end
            points of each link.
        report : Callable[[float], None], optional
            Function for reporting generator progress.
        
        Yields
        ------
        qgis.core.QgsFeature
        '''
        N = len(self.df)
        for i, link in enumerate(self.links(vdata), 1):
            report(i/N*100)
            yield link.feature(i)
        
    def headings(self, vdata):
        '''
        Returns frame with multi-index ['i', 'j'] and column 'heading'.

        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the start and end
            points of each link.
        
        Returns
        -------
        pandas.DataFrame
        '''
        coords = self.coords(vdata).to_numpy()
        dx, dy = coords[:,2] - coords[:,0], coords[:,3] - coords[:,1]
        return pd.DataFrame(np.mod(np.degrees(np.arctan2(dy, dx)), 360),
                            index=self.df.index, columns=['heading'])
    
    def lengths(self, vdata):
        '''
        Returns frame with multi-index ['i', 'j'] and column 'length'.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the start and end
            points of each link.
        
        Returns
        -------
        pandas.DataFrame
                
        Note
        ----
        Lengths are represented in the unit of the CRS in which vertex
        coordinates are represented.
        
        See also
        --------
        :attr:`VertexData.crs`
        
        :meth:`CRS.units`
        '''
        coords = self.coords(vdata).to_numpy()
        dx, dy = coords[:,2] - coords[:,0], coords[:,3] - coords[:,1]
        return pd.DataFrame(np.linalg.norm(np.column_stack([dx, dy]), axis=1),
                            index=self.df.index, columns=['length'])
    
    def links(self, vdata: VertexData):
        '''
        Yields links in the data set.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data that provides the coordinates of the start and end
            points of each vertex.
        
        Yields
        ------
        :class:`Link`
        '''
        df = pd.concat([self.df, self.coords(vdata)], axis=1)
        for index, row in df.iterrows():
            yield Link(*index, *row)
    
    def neighbor_counts(self):
        '''
        Returns dictionary mapping vertex ID to number of neighboring vertices.
        
        Returns
        -------
        Dict[int, int]
        '''
        counts = defaultdict(lambda: 0)
        for (i, j) in self.pairs:
            counts[i] += 1
            counts[j] += 1
        return dict(counts)
    
    def neighbors(self):
        '''
        Returns dictionary mapping vertex ID to set of neighboring vertex IDs.
        
        Returns
        -------
        Dict[int, Set[int]]
        '''
        neighbors = defaultdict(set)
        for (i, j) in self.df.index.to_frame().to_numpy():
            neighbors[i].add(j)
            neighbors[j].add(i)
        return dict(neighbors)
    
    @property
    def pairs(self):
        '''
        Returns
        -------
        numpy.ndarray, shape (N, 2)
            Array listing the :math:`(i, j)` pairs in the data set.
        '''
        return self.df.index.to_frame().to_numpy()
    
    def render(self, vdata: VertexData, groupname: str = '', index: int = 0,
               **kwargs) -> None:
        '''
        Renders a newly created vector layer that is populated with link
        features. The existing layer is overwritten.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the start and end
            points of each link.
        groupname : :obj:`str`, optional
            Name of group to which the new layer is inserted. The default is
            ''.
        index : :obj:`int`, optional
            Index within the group to which the layer is inserted. The default
            is 0.
        
        Keyword arguments
        -----------------
        **kwargs : :obj:`dict`, optional
            Keyword arguments that are used to define the renderer settings.
        
        See also
        --------
        :meth:`LinkLayer.renderer`:
            Returns renderer for the link layer.
        '''
        if self.layer is None:
            self.layer = LinkLayer.create(vdata.crs.epsg)
            self.layer.render(self.tags, **kwargs)
        self.layer.populate(partial(self.generate, vdata))
        if len(kwargs) > 0:
            self.layer.render(self.tags, **kwargs)
        self.layer.add(groupname, index)
    
    def split(self, splits: Dict[Tuple[int, int], List[int]]):
        links = self.df.to_dict()['tag']
        for (i, j), points in splits.items():
            tag = links.pop((i, j))
            vseq = [i, *points, j]
            for k in range(len(vseq) - 1):
                links[(vseq[k], vseq[k+1])] = tag
        arrays = np.sort(list(links.keys())).T
        df = pd.DataFrame(
            list(links.values()),
            index=pd.MultiIndex.from_arrays(arrays, names=('i', 'j')),
            columns=['tag']
            )
        return LinkData(df)
    
    @property
    def tags(self):
        '''
        Returns
        -------
        Set[str]
            Set of link tags in the data set.
        '''
        return set(self.df['tag'])
    
    def to_gpkg(self, vdata, gpkg, layername='links'):
        '''
        Saves link features to a GPKG layer. If there exists a
        :class:`LinkLayer` associated with the instance, then the contents
        of the layer are saved. Otherwise, features generated by the
        :meth:`generate` method are saved.
        
        Parameters
        ----------
        vdata : :class:`Vertexdata`
            Vertex data that provides the coordinates of the start and end
            points of each link.
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GPKG layer to which
            vertices will be saved.
        layername : str, optional
            Name of the GPKG layer to which vertices are saved. The default
            is 'links'.
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected 'str' or 'GpkgData' for argument 'gpkg'")
        
        if self.layer is None:
            gpkg.write_features(layername, partial(self.generate, vdata),
                                vdata.crs, LinkLayer.fields, 'linestring')
        else:
            self.layer.save(gpkg, layername)


class LinkLayer(Layer):
    '''
    Class for representing a link layer.
    '''
    
    @classmethod
    def create(cls, crs: int, layername: str = 'links') -> 'LinkLayer':
        '''
        Returns an instance of :class:`LinkLayer`.
        
        Parameters
        ----------
        crs : int
            EPSG code of the CRS in which link coordinates are represented.
        layername : str, optional
            Layer name. The default is 'links'.
        
        Returns
        -------
        :class:`LinkLayer`
        '''
        fields = [Field('i', 'int'), Field('j', 'int'), Field('tag', 'str')]
        return super().create('linestring', crs, layername, fields)
    
    @classmethod
    def renderer(cls, tags, **kwargs):
        '''
        Returns the renderer used for rendering links.
        
        Parameters
        ----------
        tags : Set[str]
            The set of edge categories.
        **kwargs : :obj:`dict`, optional
            See keyword arguments for :func:`rnet.utils.rendering.categorized_road_renderer`.
        
        Returns
        -------
        qgis.core.QgsCategorizedSymbolRenderer
        '''
        return categorized_road_renderer(rn.OsmSource.sort_tags(tags), **kwargs)
