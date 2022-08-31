from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Callable, Generator, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    from qgis.core import (
        QgsCategorizedSymbolRenderer,
        QgsFeature,
        QgsFeatureRequest,
        QgsProject
        )
except:
    pass

import rnet as rn
from rnet.core.element import Element
from rnet.core.field import Field
from rnet.core.buffers import Buffer, BufferDataSet
from rnet.core.geometry import edge_length, buffer
from rnet.core.layer import Layer, GpkgData
from rnet.core.linkdata import  LinkData
from rnet.core.vertexdata import VertexData
from rnet.utils import line_geometry, categorized_road_renderer


__all__ = ['Edge', 'EdgeData', 'EdgeLayer']


@dataclass
class Edge(Element):
    '''
    Data class representing an edge.
    
    Parameters
    ----------
    vseq : List[int]
        List of vertex IDs.
    tag : str
        Edge tag.
    coords : List[Tuple[float, float]] or List[Tuple[float, float, float]]
        Array containing two- or three-dimensional vertex coordiantes.
    crs : int
        EPSG code of CRS in which vertex coordinates are represented.
    '''
    
    i: int = field(init=False)
    j: int = field(init=False)
    vseq: list
    tag: str
    coords: list = field(repr=False)
    crs: int = field(repr=False)
    length: float = field(init=False, repr=False)
    
    def __post_init__(self):
        self.i, self.j = int(self.vseq[0]), int(self.vseq[-1])
        self.vseq = str(list(map(int, self.vseq)))
        self.coords = np.array(self.coords)
        self.length = edge_length(self.coords)
    
    def buffer(self, dist):
        buffers = []
        buffer_coords = buffer(self.coords, dist, *range(len(self.coords) - 1))
        for k in range(len(self.coords) - 1):
            p0, pf = self.coords[k:k+2]
            dx, dy = self.coords[k+1] - self.coords[k]
            heading = float(np.mod(90 - np.degrees(np.arctan2(dy, dx)), 360))
            buffers.append(Buffer(buffer_coords[k], self.crs, heading))
        return BufferDataSet(self.crs, *buffers)
    
    def geometry(self):
        '''
        Returns the edge geometry.
        
        Returns
        -------
        qgis.core.QgsGeometry
        '''
        return line_geometry(self.coords[:,[0,1]])

    def reverse(self):
        '''
        Reverses the edge direction.
        '''
        self.vseq.reverse()
        self.coords = np.flip(self.coords, axis=0)


@dataclass
class EdgeData:
    '''
    Class for representing undirected edge data.
    
    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Frame containing edge data with multi-index ['i', 'j'] and column
        'vsequence'.
    layer : :class:`EdgeLayer`
        Layer for visualizing edge data.
    '''
    
    df: pd.DataFrame = field(repr=False)
    layer: Layer = None
    
    def __contains__(self, ij: Tuple[int, int]) -> bool:
        return ij in self.df.index
    
    def coords(self, vdata: VertexData) -> pd.DataFrame:
        '''
        Returns frame containing edge coordinates.
        
        Edge coordinates are stored in a :class:`numpy.ndarray` of shape
        (N, 2), where N is the number of vertices along the edge.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and column 'coords'.
        '''
        vcoords = vdata.df.to_numpy()[
            np.fromiter(chain.from_iterable(self.df['vsequence']), int)
            ]
        coords = []
        for length in map(len, self.df['vsequence']):
            coords.append(vcoords[:length])
            vcoords = vcoords[length:]
        return pd.DataFrame(coords, index=self.df.index, columns=['coords'])
    
    def edges(self, vdata: VertexData, ldata: LinkData
              ) -> Generator[Edge, None, None]:
        '''
        Yields edges in the data set.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data that provides the coordinates of the vertices along
            each edge.
        ldata : :class:`LinkData`
            Link data that provides that tags of the links along each edge.
        
        Yields
        ------
        :class:`Edge`
        '''
        crs = vdata.crs
        df = pd.concat([self.df, self.tags(ldata), self.coords(vdata)], axis=1)
        for _, row in df.iterrows():
            yield Edge(*row, crs)

    @classmethod
    def from_gpkg(cls, gpkg: Union[str, GpkgData], layername: str = 'edges'
                  ) -> 'EdgeData':
        '''
        Instantiates :class:`EdgeData` from a GeoPackage layer.
        
        Parameters
        ----------
        gpkg : :obj:`str` or :class:`GpkgData`
            Path or :class:`GpkgData` instance specifying the GeoPackage
            to read.
        layername : :obj:`str`, optional
            Name of the layer to read. The default is 'edges'.
        
        Returns
        -------
        :class:`EdgeData`
        
        Raises
        ------
        TypeError
            If `gpkg` is unexpected type.
        LayerNotFoundError
            If layer with the specified name does not exist.
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("arg 'gpkg' expected type 'str' or 'GpkgData'")
        return cls.from_vl(gpkg.sublayer(layername))
    
    @classmethod
    def from_ml(cls, layername: str = 'edges') -> 'EdgeData':
        '''
        Instantiates :class:`EdgeData` from a map layer.
        
        Parameters
        ----------
        layername : :obj:`str`, optional
            Layer name. The default is 'edges'.
        
        Returns
        -------
        :class:`EdgeData`
        
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
        Instantiates :class:`EdgeData` from a vector layer.
        
        Parameters
        ----------
        vl : qgis.core.QgsVectorLayer
            Vector layer containing edge features.
        
        Returns
        -------
        :class:`EdgeData`
        '''
        req = QgsFeatureRequest()
        req.setFlags(QgsFeatureRequest.NoGeometry)
        req.setSubsetOfAttributes([1,2,3])
        attrs = np.array([f.attributes() for f in vl.getFeatures(req)])
        attrs[:,3] = [[int(x) for x in vseq[1:-1].split(', ')]
                      for vseq in attrs[:,3]]
        index = pd.MultiIndex.from_arrays(attrs[:,[1,2]].T, names=['i', 'j'])
        df = pd.DataFrame(attrs[:,3], index=index, columns=['vsequence'])
        return cls(df, EdgeLayer(vl))
    
    def generate(self, vdata: VertexData, ldata: LinkData,
                 report: Callable[[float], None] = lambda x: None
                 ) -> Generator[QgsFeature, None, None]:
        '''
        Yields edge features.
        
        Edge features have linestring geometry and attributes 'fid', 'i', 'j',
        and 'tag'.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data that provides the coordinates of the vertices along
            each edge.
        ldata : :class:`LinkData`
            Link data that provides the tags of the links along each edge.
        report : :obj:`Callable[float, None]`, optional
            Function for reporting generator progress.
        
        Yields
        ------
        :class:`qgis.core.QgsFeature`
        '''
        N = len(self.df)
        for i, edge in enumerate(self.edges(vdata, ldata), 1):
            report(i/N*100)
            yield edge.feature(i)
    
    def lengths(self, vdata: VertexData) -> pd.DataFrame:
        '''
        Returns frame containing edge lengths.
        
        Lengths are represented in the units of the CRS in which vertex
        coordinates are represented.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.

        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and column 'length'.
        '''
        lengths = map(edge_length, self.coords(vdata)['coords'].to_list())
        return pd.DataFrame(lengths, index=self.df.index, columns=['length'])

    def masked(self, vdata: VertexData, *, xmin: float = None,
               ymin: float = None, xmax: float = None, ymax: float = None
               ) -> 'EdgeData':
        '''
        Returns dataset containing only the edges whose vertices are all located
        within the specified region.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
        xmin : :obj:`float`, optional
            Minimum :math:`x`-coordinate.
        ymin : :obj:`float`, optional
            Minimum :math:`y`-coordinate.
        xmax : :obj:`float`, optional
            Maximum :math:`x`-coordinate.
        ymax : :obj:`float`, optional
            Maximum :math:`y`-coordinate.
        
        Returns
        -------
        :class:`EdgeData`
        
        Examples
        --------
            >>> G = rn.GraphData.from_osm(<path/to/osm>)
            >>> masked = G.edata.masked(G.vdata, xmin=140.1, ymin=35.4, xmax=140.2, ymax=35.5)
            >>> masked.render()
        '''
        kwargs = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        vdata_masked = vdata.masked(**kwargs)
        bools = np.isin(
            np.fromiter(chain.from_iterable(self.df['vsequence']), int),
            vdata_masked.df.index
            )
        mask = []
        for length in map(len, self.df['vsequence']):
            mask.append(min(bools[:length]))
            bools = bools[length:]
        mask = np.array(mask)
        return EdgeData(self.df.loc[mask])

    def render(self, vdata: VertexData, ldata: LinkData, groupname: str = '',
               index: int = 0, **kwargs) -> None:
        '''
        Renders a vector layer, populated with edge features. The existing
        layer is overwritten.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
        ldata : :class:`LinkData`
            Link data which provides the tags of the links along each edge.
        groupname : :obj:`str`, optional
            Name of group to which the layer is inserted. The default is ''.
        index : :obj:`int`, optional
            Index within the group at which the layer is inserted. The default
            is 0.
        
        Keyword arguments
        -----------------
        **kwargs : :obj:`dict`, optional
            Keyword arguments for customizing the renderer.
        '''
        if self.layer is None:
            self.layer = EdgeLayer.create(vdata.crs.epsg)
            self.layer.render(ldata.tags, **kwargs)
        self.layer.populate(partial(self.generate, vdata, ldata))
        if len(kwargs) > 0:
            self.layer.render(ldata.tags, **kwargs)
        self.layer.add(groupname, index)

    def tags(self, ldata: LinkData) -> pd.DataFrame:
        '''
        Returns frame containing edge tags.
        
        Edges inherit the tag of the first link in their sequence.
        
        Parameters
        ----------
        ldata : :class:`LinkData`
            Link data which provides the tags of the links along each edge.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and column 'tag'.
        '''
        tags = ldata.df['tag'].loc[map(
            tuple, np.sort([x[:2] for x in self.df['vsequence'].to_list()])
            )]
        return pd.DataFrame(tags.to_list(), index=self.df.index, columns=['tag'])

    def to_gpkg(self, vdata: VertexData, ldata: LinkData,
                gpkg: Union[str, GpkgData], layername: str = 'edges') -> None:
        '''
        Saves edge features to a GPKG layer.
        
        If there exists a :class:`EdgeLayer` associated with the instance,
        then the contents of the layer are saved. Otherwise, features
        generated by the :meth:`generate` method are saved.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
        ldata : :class:`LinkData`
            Link data which provides the tags of the links along each edge.
        gpkg : :obj:`str` or :class:`GpkgData`
            Path or :class:`GpkgData` instance specifying the GeoPackage to
            which edge data is saved.
        layername : str, optional
            Name of the GeoPackage layer to which edges are saved. The default
            is 'edges'.
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected 'str' or 'GpkgData' for argument 'gpkg'")

        if self.layer is None:
            gpkg.write_features(layername,
                                partial(self.generate, vdata, ldata),
                                vdata.crs, EdgeLayer.fields, 'linestring')
        else:
            self.layer.save(gpkg, layername)


class EdgeLayer(Layer):
    '''
    Class for representing an edge layer.
    '''

    @classmethod
    def create(cls, crs: int, layername: str = 'edges') -> 'EdgeLayer':
        '''
        Returns an instance of :class:`EdgeLayer`.
        
        Parameters
        ----------
        crs : int
            EPSG code of the CRS in which link coordinates are represented.
        layername : str, optional
            Layer name. The default is 'edges'.
        
        Returns
        -------
        :class:`EdgeLayer`
        '''
        fields = [Field('i', 'int'), Field('j', 'int'),
                  Field('vsequence', 'str'), Field('tag', 'str')]
        return super().create('linestring', crs, layername, fields)

    @classmethod
    def renderer(cls, tags: Set[str], **kwargs) -> QgsCategorizedSymbolRenderer:
        '''
        Returns the renderer used for rendering links.
        
        Parameters
        ----------
        tags : Set[str]
            The set of link categories.
        **kwargs : :obj:`dict`, optional
            See keyword arguments for :func:`rnet.utils.rendering.categorized_road_renderer`.
        
        Returns
        -------
        qgis.core.QgsCategorizedSymbolRenderer
        '''
        return categorized_road_renderer(rn.OsmSource.sort_tags(tags), **kwargs)
