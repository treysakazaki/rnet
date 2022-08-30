from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Callable, Generator, Set, Tuple

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
    df : pandas.DataFrame
        Frame containing edge data with multi-index ['i', 'j'] and column
        'vsequence'.
    '''
    
    df: pd.DataFrame = field(repr=False)
    layer: Layer = None
    
    def __contains__(self, ij: Tuple[int, int]) -> bool:
        return ij in self.df.index
    
    def coords(self, vdata: VertexData) -> pd.DataFrame:
        '''
        Returns frame with multi-index ['i', 'j'] and column 'coords'.
        
        Edge coordinates are represented by an :class:`numpy.ndarray`.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
        
        Returns
        -------
        :class:`pandas.DataFrame`
        '''
        vcoords = vdata.df.to_numpy()[
            np.fromiter(chain.from_iterable(self.df['vsequence']), int)
            ]
        coords = []
        for length in map(len, self.df['vsequence']):
            coords.append(vcoords[:length])
            vcoords = vcoords[length:]
        return pd.DataFrame(coords, index=self.df.index, columns=['coords'])
    
    def edges(self, vdata: VertexData, ldata: LinkData, **kwargs
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
        
        Keyword arguments
        -----------------
        area : :obj:`Tuple[float, float, float, float]`, optional
            4-tuple specifying :math:`(x, y)` coordinates of the bottom-left
            and top-right of a rectangular area.
        mode : {1, 2, 3}, optional
            Mode for selecting edge features:
                
            1. at least one endpoint is in the `area`,
            2. both endpoints are in the `area`, or
            3. all vertices are in the `area`.
            
            The default is 1.
        
        Yields
        ------
        :class:`Edge`
        '''
        crs = vdata.crs
        df = pd.concat([self.df, self.tags(ldata), self.coords(vdata)], axis=1)
        for _, row in df.iterrows():
            yield Edge(*row, crs)

    @classmethod
    def from_gpkg(cls, gpkg, layername='edges'):
        '''
        Instantiates :class:`EdgeData` from a GeoPackage layer.
        
        Parameters
        ----------
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GeoPackage
            containing the vertex data.
        layername : str, optional
            Name of the layer containing the vertex data. The default is
            'edges'.
        
        Returns
        -------
        :class:`EdgeData`
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected type 'str' or 'GpkgData' for argument 'gpkg'")
        return cls.from_vl(gpkg.sublayer(layername))
    
    @classmethod
    def from_ml(cls, layername='edges'):
        '''
        Instantiates :class:`EdgeData` from map layer with specified name.
        
        Parameters
        ----------
        layername : str, optional
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
        Yields link features with line geometry and attributes 'fid', 'i', 'j',
        and 'tag'.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data that provides the coordinates of the vertices along
            each edge.
        ldata : :class:`LinkData`
            Link data that provides the tags of the links along each edge.
        report : Callable[float, None], optional
            Function for reporting generator progress.
        
        Yields
        ------
        qgis.core.QgsFeature
        '''
        N = len(self.df)
        for i, edge in enumerate(self.edges(vdata, ldata), 1):
            report(i/N*100)
            yield edge.feature(i)
    
    def lengths(self, vdata: VertexData) -> pd.DataFrame:
        '''
        Returns frame with multi-index ['i', 'j'] and column 'length'.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
                
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
        lengths = map(edge_length, self.coords(vdata)['coords'].to_list())
        return pd.DataFrame(lengths, index=self.df.index, columns=['length'])

    def masked(self, vdata: VertexData, **kwargs) -> 'EdgeData':
        '''
        Returns dataset containing only the edges whose vertices are all located
        within the specified region.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
        
        Keyword arguments
        -----------------
        xmin, ymin, zmin, xmax, ymax, zmax : :obj:`float`, optional
            Coordinates specifying the desired region.
        
        Returns
        -------
        :class:`EdgeData`
        '''
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
        Renders a newly created vector layer that is populated with edge
        features. The existing layer is overwritten.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data which provides the coordinates of the vertices along
            each edge.
        ldata : :class:`LinkData`
            Link data that provides the tags of the links along each edge.
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
        :meth:`EdgeLayer.renderer`:
            Returns renderer for the link layer.
        '''
        if self.layer is None:
            self.layer = EdgeLayer.create(vdata.crs.epsg)
            self.layer.render(ldata.tags, **kwargs)
        self.layer.populate(partial(self.generate, vdata, ldata))
        if len(kwargs) > 0:
            self.layer.render(ldata.tags, **kwargs)
        self.layer.add(groupname, index)

    def tags(self, ldata):
        '''
        Returns frame with multi-index ['i', 'j'] and column 'tag'.
        
        Edges inherit the tag of the first link in their vertex sequence.
        
        Parameters
        ----------
        ldata : :class:`LinkData`
            Link data which provides the tags of the links along each edge.
        
        Returns
        -------
        pandas.DataFrame
        '''
        tags = ldata.df['tag'].loc[map(
            tuple, np.sort([x[:2] for x in self.df['vsequence'].to_list()])
            )]
        return pd.DataFrame(tags.to_list(), index=self.df.index, columns=['tag'])

    def to_gpkg(self, vdata, ldata, gpkg, layername='edges'):
        '''
        Saves edge features to a GPKG layer. If there exists a
        :class:`EdgeLayer` associated with the instance, then the contents
        of the layer are saved. Otherwise, features generated by the
        :meth:`generate` method are saved.
        
        Parameters
        ----------
        vdata : :class:`VertexData`
            Vertex data that provides the coordinates of the vertices along
            each edge.
        ldata : :class:`LinkData`
            Link data that provides the tags of the links along each edge.
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GPKG layer to which
            vertices will be saved.
        layername : str, optional
            Name of the GPKG layer to which vertices are saved. The default
            is 'vertices'.
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
