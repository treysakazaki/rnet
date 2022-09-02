from dataclasses import dataclass, field
from typing import Callable, Generator, List, Union

import numpy as np
import pandas as pd

try:
    from PyQt5.QtCore import QVariant
    from qgis.core import (
        QgsProject,
        QgsFeature,
        QgsField,
        QgsGeometry,
        QgsSingleSymbolRenderer
        )
except:
    pass

from rnet.core.crs import CRS
from rnet.core.element import Element
from rnet.core.field import Field
from rnet.core.elevationdata import ElevationQueryEngine
from rnet.core.layer import Layer, GpkgData
from rnet.utils import (
    point_geometry,
    single_marker_renderer
    )


__all__ = ['Vertex', 'Vertex2d', 'Vertex3d', 'VertexData', 'VertexLayer']


@dataclass
class Vertex(Element):
    '''
    Base class for two- or three-dimensional vertices.
    
    Vertices are used to define road geometries.
    
    Parameters
    ----------
    id : int
        Vertex ID.
    x : float
        :math:`x`-coordinate.
    y : float
        :math:`y`-coordinate.
    z : :obj:`float`, optional
        :math:`z`-coordinate. The default is None.
    '''
    
    id: int
    x: float
    y: float
    z: float = None
    
    @property
    def dims(self) -> int:
        if self.z is None:
            return 2
        else:
            return 3


@dataclass
class Vertex2d(Element):
    '''
    Data class representing two-dimensional vertices.
    
    Parameters
    ----------
    id : int
        Vertex ID.
    x : float
        `x`-coordinate.
    y : float
        `y`-coordinate.
    '''
    
    id: int
    x: float
    y: float
    
    def geometry(self) -> QgsGeometry:
        '''
        Returns the vertex geometry.
        
        Returns
        -------
        qgis.core.QgsGeometry
        '''
        return point_geometry(self.x, self.y)


@dataclass
class Vertex3d(Vertex2d):
    '''
    Data class representing three-dimensional vertices.
    
    Parameters
    ----------
    id : int
        Vertex ID.
    x : float
        `x`-coordinate.
    y : float
        `y`-coordinate.
    z : float
        `z`-coordinate.
    '''
    
    z: float


class VertexData:
    '''
    Class for representing vertex data.
    
    Vertices are the points used to define the geometry of each road.
    
    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Frame containing vertex data with index 'id' and columns ['x', 'y']
        for two-dimensional vertices, or ['x', 'y', 'z'] for three-dimensional
        coordinates.
    crs : :obj:`int` or :class:`CRS`
        EPSG code or :class:`CRS` instance describing CRS in which vertex
        coordinates are represented.
    layer : :class:`VertexLayer`, optional
        Layer for rendering vertex features.
    '''
    
    def __init__(self, df: pd.DataFrame, crs: Union[int, CRS],
                 layer: 'VertexLayer' = None) -> None:
        self.df = df
        if type(self.crs) is int:
            self.crs = CRS(crs)
        elif isinstance(crs, CRS):
            self.crs = crs
        self.layer = layer
    
    def __contains__(self, id: int) -> bool:
        return id in self.df.index
    
    def __len__(self) -> int:
        return len(self.df)
    
    @property
    def dims(self):
        '''int: Vertex dimensions.'''
        return len(self.df.columns)
    
    def elevations(self, engine: ElevationQueryEngine,
                   include_xy: bool = False) -> pd.DataFrame:
        '''
        Returns frame containing vertex elevations.
        
        Parameters
        ----------
        engine : :class:`ElevationQueryEngine`
            Engine for querying vertex elevations.
        include_xy : :obj:`bool`, optional
            If True, columns 'x' and 'y' are included in the resulting frame
            in addition to the column 'z'. The default is False.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            If `include_xy` is True, frame with index 'id' and columns
            ['x', 'y', 'z']. Otherwise, frame with index 'id' and column 'z'.
        
        See also
        --------
        :meth:`ElevationData.query`
        '''
        if engine.crs == self.crs:
            elevations = list(engine.query(self.df[['x', 'y']].to_numpy()))
        else:
            transformed = self.crs.transform(self.df[['x', 'y']].to_numpy(),
                                             engine.crs)
            elevations = list(engine.query(transformed))
        
        if include_xy:
            return pd.DataFrame(
                np.column_stack([self.df.to_numpy(), elevations]),
                index=self.df.index, columns=['x', 'y', 'z'])
        else:
            return pd.DataFrame(elevations, index=self.df.index, columns=['z'])
    
    def expand(self, engine: ElevationQueryEngine) -> None:
        '''
        Add 'z' column representing vertex elevations.
        
        Parameters
        ----------
        engine : :class:`ElevationQueryEngine`
            Engine for querying vertex elevations.
        '''
        if self.dims == 3:
            return
        
        if engine.crs == self.crs:
            coords = self.df[['x', 'y']].to_numpy()
        else:
            coords = self.crs.transform(self.df[['x', 'y']].to_numpy(),
                                        engine.crs)
        self.df = pd.concat([
            self.df,
            pd.DataFrame(engine.query(coords), index=self.df.index, columns=['z'])
            ], axis=1)
    
    def flatten(self) -> None:
        '''
        Remove 'z' column.
        '''
        if self.dims == 2:
            return
        self.df = self.df[['x', 'y']]
    
    @classmethod
    def from_layer(cls, layer: 'VertexLayer') -> 'VertexData':
        '''
        Instantiates :class:`VertexData` from a layer.
        
        Parameters
        ----------
        layer : :class:`VertexLayer`
            Layer containing vertex features.
        
        Returns
        -------
        :class:`VertexData`
        '''
        attrs = np.array([f.attributes() for f in layer.features()])
        df = pd.DataFrame(attrs[:,2:].astype(float), index=attrs[:,1],
                          columns=layer.field_names[2:])
        df.index.name = 'id'
        return cls(df, layer.crs, layer)
    
    @classmethod
    def from_gpkg(cls, gpkg: Union[str, GpkgData], layername: str = 'vertices'
                  ) -> 'VertexData':
        '''
        Instantiates :class:`VertexData` from a GeoPackage layer.
        
        Parameters
        ----------
        gpkg : :obj:`str` or :class:`GpkgData`
            Path or :class:`GpkgData` instance specifying the GeoPackage
            containing vertex data.
        layername : str, optional
            Name of the layer containing vertex data. The default is
            'vertices'.
        
        Returns
        -------
        :class:`VertexData`
        
        See also
        --------
        :class:`GpkgData`
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected type 'str' or 'GpkgData' for argument 'gpkg'")
        return cls.from_layer(VertexLayer(gpkg.sublayer(layername)))
    
    @classmethod
    def from_ml(cls, layername: str = 'vertices') -> 'VertexData':
        '''
        Instantiates :class:`VertexData` from map layer with specified name.
        
        Parameters
        ----------
        layername : :obj:`str`, optional
            Layer name. The default is 'vertices'.
        
        Returns
        -------
        :class:`VertexData`
        
        Raises
        ------
        ValueError
            If the project contains multiple layers with the specified name.
        '''
        vl = QgsProject.instance().mapLayersByName(f'{layername}')
        if len(vl) == 1:
            return cls.from_layer(VertexLayer(vl[0]))
        elif len(vl) == 0:
            raise ValueError(f'no map layers named {layername!r}')
        else:
            raise ValueError(f'found multiple map layers named {layername!r}')
    
    def generate(self, report: Callable[[float], None] = lambda x: None
                 ) -> Generator[QgsFeature, None, None]:
        '''
        Yields vertex features. Vertex features have point geometry and
        attributes 'id', 'x', and 'y'.
        
        Parameters
        ----------
        report : :obj:`Callable[[float], None]`, optional
            Function for reporting generator progress.
        
        Yields
        ------
        :class:`qgis.core.QgsFeature`
        '''
        N = len(self.df)
        for i, vertex in enumerate(self.vertices(), 1):
            report(i/N*100)
            yield vertex.feature(i)
    
    def masked(self, *, xmin: float = None, ymin: float = None,
               zmin: float = None, xmax: float = None, ymax: float = None,
               zmax: float = None) -> 'VertexData':
        '''
        Returns dataset containing only the vertices that are located within
        the specified region.
        
        Parameters
        ----------
        xmin, ymin, zmin, xmax, ymax, zmax : :obj:`float`, optional
            Coordinates specifying the desired region.
        
        Returns
        -------
        :class:`VertexData`
        '''
        mask = np.full(len(self.df), True)
        coords = self.df.to_numpy()
        if xmin is not None:
            mask = np.min([mask, coords[:,0] >= xmin], axis=0)
        if ymin is not None:
            mask = np.min([mask, coords[:,1] >= ymin], axis=0)
        if zmin is not None:
            mask = np.min([mask, coords[:,2] >= zmin], axis=0)
        if xmax is not None:
            mask = np.min([mask, coords[:,0] <= xmax], axis=0)
        if ymax is not None:
            mask = np.min([mask, coords[:,1] <= ymax], axis=0)
        if zmax is not None:
            mask = np.min([mask, coords[:,2] <= zmax], axis=0)
        return VertexData(self.df.loc[mask], self.crs)

    def rand(self, N: int = 1, replace: bool = False) -> List[int]:
        '''
        Returns the IDs of `N` randomly chosen vertices.
        
        Parameters
        ----------
        N : int, optional
            Number of vertices to choose. The default is 1.
        replace : bool, optional
            Whether to choose vertex IDs with replacement. The default is
            False.
        
        Returns
        -------
        List[int]
        '''
        return list(np.random.choice(list(self.df.index), N, replace))
    
    def render(self, groupname: str = '', index: int = 0, **kwargs) -> None:
        '''
        Renders vertex features. Existing features are overwritten.

        Parameters
        ----------
        engine : :class:`ElevationQueryEngine`, optional
            Engine for querying the vertex elevations.
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
        :meth:`VertexLayer.renderer`
            Returns renderer for the vertex layer.
        '''
        if self.layer is None:
            self.layer = VertexLayer.create(self.crs.epsg, self.dims)
            self.layer.render(**kwargs)
        self.layer.populate(self.generate)
        if len(kwargs) > 0:
            self.layer.render(**kwargs)
        self.layer.add(groupname, index)

    def to_csv(self, path_to_csv: str) -> None:
        raise NotImplementedError

    def to_gpkg(self, gpkg: Union[str, GpkgData], layername: str = 'vertices'
                ) -> None:
        '''
        Saves vertex features to a GPKG layer. If there exists a
        :class:`VertexLayer` associated with the instance, then the contents
        of the layer are saved. Otherwise, features generated by the
        :meth:`generate` method are saved.
        
        Parameters
        ----------
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GPKG to which
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
            gpkg.write_features(layername, self.generate, self.crs,
                                VertexLayer.fields, 'point')
        else:
            self.layer.save(gpkg, layername)

    def transform(self, dst: int) -> None:
        '''
        Transforms :math:`(x, y)` coordinates stored in the ``df`` attribute
        to a new coordinate system.
        
        Parameters
        ----------
        dst : int
            EPSG code of destination CRS.
        
        Raises
        ------
        EPSGError
            If the EPSG code of the destination CRS is invalid.
        
        See also
        --------
        :meth:`transformed`
        '''
        if dst == self.crs.epsg:
            pass
        else:
            coords = self.crs.transform(self.df[['x', 'y']].to_numpy(), dst)
            self.df['x'] = coords[:,0]
            self.df['y'] = coords[:,1]
            self.crs = CRS(dst)
            self.layer = None

    def transformed(self, dst: int) -> 'VertexData':
        '''
        Returns a new :class:`VertexData` instance with vertex coordinates
        transformed.
        
        Paramters
        ---------
        dst : int
            EPSG code of destination CRS.
        
        Returns
        -------
        :class:`VertexData`:
        
        Raises
        ------
        EPSGError
            If the EPSG code of the destination CRS is invalid.
        
        See also
        --------
        :meth:`transform`
        '''
        df = self.df.copy()
        if dst == self.crs:
            pass
        else:
            coords = self.crs.transform(df[['x', 'y']].to_numpy(), dst)
            df['x'] = coords[:,0]
            df['y'] = coords[:,1]
        return VertexData(df, dst)

    def vertices(self) -> Generator[Union[Vertex2d, Vertex3d], None, None]:
        '''
        Yields vertices in the data set.
        
        Yields
        ------
        :class:`Vertex2d` or :class:`Vertex3d`
        '''
        if self.dims == 2:
            for id, row in self.df.iterrows():
                yield Vertex2d(id, *list(row))
        elif self.dims == 3:
            for id, row in self.df.iterrows():
                yield Vertex3d(id, *list(row))


class VertexLayer(Layer):
    '''
    Class for representing a vertex layer.
    '''

    @property
    def dims(self):
        '''int: Number of vertex dimensions.'''
        return len(self.vl.fields()) - 2

    @classmethod
    def create(cls, crs: int, dims: int, layername: str = 'vertices'
               ) -> 'VertexLayer':
        '''
        Returns an instance of :class:`VertexLayer`.
        
        Parameters
        ----------
        crs : int
            EPSG code of the CRS in which vertex coordinates are represented.
        dims : {2, 3}
            Vertex dimensions.
        layername : :obj:`str`, optional
            Layer name. The default is 'vertices'.
        
        Returns
        -------
        :class:`VertexLayer`
        
        Raises
        ------
        ValueError
            If `dims` is neither 2 or 3.
        '''
        fields = [Field('id', 'int'), Field('x', 'double'), Field('y', 'double')]
        if dims == 2:
            pass
        elif dims == 3:
            fields.append(Field('z', 'double'))
        else:
            raise ValueError("arg 'dims' expected value 2 or 3")
        return super().create('point', crs, layername, fields)

    @staticmethod
    def renderer(**kwargs) -> QgsSingleSymbolRenderer:
        '''
        Returns the renderer used for rendering vertices.
        
        Parameters
        ----------
        **kwargs : dict, optional
            See keyword arguments for :func:`rnet.utils.symbols.marker_symbol`.
        
        Returns
        -------
        qgis.core.QgsSingleSymbolRenderer
        '''
        kwargs.setdefault('color', (210,210,210))
        return single_marker_renderer(**kwargs)

    def toggle_dims(self) -> None:
        '''
        Toggle between two- and three-dimensional vertices.
        '''
        if self.dims == 2:
            self.vl.dataProvider().addAttributes([QgsField('z', QVariant.Double)])
        elif self.dims == 3:
            self.vl.dataProvider().deleteAttributes([4])
        else:
            raise ValueError('unexpected number of fields')
        self.vl.updateFields()
