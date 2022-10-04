import dataclasses
from dataclasses import dataclass
from typing import Callable, Generator, List, Union

import numpy as np
import pandas as pd

try:
    from qgis.core import (
        NULL,
        QgsProject,
        QgsFeature,
        QgsGeometry,
        QgsSingleSymbolRenderer
        )
except:
    pass

from rnet.core.crs import CRS
from rnet.core.element import Element
from rnet.core.elevationdata import ElevationQueryEngine
from rnet.core.layer import Layer, GpkgData
from rnet.utils import point_geometry, single_marker_renderer


__all__ = ['Vertex', 'VertexData', 'VertexLayer']


@dataclass
class Vertex(Element):
    '''
    Class for representing a vertex item.
    
    Vertices are used to define road geometries.
    
    Parameters
    ----------
    id : int
        Vertex ID.
    x, y : float
        :math:`x`- and :math:`y`-coordinates.
    z : :obj:`float`, optional
        :math:`z`-coordinate. The default is None.
    gr : :obj:`int`, optional
        Group to which the vertex belongs. The default is -1, indicating no
        association.
    '''
    
    id: int
    x: float
    y: float
    z: float = None
    gr: int = -1
    idx: int = -1
    
    @property
    def dims(self) -> int:
        return 2 if self.z is None else 3
    
    def geometry(self) -> QgsGeometry:
        return point_geometry(self.x, self.y)


class VertexData:
    '''
    Class for representing vertex data.
    
    Vertices are the points used to define the geometry of each road.
    
    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Frame summarizing vertex dataset. with index 'id' and columns ['x', 'y']
        for two-dimensional vertices, or ['x', 'y', 'z'] for three-dimensional
        coordinates.
    layer : :class:`VertexLayer`, optional
        Layer for rendering vertex features.
    '''

    DEFAULT_NAME = 'vertices'

    def __init__(self, df: pd.DataFrame, layer: 'VertexLayer' = None):
        for field_name in Vertex.field_names():
            field = Vertex.__dataclass_fields__[field_name]
            if field_name not in df.columns:
                default = field.default
                if isinstance(default, dataclasses._MISSING_TYPE):
                    raise ValueError('missing required column')
                df[field_name] = default
            else:
                df[field_name] = df[field_name].astype(field.type)
        df = df[Vertex.field_names()]
        df.index = df.index.astype(int)
        self._df = df
        self.layer = layer

    def __iter__(self):
        self._i = -1
        self.__df = self._df.reset_index().astype('object')
        return self

    def __next__(self):
        try:
            self._i += 1
            return Vertex(*self.__df.iloc[self._i].tolist())
        except IndexError:
            raise StopIteration

    # == Constructors ========================================================

    @classmethod
    def empty(cls) -> 'VertexData':
        '''
        Return empty vertex dataset.
        
        Returns
        -------
        :class:`VertexData`
        '''
        df = pd.DataFrame([], columns=Vertex.field_names())
        df.index.name = 'id'
        df.attrs['crs'] = None
        return cls(df)
    
    @classmethod
    def _from_layer(cls, layer: 'VertexLayer') -> 'VertexData':
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
        attrs = np.where(attrs == NULL, None, attrs).astype(float)
        df = pd.DataFrame(attrs[:,2:], index=attrs[:,1],
                          columns=layer.field_names[2:])
        df.index.name = 'id'
        df.attrs['crs'] = layer.crs.epsg
        return cls(df, layer)
    
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
        return cls._from_layer(VertexLayer(gpkg.sublayer(layername)))
    
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
            return cls._from_layer(VertexLayer(vl[0]))
        elif len(vl) == 0:
            raise ValueError(f'no map layers named {layername!r}')
        else:
            raise ValueError(f'found multiple map layers named {layername!r}')

    # == Descriptions ========================================================

    @property
    def cols(self):
        '''List[str]: List of active columns.'''
        cols = []
        for field_name in Vertex.field_names():
            default = Vertex.__dataclass_fields__[field_name].default
            if isinstance(default, dataclasses._MISSING_TYPE):
                cols.append(field_name)
            elif default is None:
                if not np.all(self._df[field_name].isnull()):
                    cols.append(field_name)
            elif default is not None:
                if np.any(self._df[field_name] != default):
                    cols.append(field_name)
        return cols

    @property
    def crs(self):
        ''':class:`CRS`: CRS in which vertex coordinates are represented.
        
        Raises
        ------
        KeyError
            If ``crs`` attribute is missing from data frame.
        '''
        return CRS(self.df.attrs['crs'])

    @property
    def dims(self):
        '''int: Vertex dimensions.'''
        return 3 if 'z' in self.df.columns else 2

    @property
    def df(self):
        ''':class:`pandas.DataFrame`: Frame summarizing vertex data.'''
        return self._df[self.cols]

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

    def info(self):
        pass
    
    def within_circle(self, x: float, y: float, r: float) -> pd.DataFrame:
        '''
        Returns subset of vertices that are located within a circle.
        
        Parameters
        ----------
        x, y : float
            Circle center.
        r : float
            Circle radius.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame containing only the vertices that are located within the
            circle.
        '''
        xmin, xmax = x - r, x + r
        ymin, ymax = y - r, y + r
        mask = np.full(len(self.df), True)
        coords = self.df.to_numpy()
        if xmin is not None:
            mask = np.min([mask, coords[:,0] >= xmin], axis=0)
        if ymin is not None:
            mask = np.min([mask, coords[:,1] >= ymin], axis=0)
        if xmax is not None:
            mask = np.min([mask, coords[:,0] <= xmax], axis=0)
        if ymax is not None:
            mask = np.min([mask, coords[:,1] <= ymax], axis=0)
        df = self.df.loc[mask]
        coords = df.to_numpy()
        dx = coords[:,0] - x
        dy = coords[:,1] - y
        mask = np.min([mask, dx**2 + dy**2 <= r**2], axis=0)
        return df.loc[mask]

    # == Manipulation ========================================================

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
        self._df['z'] = list(engine.query(coords))

    def flatten(self) -> None:
        '''
        Remove 'z' column.
        '''
        self._df['z'] = None

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
            coords = self.crs.transform(self._df[['x', 'y']].to_numpy(), dst)
            self._df['x'] = coords[:,0]
            self._df['y'] = coords[:,1]
            self._df.attrs['crs'] = dst
            self.layer = None

    def transformed(self, dst: int) -> 'VertexData':
        '''
        Returns a new :class:`VertexData` instance with vertex coordinates
        transformed.
        
        Parameters
        ----------
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
            df.attrs['crs'] = dst
        return VertexData(df, dst)

    # == Iteration ===========================================================

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
        for i, vertex in enumerate(self, 1):
            report(i/N*100)
            yield vertex.feature(i)

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

    # == Output ==============================================================

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
            self.layer = VertexLayer.create(self.crs.epsg)
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


class VertexLayer(Layer):
    '''
    Class for representing a vertex layer.
    '''

    @classmethod
    def create(cls, crs: int, layername: str = 'vertices') -> 'VertexLayer':
        '''
        Returns an instance of :class:`VertexLayer`.
        
        Parameters
        ----------
        crs : int
            EPSG code of the CRS in which vertex coordinates are represented.
        layername : :obj:`str`, optional
            Layer name. The default is 'vertices'.
        
        Returns
        -------
        :class:`VertexLayer`
        '''
        return super().create('point', crs, layername, Vertex.fields())

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
