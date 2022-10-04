from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Callable, Generator, Dict, List
try:
    from qgis.core import (
        QgsFeature,
        QgsGeometry,
        QgsSingleSymbolRenderer
        )
except:
    pass
from rnet.core.vertexdata import VertexData
from rnet.core.element import Element
from rnet.core.layer import Layer, GpkgData
from rnet.elements.placedata import PlaceData
from rnet.utils import point_geometry, rulebased_node_renderer


__all__ = ['Node', 'NodeData', 'NodeLayer']


@dataclass
class Node(Element):
    '''
    Class for representing a node item.
    
    Nodes represent intersections and dead ends in a road network.
    
    Parameters
    ----------
    id : int
        Node ID.
    x, y : float
        :math:`x`- and :math:`y`-coordinates.
    z : :obj:`float`, optional
        :math:`z`-coordinate. The default is None.
    gr : :obj:`int`, optional
        Group to which the node belongs. The default is -1, indicating no
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


class NodeData(VertexData):
    '''
    Class for representing node data.
    '''
    
    DEFAULT_NAME = 'nodes'
    
    def __init__(self, df, layer=None):
        super().__init__(df, layer)

    def __iter__(self):
        self._i = -1
        self.__df = self._df.reset_index().astype('object')
        return self

    def __next__(self):
        try:
            self._i += 1
            return Node(*self.__df.iloc[self._i].tolist())
        except IndexError:
            raise StopIteration

    # == Descriptions ========================================================
    
    def anodes(self, place_data: PlaceData, r: float) -> pd.DataFrame:
        _df = pd.DataFrame(index=self._df.index, columns=list(place_data.df.index))
        _df.index.name = 'id'
        coords = self._df[['x', 'y']].to_numpy()
        rsq = r ** 2
        for place in place_data:
            x, y = place.x, place.y
            dx = coords[:,0] - x
            dy = coords[:,1] - y
            _df[place.id] = (dx**2 + dy**2 <= rsq)
        groups = place_data.groups(r)
        df = pd.DataFrame(index=_df.index, columns=list(range(len(groups))))
        for group in place_data.groups(r):
            member_ids = [m.id for m in group.members]
            df[group.id] = np.max(_df[member_ids].to_numpy(), axis=1)
        return df

    def bnodes(self) -> Dict[int, List[int]]:
        bnodes = defaultdict(list)
        for n in self:
            if n.gr != -1:
                bnodes[n.gr].append((n.id, n.idx))
        bnodes = dict(bnodes)
        num_groups = len(bnodes)
        for i in range(num_groups):
            arr = np.array(bnodes[i])
            arr = arr[np.argsort(arr[:,1])]
            bnodes[i] = arr[:,0]
        return bnodes

    # == Constructors ========================================================

    @classmethod
    def from_gpkg(cls, gpkg, layername='nodes'):
        '''
        Instantiates :class:`NodeData` from a GeoPackage layer.
        
        Parameters
        ----------
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GeoPackage
            containing the node data.
        layername : str, optional
            Name of the layer containing the node data. The default is
            'nodes'.
        
        Returns
        -------
        :class:`NodeData`
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("arg 'gpkg' expected type 'str' or 'GpkgData'")
        return cls._from_layer(NodeLayer(gpkg.sublayer(layername)))
    
    @classmethod
    def from_ml(cls, layername='nodes'):
        '''
        Instantiates :class:`NodeData` from map layer with specified name.
        
        Parameters
        ----------
        layername : str, optional
            Layer name. The default is 'nodes'.
        
        Returns
        -------
        :class:`NodeData`
        
        Raises
        ------
        ValueError
            If the project contains multiple layers with the specified name.
        '''
        return super().from_ml(layername)

    # == Iteration ===========================================================

    def generate(self, report: Callable[[float], None] = lambda x: None
                 ) -> Generator[QgsFeature, None, None]:
        '''
        Yields node features with point geometry and attributes 'id', 'x',
        and 'y'.
        
        Parameters
        ----------
        report : Callable[[float], None], optional
            Function for reporting generator progress.
        
        Yields
        ------
        :class:`qgis.core.QgsFeature`
        '''
        N = len(self.df)
        for i, node in enumerate(self, 1):
            report(i/N*100)
            yield node.feature(i)

    # == Output ==============================================================

    def render(self, groupname: str = '', index: int = 0, **kwargs) -> None:
        '''
        Renders a newly created vector layer that is populated with node
        features. The existing layer is overwritten.
        
        Parameters
        ----------
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
        :meth:`NodeLayer.renderer`
            Returns renderer for the vertex layer.
        '''
        if self.layer is None:
            self.layer = NodeLayer.create(self.crs.epsg)
            self.layer.render(**kwargs)
        self.layer.populate(self.generate)
        if len(kwargs) > 0:
            self.layer.render(**kwargs)
        self.layer.add(groupname, index)

    def to_gpkg(self, gpkg, layername='nodes'):
        '''
        Saves node features to a GPKG layer. If there exists a
        :class:`NodeLayer` associated with the instance, then the contents
        of the layer are saved. Otherwise, features generated by the
        :meth:`generate` method are saved.
        
        Parameters
        ----------
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GPKG layer to which
            vertices will be saved.
        layername : str, optional
            Name of the GPKG layer to which vertices are saved. The default
            is 'nodes'.
        '''
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected 'str' or 'GpkgData' for argument 'gpkg'")
        
        if self.layer is None:
            gpkg.write_features(layername, self.generate, self.crs,
                                NodeLayer.fields, 'point')
        else:
            self.layer.save(gpkg, layername)


class NodeLayer(Layer):
    '''
    Class for representing a node layer.
    '''

    @classmethod
    def create(cls, crs: int, layername: str = 'nodes') -> 'NodeLayer':
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
        return super().create('point', crs, layername, Node.fields())

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
        return rulebased_node_renderer(**kwargs)
