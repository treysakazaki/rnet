from dataclasses import dataclass

import networkx as nx

from rnet.core.elevationdata import ElevationQueryEngine
from rnet.core.layer import LayerGroup
from rnet.core.mapdata import MapData
from rnet.core.sequences import Path
from rnet.core.nodedata import NodeData
from rnet.core.edgedata import EdgeData


__all__ = ['GraphData']


@dataclass
class GraphData:
    '''
    Class representing a graph.
    
    Parameters
    ----------
    ndata : :class:`NodeData`
        Node data.
    edata : :class:`EdgeData`
        Edge data.
    mdata : :class:`MapData`
        Map data.
    '''
    
    ndata: NodeData
    edata: EdgeData
    mdata: MapData
    _edge_lengths = None
    _nx = None
    
    @property
    def crs(self):
        '''int: EPSG code of the CRS in which vertex and node coordinates are
        represented.'''
        return self.vdata.crs
    
    @property
    def dims(self):
        return self.vdata.dims
    
    def edge_coords(self):
        '''
        Returns the coordinates of the vertices that define each edge.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and column 'coords'.
        
        See also
        --------
        :meth:`EdgeData.coords`
        '''
        return self.edata.coords(self.vdata)
    
    def edge_lengths(self, dst=None):
        '''
        Returns edge lengths.
        
        Paramters
        ---------
        dst : :obj:`int`, optional
            EPSG code of CRS in which lengths are represented. If None, then
            the :attr:`crs` property is used. The default is None.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and column 'length'.
        '''
        if dst is None:
            return self.edata.lengths(self.vdata)
        else:
            return self.edata.lengths(self.vdata.transformed(dst))
    
    def edge_tags(self):
        '''
        Returns edge tags.
        
        Returns
        -------
        :class:`pandas.DataFrame`
            Frame with multi-index ['i', 'j'] and column 'tag'.
        '''
        return self.edata.tags(self.ldata)
    
    def edges(self):
        '''
        Yields edges in the data set.
        
        Parameters
        ----------
        engine : :class:`ElevationQueryEngine`
            Engine for querying vertex elevations.
        
        Yields
        ------
        :class:`Edge`
        '''
        return self.edata.edges(self.vdata, self.ldata)
    
    def expand(self, engine: ElevationQueryEngine) -> None:
        self.vdata.expand(engine)
        self.ndata.df['z'] = self.vdata.df['z'].loc[self.ndata.df.index]
    
    def flatten(self) -> None:
        self.vdata.flatten()
        self.ndata.flatten()
    
    @classmethod
    def from_gpkg(cls, path_to_gpkg):
        '''
        Instantiates :class:`GraphData` from GPKG layers.
        
        Parameters
        ----------
        path_to_gpkg : str
            Path to GPKG file. The GPKG file must contain the layers
            'vertices', 'links', 'nodes', and 'edges'.
        
        Returns
        -------
        :class:`GraphData`
        '''
        ndata = NodeData.from_gpkg(path_to_gpkg)
        edata = EdgeData.from_gpkg(path_to_gpkg)
        mdata = MapData.from_gpkg(path_to_gpkg)
        return cls(ndata, edata, mdata)
    
    @classmethod
    def from_group(cls, groupname):
        group = LayerGroup(groupname)
        ndata = NodeData.from_vl(group.layer('nodes'))
        edata = EdgeData.from_vl(group.layer('edges'))
        mdata = MapData.from_group(groupname)
        return cls(ndata, edata, mdata)
    
    @classmethod
    def from_osm(cls, path_to_osm, *, include=None):
        '''
        Instantiates :class:`GraphData` from the features in an OSM file.
        
        Parameters:
            path_to_osm (str):
        
        Keyword arguments:
            include (:obj:`str` or :obj:`List[str]`, optional): 
        
        Returns:
            GraphData:
        '''
        if include is None:
            mapdata = MapData.from_osm(path_to_osm)
        else:
            mapdata = MapData.from_osm(path_to_osm, include=include)
        return cls(mapdata.nodes(), mapdata.edges(), mapdata)

    @property
    def ldata(self):
        '''
        Alias for :attr:`mdata.link_data`.
        '''
        return self.mdata.ldata
    
    def link_headings(self):
        return self.ldata.headings(self.vdata)
    
    def node_elevations(self, engine, include_xy=False):
        '''
        Returns frame containing elevation at each node.
        
        Parameters
        ----------
        engine : :class:`ElevationQueryEngine`
            Engine for querying elevations.
        include_xy : bool, optional
            If True, then the returned frame will contain the columns 'x', 'y',
            and 'z'. If False, only the 'z' column will be included. The
            default is False.
        
        Returns
        -------
        pandas.DataFrame
        
        See also
        --------
        :meth:`NodeData.elevations`
        '''
        return self.ndata.elevations(engine, include_xy)
    
    def nx(self, dst=None):
        '''
        Returns ``networkx.Graph`` object representing a weighted undirected
        graph. Weights are obtained by the ``edge_lengths()`` method.
        
        Parameters:
            dst (:obj:`int`, optional): EPSG code of the CRS in which edge
                weights (lengths) are represented. If None, then the CRS of
                the vertex coordinates is used.
        
        Hint:
            The CRS of the vertex data is accessible via the ``vdata.crs``
            property.
        
        See also:
            :meth:`edge_lengths()`
                Returns edge lengths in a frame with multi-index ['i', 'j'],
                and column 'length'.
        '''
        if self._nx is None or self._nx[0] != dst:
            G = nx.Graph()
            G.add_weighted_edges_from(
                self.edge_lengths(dst).to_records().tolist())
            self._nx = (dst, G)
        return self._nx[1]
    
    def rand(self, N=1, replace=False):
        '''
        Returns `N` randomly selected node IDs.
        
        Parameters
        ----------
        N : int, optional
            Number of node IDs to return. The default is 1.
        replace : bool, optional
            Whether to choose node IDs with replacement. The default is False.
        
        Returns
        -------
        list of int
        '''
        return self.ndata.rand(N, replace)
    
    def render(self, groupname: str = '', params: dict = {}) -> None:
        '''
        Renders all graph elements.
        
        Parameters
        ----------
        groupname : :obj:`str`, optional
            Name of group to which the new layers are added. The group is
            created if it does not already exist. If '', then the layers
            are added to the layer tree root. The default is ''.
        params : dict of dict, optional
            Dictionary with any of the keys {'vertices', 'links', 'nodes',
            'edges'} for providing keyword arguments defining renderer
            settings.
        '''
        self.ldata.render(self.vdata, groupname, **params.get('links', {}))
        self.edata.render(self.vdata, self.ldata, groupname, **params.get('edges', {}))
        self.vdata.render(groupname, **params.get('vertices', {}))
        self.ndata.render(groupname, **params.get('nodes', {}))
    
    def sequence(self, nsequence):
        '''
        Returns an instance of :class:`NodeSequence`.
        
        Parameters:
            nsequence (List[int]): Sequence of node IDs.
        
        Returns:
            rnet.NodeSequence:
        '''
        pass
    
    def shortest_path(self, i, j, dst=None):
        '''
        Returns shortest path between node `i` and node `j`.
        
        Parameters:
            i (int): Start node.
            j (int): End node.
        
        Returns:
            List[int]:
        
        Hint:
            For best results, first call the :meth:`nx` method to generate a
            ``nx.Graph`` object. Then, use :meth:`shortest_path` for shortest
            path queries.
        '''
        return Path(self, nx.shortest_path(self.nx(dst), i, j, 'weight'))
    
    def flatten(self):
        '''
        Convert three-dimensional to two-dimensional model by removing the
        'z' column in the vertex and node data frames.
        '''
        pass
    
    def to_gpkg(self, gpkg):
        '''
        Saves the graph to a GeoPackage.
        
        Parameters
        ----------
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GPKG to which
            features will be saved.
        '''
        # TODO: bug fix
        self.ldata.to_gpkg(self.vdata, gpkg)
        self.edata.to_gpkg(self.vdata, self.ldata, gpkg)
        self.vdata.to_gpkg(gpkg)
        self.ndata.to_gpkg(gpkg)
    
    def transform(self, dst):
        '''
        Transforms :math:`(x, y)` coordinates of the vertices and nodes.
        
        Parameters
        ----------
        dst : int
            EPSG code of the CRS to which vertex and node coordinates will
            be transformed.
        '''
        self.vdata.transform(dst)
        self.ndata.transform(dst)
    
    def transformed(self, dst):
        '''
        Returns a new instance of :class:`GraphData` with vertex and node
        coordinates transformed to a new CRS.
        
        Parameters
        ----------
        dst : int
            EPSG code of the destination CRS.
        
        Returns
        -------
        :class:`GraphData`
        '''
        return GraphData(self.ndata.transformed(dst),
                         self.edata,
                         MapData(self.vdata.transformed(dst), self.ldata))
    
    @property
    def vdata(self):
        '''
        Alias for :attr:`mdata.vertex_data`.
        '''
        return self.mdata.vdata
    