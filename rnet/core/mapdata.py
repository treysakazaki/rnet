from dataclasses import dataclass
import numpy as np
import pandas as pd

from rnet.core.osmsource import OsmSource
from rnet.core.nodedata import NodeData
from rnet.core.edgedata import EdgeData
from rnet.core.vertexdata import VertexData
from rnet.core.linkdata import LinkData


__all__ = ['MapData']


@dataclass
class MapData:
    '''
    Class for representing map data.
    
    Map data consists of vertices and links.
    
    Parameters
    ----------
    vertex_data : :class:`VertexData`
        Vertex data.
    link_data : :class:`LinkData`
        Link data.
    '''
    
    vertex_data: VertexData
    link_data: LinkData
    
    def edges(self, add=None):
        neighbors = self.ldata.neighbors()
        nodes = set(self.nodes(add).df.index)
        
        ijpairs = []
        sequences = []
        history = set()
        unvisited = nodes.copy()
        
        while True:
            try:
                leaves = {unvisited.pop()}
            except KeyError:
                break
            else:
                while len(leaves) > 0:
                    new_leaves = set()
                    for o in leaves:
                        for n in neighbors[o]:
                            if (o, n) in history:
                                continue
                            vseq = [o, n]
                            p, q = o, n
                            while q not in nodes:
                                x = neighbors[q].difference({p}).pop()
                                vseq.append(x)
                                p, q = q, x
                            i, j = vseq[0], vseq[-1]
                            new_leaves.add(j)
                            history.add((q, p))
                            
                            if i > j:
                                i, j = j, i
                                vseq.reverse()
                            
                            ijpairs.append((i, j))
                            sequences.append((vseq,))
                    unvisited.difference_update(leaves)
                    leaves = new_leaves.intersection(unvisited)
        
        index = pd.MultiIndex.from_tuples(ijpairs, names=['i', 'j'])
        df = pd.DataFrame(sequences, index=index, columns=['vsequence'])
        
        # Remove duplicates
        _, indices = np.unique(df.index.to_numpy(), return_index=True)
        df = df.iloc[list(indices)]
        
        return EdgeData(df)
    
    @classmethod
    def from_gpkg(cls, path_to_gpkg):
        vdata = VertexData.from_gpkg(path_to_gpkg)
        ldata = LinkData.from_gpkg(path_to_gpkg)
        return cls(vdata, ldata)
    
    @classmethod
    def from_osm(cls, path_to_osm, *, include=None):
        if include is None:
            osm = OsmSource(path_to_osm)
        else:
            osm = OsmSource(path_to_osm, include=include)
        return cls(osm.vertices(), osm.links())

    @property
    def ldata(self):
        '''Alias for ``link_data`` attribute.'''
        return self.link_data
    
    def link_coords(self):
        return self.ldata.coords(self.vdata)
    
    def link_headings(self):
        return self.ldata.headings(self.vdata)
    
    def link_lengths(self):
        return self.ldata.lengths(self.vdata)
    
    def nodes(self, add=None):
        '''
        Returns an instance of :class:`NodeData` initialized by the set of
        vertices that have exactly one or more than two neighbors.
        
        Parameters:
            add (List[int]): Additional nodes. Default: None.
        
        Returns:
            :class:`NodeData`:
        '''
        nodes = set(i for i, n in self.ldata.neighbor_counts().items() if n != 2)
        if add is None:
            pass
        else:
            nodes.update(add)
        return NodeData(self.vdata.df.loc[sorted(nodes)], self.vdata.crs)
    
    def render(self, groupname: str = '', params: dict = {}):
        self.vdata.render(groupname, **params.get('vertices', {}))
        self.ldata.render(self.vdata, groupname, **params.get('links', {}))
    
    def transform(self, dst):
        '''
        Transforms the vertex data stored in the ``vertex_data`` attribute.
        
        Parameters:
            dst (int): EPSG code of the destination CRS.
        '''
        self.vdata.transform(dst)
    
    def transformed(self, dst):
        '''
        Returns a new instance of :class:`MapData` with transformed vertices.
        
        Parameters:
            dst (int): EPSG code of the destination CRS.
        
        Returns:
            :class:`MapData`:
        '''
        return MapData(self.vdata.transformed(dst), self.ldata)
    
    @property
    def vdata(self):
        '''Alias for ``vertex_data`` attribute.'''
        return self.vertex_data
    