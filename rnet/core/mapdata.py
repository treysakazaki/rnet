from collections import defaultdict
from dataclasses import dataclass
from functools import partial
import numpy as np
import pandas as pd
from typing import Set

from rnet.core.osmsource import OsmSource
from rnet.core.nodedata import NodeData
from rnet.core.edgedata import EdgeData
from rnet.core.vertexdata import VertexData
from rnet.core.linkdata import LinkData
from rnet.elements.placedata import PlaceData
from rnet.core.geometry import segment_circle_intersection


__all__ = ['MapData']


def intersect_map_places(mdata: 'MapData', pdata: PlaceData, r: float):
    '''
    Returns a new instance of :class:`MapData` with border nodes added to the
    set of vertices and links broken to accomodate the border nodes.
    
    Parameters
    ----------
    mdata
    pdata
    
    Returns
    -------
    mdata : :class:`MapData`
        New instance of :class:`MapData`.
    bnodes : Set[int]
        Border node IDs.
    '''
    link_coords = mdata.ldata.coords(mdata.vdata)
    groups = pdata.groups(r)
    group_ids = {member.id: group.id for group in groups for member in group}
    outer_arcs = defaultdict(list)
    i = 0
    for group in groups:
        for (pid, t1, t2) in group.arcs():
            outer_arcs[pid].append((i, t1, t2))
            i += 1
    
    points = []
    ijpairs = []
    groups = []
    angles = []
    ij = mdata.ldata.df.index.to_frame().to_numpy()
    for place_id, arcs in outer_arcs.items():
        _, x, y = pdata.df.loc[place_id].to_numpy()
        xmin, xmax, ymin, ymax = x - r, x + r, y - r, y + r
        masked_vertices = mdata.vdata.masked(
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        mask = np.max(np.isin(ij.flatten(), masked_vertices.df.index).reshape(-1, 2),
                      axis=1)
        coords = link_coords.loc[mask].to_numpy()
        new_points, index = segment_circle_intersection(coords, x, y, r, True)
        
        # Keep only if point of intersection is on an outer arc
        t = np.degrees(np.arctan2(new_points[:,1] - y, new_points[:,0] - x))
        indices = []
        for (i, tmin, tmax) in arcs:
            inds = list(np.flatnonzero(
                ((t>=tmin)&(t<=tmax))|((t+360>=tmin)&(t+360<=tmax))
                ))
            indices.extend(inds)
            angles.append(np.column_stack((t[inds], np.full(len(inds), i))))
        new_points = new_points[indices]
        index = index[indices]

        # Record results
        points.append(new_points)
        ijpairs.append(ij[np.flatnonzero(mask)][index])
        groups.append(np.full(len(new_points), group_ids[place_id]))

    points = np.vstack(points)
    N = len(points)
    ijpairs = np.vstack(ijpairs)
    angles = np.vstack(angles)
    angles = np.array(angles, dtype=[('t', 'float'), ('i', 'int')])
    print(N, angles, len(angles))
    df = VertexData.empty()._df
    df['x'] = points[:,0]
    df['y'] = points[:,1]
    df['gr'] = np.hstack(groups)
    order = np.argsort(angles[:,0], order=('i', 't'))
    ranks = np.argsort(order)
    df['idx'] = ranks
    df.index.name = 'id'
    df = df.reset_index()
    # assert np.all(np.diff(mdata.vdata.df.index)==1)
    offset = max(mdata.vdata.df.index) + 1
    # offset = mdata.vdata.df.index.stop
    df.index += offset

    df = pd.concat([mdata.vdata._df, df])
    df.attrs['crs'] = mdata.vdata.crs.epsg
    vdata = VertexData(df)

    splits = defaultdict(list)
    for i, ij in enumerate(ijpairs, offset):
        splits[tuple(ij)].append(i)
    ldata = mdata.ldata.split(splits)

    mdata = MapData(vdata, ldata)
    bnodes = range(offset, offset + N)


    return mdata, set(bnodes)


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

    def edges(self):
        neighbors = self.ldata.neighbors()
        nodes = set(self.nodes().df.index)
        
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

    def intersect(self, place_data: PlaceData, r: float) -> 'MapData':
        mdata, bnodes = intersect_map_places(self, place_data, r)
        mdata.nodes = partial(mdata.nodes, bnodes)
        return mdata

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
    
    def nodes(self, add: Set[int] = None):
        '''
        Returns an instance of :class:`NodeData` initialized by the set of
        vertices that have exactly one or more than two neighbors.
        
        Parameters
        ----------
        add : :obj:`Set[int]`, optional
            Additional nodes. The default is None.
        
        Returns
        -------
        :class:`NodeData`:
        '''
        nodes = set(i for i, n in self.ldata.neighbor_counts().items() if n != 2)
        if type(add) is set:
            nodes.update(add)
        return NodeData(self.vdata.df.loc[sorted(nodes)])
    
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
