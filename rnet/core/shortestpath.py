from collections import defaultdict
from heapq import heappush, heappop
import numpy as np
from typing import Dict, Tuple, List, Union

from rnet.core.graphdata import GraphData


__all__ = ['ShortestPathEngine']


class ConnectivityError(Exception):
    '''
    Raised if a path between nodes does not exist.
    
    Parameters
    ----------
    s : int
        Source node ID.
    g : int
        Destination node ID.
    '''
    def __init__(self, s: int, g: int) -> None:
        super().__init__(f'no path from {s} to {g}')


class ShortestPathEngine:
    '''
    Engine for querying shortest paths.
    
    Parameters
    ----------
    gdata : :class:`GraphData`
        Graph data.
    method : {'dijkstra'}
        Method for querying shortest paths.
    '''
    
    def __init__(self, gdata: GraphData, method='dijkstra') -> None:
        self._rand = lambda: gdata.rand(2)
        df = gdata.edge_lengths()
        costs = dict(zip(map(tuple, df.index), df['length']))
        if method == 'dijkstra':
            self._engine = Dijkstra(costs)
            self.query = lambda s, g: self._engine.query(s, g)
    
    def rand(self):
        return self.query(*self._rand())


class Dijkstra:
    '''
    Class for conducting shortest path queries via Dijkstra's algorithm.
    
    Parameters
    ----------
    costs : Dict[Tuple[int, int], float]
        Dictionary mapping :math:`(i, j)` pairs to edge weights.
    
    Keyword arguments
    -----------------
    ordered : :obj:`bool`, optional
        Whether edges are ordered. The default is False.
    return_paths : :obj:`bool`, optional
        Whether to return paths on query. The default is True.
    '''
    
    def __init__(self, costs: Dict[Tuple[int, int], float], *,
                 ordered: bool = False, return_paths: bool = True) -> None:
        neighbors = defaultdict(set)
        if ordered:
            self.cost = lambda i, j: costs[(i, j)]
            for (i, j) in costs:
                neighbors[i].add(j)
        else:
            self.cost = lambda i, j: costs[tuple(sorted([i, j]))]
            for (i, j) in costs:
                neighbors[i].add(j)
                neighbors[j].add(i)
        self.neighbors = dict(neighbors)
        del neighbors
        self.nodes = set(np.array(list(costs)).flatten())
        self.visited = {}
        self.queues = {}
        self.queried = {}
        self.return_paths = return_paths
        if return_paths:
            self.origins = {}
    
    def query(self, s: int, g: int) -> Union[float, Tuple[List[int], float]]:
        '''
        Returns the length of the shortest path from `s` to `g`.
        
        Parameters
        ----------
        s : int
            Start node ID.
        g : int
            Goal node ID.
        
        Returns
        -------
        :obj:`float` or :obj:`Tuple[List[int], float]`
        
        Raises
        ------
        ValueError
            If either `s` or `g` does not exist.
        ConnectivityError
            If no path exists from `s` to `g`.
        '''
        if s not in self.nodes:
            raise ValueError(f'node {s} does not exist')
        if g not in self.nodes:
            raise ValueError(f'node {g} does not exist')
        
        if self.return_paths:
            try:
                return self._construct(s, g), self.queried[s][g]
            except KeyError:
                self._update(s, g)
                return self._construct(s, g), self.queried[s][g]
        else:
            try:
                return self.queried[s][g]
            except KeyError:
                self._update(s, g)
                return self.queried[s][g]
    
    def _construct(self, s: int, g: int) -> List[int]:
        origins = self.origins[s]
        path = [g]
        while path[0] != s:
            path.insert(0, origins[path[0]])
        return path
    
    def _update(self, s: int, g: int) -> None:
        visited = self.visited.setdefault(s, set())
        queried = self.queried.setdefault(s, {})
        if self.return_paths:
            origins = self.origins.setdefault(s, {})
        
        queue = self.queues.setdefault(s, [])
        if len(visited) == 0:
            heappush(queue, (0.0, s))
        
        while queue:
            c, n = heappop(queue)
            if n == g:
                heappush(queue, (c, n))
                break
            for m in self.neighbors[n].difference(visited):
                d = c + self.cost(n, m)
                if d < queried.get(m, np.inf):
                    queried[m] = d
                    heappush(queue, (d, m))
                    if self.return_paths:
                        origins[m] = n
            visited.add(n)
        else:
            raise ConnectivityError(s, g)


class BidirectionalDijkstra(Dijkstra):
    # TODO:
    def _update(self, s: int, g: int) -> None:
        visited_forward = self.visited.setdefault(s, set())
        visited_reverse = self.visited.setdefault(g, set())
        queried_forward = self.queried.setdefault(s, {})
        queried_reverse = self.queried.setdefault(g, {})
        queue_forward = self.queues.setdefault(s, [])
        queue_reverse = self.queues.setdefault(g, [])
        if len(visited_forward) == 0:
            heappush(queue_forward, (0.0, s))
        if len(visited_reverse) == 0:
            heappush(queue_reverse, (0.0, g))
        
        mu = np.inf
        
        while len(queue_forward) > 0 and len(queue_reverse) > 0:
            while True:
                c, n = heappop(queue_forward)
                if n in visited_forward:
                    continue
            for m in self.neighbors[n].difference(visited_forward):
                d = c + self.costs[tuple(sorted([n, m]))]
                if d < queried_forward.get(m, np.inf):
                    queried_forward[m] = d
                    heappush(queue_forward, (d, m))
                if m in visited_reverse:
                    mu = min(mu, d + queried_reverse[m])
            
            while True:
                c, n = heappop(queue_reverse)
                if n in visited_reverse:
                    continue
            for m in self.neighbors[n].difference(visited_reverse):
                d = c + self.costs[tuple(sorted([n, m]))]
                if d < queried_reverse.get(m, np.inf):
                    queried_reverse[m] = d
                    heappush(queue_reverse, (d, m))
                if m in visited_forward:
                    mu = min(mu, d + queried_forward[m])
            
            

def test(gdata):
    import networkx as nx
    import time
    G = gdata.nx()
    d = ShortestPathEngine(gdata)
    
    s = gdata.rand()[0]
    for _ in range(10):
        #g = gdata.rand()[0]
        s, g = gdata.rand(2)
        print(s, '->', g)
        
        try:
            start = time.perf_counter()
            nxlength = nx.shortest_path_length(G, s, g, weight='weight')
            print(nxlength, f'{int((time.perf_counter()-start)*1e3):>3} ms')
        except:
            print('Infeasible', end='\n\n')
            continue
        else:
            nxpath = nx.shortest_path(G, s, g, weight='weight')
        
        start = time.perf_counter()
        mypath, mylength = d.query(s, g)
        print(mylength, f'{int((time.perf_counter()-start)*1e3):>3} ms')
        print(nxpath == mypath, end='\n\n')
