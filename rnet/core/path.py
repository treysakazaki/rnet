from dataclasses import dataclass
from typing import List

from rnet.core.graphdata import GraphData
from rnet.core.shortestpath import ShortestPathEngine


@dataclass
class Path:
    
    sequence: List[int]
    length: float
    
    def __post_init__(self):
        self.S, self.G = self.sequence[0], self.sequence[-1]


class PathFactory:
    
    def __init__(self, gdata: GraphData) -> None:
        self.engine = ShortestPathEngine(gdata)
    
    def from_route(self, route):
        cost = 0.0
        for k in range(len(route) - 1):
            cost += self.engine.query(route[k], route[k+1])
        return Path(route, cost)