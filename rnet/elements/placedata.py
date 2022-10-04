from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Callable, Generator, List, Tuple, Union
from qgis.core import QgsFeature
from rnet.core.crs import CRS
from rnet.utils import read_csv, ccl
from rnet.core.element import Element
from rnet.core.layer import Layer, GpkgData
from rnet.utils import (
    point_geometry,
    polygon_geometry,
    single_marker_renderer,
    single_fill_renderer
    )
from rnet.core.path import Path, PathLayer


def outer_arcs(circles):
    '''
    Returns outermost arcs of a group of intersecting circles.
    
    Parameters:
        circles (Dict[int, Tuple[float, float, float]]): Dictionary mapping
            place ID to :math:`(x, y, r)` tuple.
    
    Returns:
        List[Tuple[int, float, float]]: sorted_arcs
            List of tuples containing place ID, start, end.
    '''
    
    if len(circles) == 1:
        return [(list(circles.keys())[0], -180.0, 180.0)]
    
    angles = {c: [] for c in circles}
    equals = dict()
    
    # Compute points of intersection for all pairs of circles
    # Ref. http://paulbourke.net/geometry/circlesphere/
    for c1, c2 in list(combinations(circles, 2)):
        xc1, yc1, r1 = circles[c1]
        xc2, yc2, r2 = circles[c2]
        P1 = np.array([xc1, yc1])
        P2 = np.array([xc2, yc2])
        
        try:
            d = np.linalg.norm(P2 - P1)
            assert d != 0
            assert d <= r1 + r2
            assert d >= np.abs(r1 - r2)
        except AssertionError:
            continue
        else:
            a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            h = np.sqrt(r1 ** 2 - a ** 2)
            x, y = P1 + (a / d) * (P2 - P1)
            
            # Points of intersection
            xa, ya = x + (h / d) * (yc2 - yc1), y - (h / d) * (xc2 - xc1)
            xb, yb = x - (h / d) * (yc2 - yc1), y + (h / d) * (xc2 - xc1)
            
            # Angles
            ta1 = np.round(np.degrees(np.arctan2(ya - yc1, xa - xc1)), 7)
            tb1 = np.round(np.degrees(np.arctan2(yb - yc1, xb - xc1)), 7)
            ta2 = np.round(np.degrees(np.arctan2(ya - yc2, xa - xc2)), 7)
            tb2 = np.round(np.degrees(np.arctan2(yb - yc2, xb - xc2)), 7)
            angles[c1].extend([ta1, tb1])
            angles[c2].extend([ta2, tb2])
            equals.update({(c1, ta1): (c2, ta2), (c2, ta2): (c1, ta1),
                           (c1, tb1): (c2, tb2), (c2, tb2): (c1, tb1)})
    
    # Extract outer arcs
    outer_arcs = []
    for c, t in angles.items():
        t.sort()
        t.append(np.round(t[0] + 360.0, 7))
        xc, yc, rc = circles[c]
        for k in range(len(t) - 1):
            theta = np.average([t[k], t[k+1]])
            X = xc + rc * np.cos(np.radians(theta))
            Y = yc + rc * np.sin(np.radians(theta))
            for circ in set(circles).difference({c}):
                x, y, r = circles[circ]
                if (X - x) ** 2 + (Y - y) ** 2 <= r ** 2:
                    break
            else:
                outer_arcs.append((c, t[k], t[k+1]))
    
    # Sort outer arcs
    sorted_arcs = [outer_arcs.pop(0)]
    while len(outer_arcs) > 0:
        last_point = sorted_arcs[-1][::2]
        try:
            next_point = equals[last_point]
        except KeyError:
            next_point = equals[
                (last_point[0], np.round(last_point[1] - 360.0, 7))]
        finally:
            sorted_arcs.append(outer_arcs.pop(
                [x[:2] for x in outer_arcs].index(next_point)))
    
    return sorted_arcs


@dataclass
class Place(Element):
    id: int
    name: str
    x: float
    y: float
    group_id: int = None
    
    def geometry(self):
        return point_geometry(self.x, self.y)


@dataclass
class Circle:
    x: float
    y: float
    r: float
    
    def __iter__(self):
        return iter((self.x, self.y, self.r))
    
    def outer_arcs(self, other: 'Circle') -> List[Tuple['Circle', float, float]]:
        """
        Returns points of intersection between `self` and `other`.
        
        Parameters
        ----------
        other : :class:`Circle`
            Another circle.
        
        Returns
        -------
        arcs : List[Tuple[:class:`Circle`, float, float]]
        
        Reference
        ---------
        http://paulbourke.net/geometry/circlesphere/
        """
        dx, dy = other.x - self.x, other.y - self.y
        d = np.linalg.norm([dx, dy])
        try:
            assert d != 0
            assert d <= self.r + other.r
            assert d >= np.abs(self.r - other.r)
        except AssertionError:
            return [(self, 0.0, 2*np.pi), (other, 0.0, 2*np.pi)]
        else:
            arcs = []
            a = (self.r**2 - other.r**2 + d**2) / (2*d)
            b = (d**2 - self.r**2 + other.r**2) / (2*d)
            # self
            t, dt = np.arctan2(dy, dx), np.arccos(a/self.r)
            t1, t2 = t + dt, t - dt
            if t2 < t1:
                t2 += 2 * np.pi
            arcs.append((self, t1, t2))
            # other
            t, dt = t + np.pi, np.arccos(b/other.r)
            t1, t2 = t + dt, t - dt
            if t2 < t1:
                t2 += 2 * np.pi
            arcs.append((other, t1, t2))
            return arcs
    
    def points(self, N: int = 720) -> np.ndarray:
        t = np.linspace(0, 2*np.pi, N)
        return np.array([self.x, self.y]) + self.r * np.column_stack((np.cos(t), np.sin(t)))


@dataclass
class PlaceGroup(Element):
    id: int
    r: float
    members: List[Place] = field(repr=False, default_factory=list)
    
    def __iter__(self):
        return iter(self.members)
    
    def __len__(self):
        return len(self.members)
    
    def append(self, member: Place) -> None:
        self.members.append(member)
    
    def arcs(self) -> List[Tuple[int, float, float]]:
        circles = {m.id: (m.x, m.y, self.r) for m in self.members}
        return outer_arcs(circles)

    def geometry(self):
        arcs = self.arcs()
        points = []
        circles = {m.id: (m.x, m.y, self.r) for m in self.members}
        for (pid, t1, t2) in arcs:
            x, y, r = circles[pid]
            t = np.radians(np.arange(t1, t2, 0.5))
            C, S = np.cos(t), np.sin(t)
            points.append(np.column_stack((x + r * C, y + r * S)))
        points = np.vstack(points)
        return polygon_geometry(points)
    
    @property
    def member_ids(self):
        return [m.id for m in self.members]


class PlaceData:
    '''
    Class for representing place data.
    
    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Frame summarizing place dataset with index 'id' and columns
        ['name', 'x', 'y'].
    '''
    
    DEFAULT_NAME = 'places'

    def __init__(self, df: pd.DataFrame, layer = None):
        self._df = df
        self.layer = layer
    
    def __iter__(self):
        self._i = -1
        self.__df = self._df.reset_index().astype('object')
        return self
    
    def __next__(self):
        try:
            self._i += 1
            return Place(*self.__df.iloc[self._i].tolist())
        except IndexError:
            raise StopIteration

    # == Constructors ========================================================

    @classmethod
    def from_csv(cls, path_to_csv: str, crs: int = 4326) -> 'PlaceData':
        '''
        Instantiates :class:`PlaceData` from a CSV file.
        
        The CSV file should have the columns 'id', 'name', 'x', and 'y'.
        
        Parameters
        ----------
        path_to_csv : str
            Path to CSV file.
        crs : int
            EPSG code of CRS in which place locations are represented.
        
        Returns
        -------
        :class:`PlaceData`
        '''
        df = read_csv(path_to_csv, 'id', ['name', 'x', 'y'])
        df.attrs['crs'] = crs
        return cls(df)

    # == Descriptions ========================================================

    @property
    def crs(self):
        ''':class:`CRS`: EPSG code of CRS in which place locations are
        represented.
        
        Raises
        ------
        KeyError
            If :attr:`crs` attribute is missing from the data frame.
        EPSGError
            If data frame attribute :attr:`crs` is an invalid EPSG code.
        '''
        return CRS(self.df.attrs['crs'])

    @property
    def df(self):
        return self._df

    def groups(self, r: float = 500.0) -> List[PlaceGroup]:
        """
        Returns list of place groups.
        
        Parameters
        ----------
        r : :obj:`float`, optional
            Radius for identifying neighboring places. The default is 500.0.
        
        Returns
        -------
        list of :class:`PlaceGroup`
        """
        tree = cKDTree(self.df[['x', 'y']].to_numpy())
        neighbors = {i: set() for i in range(len(self.df))}
        for (i, j) in tree.query_pairs(r):
            neighbors[i].add(j)
            neighbors[j].add(i)
        df = self.df.reset_index()
        groups = []
        for i, members in enumerate(ccl(neighbors)):
            group = PlaceGroup(i, r)
            for m in members:
                group.append(Place(*df.iloc[m].tolist(), i))
            groups.append(group)
        return groups

    # == Iteration ===========================================================

    def generate(self, report: Callable[[float], None] = lambda x: None
                 ) -> Generator[QgsFeature, None, None]:
        '''
        Yields place features.
        
        Parameters
        ----------
        report : :obj:`Callable[[float], None]`, optional
            Function for reporting generator progress.
        
        Yields
        ------
        :class:`qgis.core.QgsFeature`
        '''
        N = len(self.df)
        for i, place in enumerate(self, 1):
            report(i/N*100)
            yield place.feature(i)

    # == Manipulation ========================================================

    def transform(self, dst):
        """
        Transforms place coordinates to a new coordinate system.
        
        Parmeters
        ---------
        dst : int
            EPSG code of destination CRS.
        
        Raises
        ------
        EPSGError
            If `dst` is not a valid EPSG code.
        """
        if dst == self.crs.epsg:
            return
        coords = self.crs.transform(self.df[['x', 'y']].to_numpy(), dst)
        self.df['x'] = coords[:,0]
        self.df['y'] = coords[:,1]
        self.df.attrs['crs'] = dst
        self.layer = None

    # == Output ==============================================================

    def render(self, groupname: str = '', index: int = 0, **kwargs):
        if self.layer is None:
            self.layer = PlaceLayer.create(self.crs.epsg)
            self.layer.render(**kwargs)
        self.layer.populate(self.generate)
        if len(kwargs) > 0:
            self.layer.render(**kwargs)
        self.layer.add(groupname, index)
    
    def to_gpkg(self, gpkg: Union[str, GpkgData], layername: str = 'places'
                ) -> None:
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected 'str' or 'GpkgData' for argument 'gpkg'")
        
        if self.layer is None:
            gpkg.write_features(layername, self.generate, self.crs,
                                PlaceLayer.fields, 'point')
        else:
            self.layer.save(gpkg, layername)


class AreaData:
    
    def __init__(self, gr: List[PlaceGroup]):
        self.gr = gr
        self.layer = None
    
    def generate(self, report: Callable[[float], None] = lambda x: None
                 ) -> Generator[QgsFeature, None, None]:
        N = len(self.gr)
        for i, gr in enumerate(self.gr, 1):
            report(i/N*100)
            yield gr.feature(i)

    def render(self, groupname: str = '', index: int = 0, **kwargs):
        if self.layer is None:
            self.layer = AreaLayer.create(6677)
            self.layer.render(**kwargs)
        self.layer.populate(self.generate)
        if len(kwargs) > 0:
            self.layer.render(**kwargs)
        self.layer.add(groupname, index)
    
    def to_gpkg(self, gpkg: Union[str, GpkgData], layername: str = 'areas'
                ) -> None:
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            raise TypeError("expected 'str' or 'GpkgData' for argument 'gpkg'")
        
        if self.layer is None:
            gpkg.write_features(layername, self.generate, self.crs,
                                AreaLayer.fields, 'polygon')
        else:
            self.layer.save(gpkg, layername)


class PlaceLayer(Layer):

    @classmethod
    def create(cls, crs: int, layername: str = 'places') -> 'PlaceLayer':
        return super().create('point', crs, layername, Place.fields())

    @staticmethod
    def renderer(**kwargs):
        kwargs.setdefault('color', (168,50,72))
        kwargs.setdefault('size', 3.0)
        return single_marker_renderer(**kwargs)


class AreaLayer(Layer):
    
    @classmethod
    def create(cls, crs: int, layername: str = 'areas') -> 'AreaLayer':
        return super().create('polygon', crs, layername, PlaceGroup.fields())

    @staticmethod
    def renderer(**kwargs):
        kwargs.setdefault('color', (168,50,72))
        kwargs.setdefault('opacity', 0.2)
        return single_fill_renderer(**kwargs)
