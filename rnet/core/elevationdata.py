from dataclasses import dataclass
from typing import Generator, Callable

import numpy as np

from rnet.core.tifsource import TifSource
from rnet.utils import abspath


__all__ = ['ElevationQueryEngine']


def idw_query(xdata: np.ndarray, ydata: np.ndarray, zdata: np.ndarray,
              points: np.ndarray, r: float = 1e-3, p: int = 2) -> np.ndarray:
    '''
    Parameters
    ----------
    xdata : :class:`numpy.ndarray`, shape (nx,)
        X data from left to right.
    ydata : :class:`numpy.ndarray`, shape (ny,)
        Y data from top to bottom.
    zdata : :class:`numpy.ndarray`, shape (nx, ny)
        Z data with origin at top-left.
    points : :class:`numpy.ndarray`, shape (N, 2)
        Two-dimensional points to query.
    
    Returns
    -------
    :class:`numpy.ndarray`, shape (N,)
        Elevations.
    '''
    nx = len(xdata)
    ny = len(ydata) 
    dx = xdata[1] - xdata[0]
    dy = ydata[1] - ydata[0]
    indices_x = np.searchsorted(xdata, points[:,0])
    indices_y = ny - np.searchsorted(ydata[::-1], points[:,1]) - 1
    dx = int(r/dx) + 1
    dy = int(-r/dy) + 1
    elevs = []
    for p, xi, yi in zip(points, indices_x, indices_y):
        left, right = max(0,xi-dx), min(xi+dx+1,nx)
        top, bottom = max(0,yi-dy), min(yi+dy+1,ny)
        z = zdata[top:bottom,left:right]  # Elevations of nearby points
        xs, ys = np.meshgrid(xdata[left:right]-p[0], ydata[top:bottom]-p[1])
        d = np.sqrt(xs**2 + ys**2)  # Distances to nearby points
        elev = float(np.sum(z/d) / np.sum(1/d))
        elevs.append(elev)
    return np.array(elevs)


@dataclass
class ElevationData:
    '''
    Data class for representing elevation data.
    
    Parameters
    ----------
    crs : int
        EPSG code representing the CRS in which :math:`(x, y)` coordinates are
        represented.
    xdata : numpy.ndarray, shape (nx,)
        One-dimensional array containing `x` ticks.
    ydata : numpy.ndarray, shape (ny,)
        One-dimensional array containing `y` ticks.
    zdata : numpy.ndarray, shape (ny, nx)
        Two-dimensional array containing `z`-coordinates.
    '''
    
    crs: int
    xdata: np.ndarray
    ydata: np.ndarray
    zdata: np.ndarray
    
    def engine(self, r: float = 0.001, p: int = 2) -> 'ElevationQueryEngine':
        '''
        Returns an instance of :class:`ElevationQueryEngine`.
        
        Parameters
        ----------
        r : float, optional
            Search radius for nearest neighbors. The default is 0.001.
        p : int, optional
            Power setting for IDW interpolation. The default is 2.
        
        Returns
        -------
        :class:`ElevationQueryEngine`
        '''
        return ElevationQueryEngine(self, r, p)
    
    @classmethod
    def from_tif(cls, tif_path: str) -> 'ElevationData':
        '''
        Instantiates :class:`ElevationData` based on data taken from a single
        TIF file.
        
        Parameters
        ----------
        tif_path : str
            Path to TIF file.
        
        Returns
        -------
        :class:`ElevationData`
        '''
        tif = TifSource(tif_path)
        return cls(4326, tif.x(), tif.y(), tif.z())
    
    @classmethod
    def from_tifs(cls, *tif_paths: str, directory: str = None,
                  filename: str = 'merged.tif') -> 'ElevationData':
        '''
        Instantiates :class:`ElevationData` based on data taken from multiple
        TIF files.
        
        Parameters
        ----------
        *tif_paths : Tuple[str]
            Paths to TIF files.
        
        Keyword arguments
        -----------------
        directory : str, optional
            Output directory for the merged TIF file. If None, then the common
            path among `tif_paths` is used. The default is None.
        filename : str, optional
            Output file name for the merged TIF file. The default is
            'merged.tif'.
        '''
        tif_paths = list(map(abspath, tif_paths))
        if len(tif_paths) == 1:
            return cls.from_tif(tif_paths[0])
        else:
            tif = TifSource.from_tifs(
                *tif_paths, directory=directory, filename=filename)
        return cls(4326, tif.x(), tif.y(), tif.z())


class ElevationQueryEngine:
    '''
    Engine for querying elevations.
    
    Parameters
    ----------
    data : :class:`ElevationData`
        Elevation data.
    r : float, optional
        Search radius for nearest neighbors. The default is 0.001.
    p : int, optional
        Power setting for IDW interpolation. The default is 2.
    
    Examples
    --------
    The :class:`ElevationQueryEngine` class may be instantiated from a single
    TIF file using the :meth:`from_tif` method:
        
        >>> eng = rn.ElevationQueryEngine.from_tif(<path/to/tif>)
    
    or from multiple TIF files using the :meth:`from_tifs` method:
        
        >>> eng = rn.ElevationQueryEngine.from_tifs(<path/to/tif_1>, <path/to/tif_2>)
    
    Use the :meth:`query` method to query elevations:
        
        >>> elevs = eng.query(np.array([[140.1, 35.5], [140.2, 35.5]]))
        >>> next(elevs)
        3.837348117799038
        >>> next(elevs)
        71.07172521280032
    
    By default, :obj:`None` is yielded for coordinates outside of the query
    area:
        
        >>> elevs = eng.query(np.array([[135, 35]]))
        >>> z = next(elevs)
        >>> type(z)
        <class 'NoneType'>
    '''
    
    def __init__(self, data: ElevationData, r: float = 0.001, p: int = 2
                 ) -> None:
        self.crs = data.crs
        self.xdata = data.xdata
        self.ydata = data.ydata
        self.zdata = data.zdata
        self.r = r
        self.p = p
        self.queried = {}
    
    def __repr__(self):
        return f'ElevationQueryEngine(r={self.r}, p={self.p})'
    
    @classmethod
    def from_tif(cls, tif_path: str, *, r: float = 0.001, p: int = 2
                 ) -> 'ElevationQueryEngine':
        '''
        Instantiates :class:`ElevationQueryEngine` based on data taken from a
        single TIF file.
        
        Parameters
        ----------
        tif_path : str
            Path to TIF file.
        
        Keyword arguments
        -----------------
        r : float, optional
            Search radius for nearest neighbors. The default is 0.001.
        p : int, optional
            Power setting for IDW interpolation. The default is 2.
        
        Returns
        -------
        :class:`ElevationQueryEngine`
        '''
        data = ElevationData.from_tif(tif_path)
        return cls(data, r, p)
    
    @classmethod
    def from_tifs(cls, *tif_paths: str, directory: str = None,
                  filename: str = 'merged.tif', r: float = 0.001, p: int = 2
                  ) -> 'ElevationQueryEngine':
        '''
        Instantiates :class:`ElevationQueryEngine` based on data taken from
        multiple TIF files.
        
        Parameters
        ----------
        *tif_paths : Tuple[str]
            Paths to TIF files.
        
        Keyword arguments
        -----------------
        directory : str, optional
            Output directory for the merged TIF file. If None, then the common
            path among `tif_paths` is used. The default is None.
        filename : str, optional
            Output file name for the merged TIF file. The default is
            'merged.tif'.
        r : float, optional
            Search radius for nearest neighbors. The default is 0.001.
        p : int, optional
            Power setting for IDW interpolation. The default is 2.
        '''
        data = ElevationData.from_tifs(*tif_paths, directory=directory,
                                       filename=filename)
        return cls(data, r, p)
    
    def query(self, coords: np.ndarray, ignore_errors: bool = True, *,
              report: Callable[[float], None] = lambda x: None
              ) -> Generator[float, None, None]:
        '''
        Yields the elevations at each location in `coords`.
        
        Parameters
        ----------
        coords : :class:`numpy.ndarray`, shape (N, 2)
            Array containing the :math:`(x, y)` coordinates of N points whose
            elevations are to be queried.
        ignore_errors : :obj:`bool`, optional
            If True, errors are raised for coordinate pairs that are out of
            bounds. The default is True.
        
        Keyword arguments
        -----------------
        report : :obj:`Callable[[float], None]`, optional
            Function for reporting progress.
        
        Yields
        ------
        float
        
        Raises
        ------
        ValueError
            If `ignore_errors` is True and a coordinate pair is out of bounds.
        '''
        xdata, ydata, zdata = self.xdata, self.ydata, self.zdata
        r, p = self.r, self.p
        return idw_query(xdata, ydata, zdata, coords, r, p)
