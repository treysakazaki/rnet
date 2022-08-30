from dataclasses import dataclass
from typing import Generator, Callable, List

import numpy as np

try:
    from qgis.core import QgsTask
except:
    pass

from rnet.core.geometry import indices_in_circle
from rnet.core.tifsource import TifSource
from rnet.utils import abspath, create_and_queue


__all__ = ['ElevationQueryEngine']


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
        
        N = len(coords)
        
        for i, (x, y) in enumerate(coords):
            report(i/N*100)
            if (x, y) in self.queried:
                yield self.queried[(x, y)]
            else:
                try:
                    xi, yi, dists = indices_in_circle(xdata, ydata, x, y, r)
                except AssertionError:
                    if ignore_errors:
                        yield
                    else:
                        raise ValueError(f'({x}, {y}) out of bounds')
                else:
                    d = np.power(dists, p)
                    z = np.array(
                        [zdata[i,j] for (i,j) in np.column_stack([yi,xi])])
                    elev = float(np.sum(z/d) / np.sum(1/d))
                    self.queried[(x, y)] = elev
                    yield elev

    def query_task(self, coords: np.ndarray,
                   on_finished: Callable[[List[float]], None] = lambda x: None
                   ) -> None:
        '''
        Queues :class:`ElevationQueryTask`.
        
        Parameters
        ----------
        coords : :class:`numpy.ndarray`, shape (N, 2)
            Array of coordinates to be transformed.
        on_finished : :obj:`Callable[[List[float]], None]`, optional
            Function to call when task is finished. Takes list of elevations as
            the argument.
        '''
        create_and_queue(ElevationQueryTask, self, coords, on_finished)


class ElevationQueryTask(QgsTask):
    '''
    Task for querying elevations.
    
    Parameters
    ----------
    engine : :class:`ELevationQueryEngine`
        Engine for querying elevations.
    coords : :class:`numpy.ndarray`, shape (N, 2)
        Array containing coordinates whose elevations are to be queried.
    on_finished : Callable[[List[float]], None]
        Function to call when task is finished. Takes list of elevations as the
        argument.
    '''
    
    def __init__(self, engine: ElevationQueryEngine, coords: np.ndarray,
                 on_finished: Callable[[List[float]], None]) -> None:
        super().__init__('Querying elevations')
        self.engine = engine
        self.coords = coords
        self.on_finished = on_finished
    
    def run(self) -> bool:
        self.result = list(self.engine.query(self.coords, report=self.setProgress))
        return True
    
    def finished(self, success: bool) -> None:
        if success:
            self.on_finished(self.result)
