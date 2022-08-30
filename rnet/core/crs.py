from dataclasses import dataclass, field
from typing import Union, Callable
import numpy as np
try:
    from osgeo.osr import (
        CoordinateTransformation,
        SpatialReference
        )
    from qgis.core import QgsCoordinateReferenceSystem, QgsTask
except:
    pass
from rnet.utils.taskmanager import create_and_queue


__all__ = ['transform', 'CRS']


class Error(Exception):
    pass


class EPSGError(Error):
    '''
    Raised if EPSG code is invalid.
    
    Parameters
    ----------
    epsg : int
        EPSG code.
    '''
    def __init__(self, epsg: int):
        super().__init__(f'invalid EPSG code {epsg}')


def validate(crs: Union[int, QgsCoordinateReferenceSystem]) -> None:
    '''
    Raises CrsError if `crs` is invalid.
    
    Parameters
    ----------
    crs : :obj:`int` or :class:`qgis.core.QgsCoordinateReferenceSystem`
        EPSG code or :class:`QgsCoordinateReferenceSystem` object.
    
    Raises
    ------
    TypeError
        If `crs` is unexpected type.
    EPSGError
        If EPSG code is invalid.
    '''
    if type(crs) is int:
        epsg = crs
        crs = QgsCoordinateReferenceSystem.fromEpsgId(crs)
    elif isinstance(crs, QgsCoordinateReferenceSystem):
        epsg = int(crs.authid().lstrip('EPSG:'))
    else:
        raise TypeError("arg 'crs' expected 'int' or 'QgsCoordinateReferenceSystem'")
    
    if crs.isValid():
        return
    else:
        raise EPSGError(epsg)


def ct(src: int, dst: int) -> CoordinateTransformation:
    '''
    Returns :class:`osgeo.osr.CoordinateTransformation` object from a
    source to a destination CRS.
    
    Parameters
    ----------
    srs : int
        EPSG code of source CRS.
    dst : int
        EPSG code of destination CRS.
    
    Returns
    -------
    :class:`osgeo.osr.CoordinateTransformation`
    
    Raises
    ------
    EPSGError
        If either `srs` or `dst` EPSG code is invalid.
    
    Example
    -------
        >>> coords = np.array([[140.1, 35.5],
        ...                    [140.2, 35.5]])
        >>> tr = ct(4326, 6677)
        >>> transformed = np.array(tr.TransformPoints(coords[:,[1,0]]))
        >>> transformed[:,[1,0]]
        array([[ 24192.11362008, -55438.94763998],
               [ 33264.19136566, -55409.83074051]])
    '''
    src = CRS(src)
    dst = CRS(dst)
    return CoordinateTransformation(src.ref, dst.ref)


def transform(coords: np.ndarray, src: int, dst: int) -> np.ndarray:
    '''
    Transforms `coords` from source CRS to destination CRS.
    
    Parameters
    ----------
    coords : :class:`numpy.ndarray`, shape (N, 2)
        Array of two-dimensional coordinates to be transformed.
    src : int
        EPSG code of source CRS.
    dst : int
        EPSG code of destination CRS.
    
    Returns
    -------
    transformed : :class:`numpy.ndarray`, shape (N, 2)
        Array of two-dimensional transformed coordinates.
    
    Raises
    ------
    EPSGError
        If either `src` or `dst` EPSG code is invalid.
    '''
    N, M = coords.shape
    if M != 2:
        raise ValueError(f'expected 2D points, received {M} dimensions')
    tr = ct(src, dst)
    transformed = np.array(tr.TransformPoints(coords[:,[1,0]]))
    return transformed[:,[1,0]]


class CRS:
    '''
    Class for representing a coordinate reference system.
    
    Parameters
    ----------
    crs : :obj:`int` or :class:`qgis.core.QgsCoordinateReferenceSystem`
        EPSG code or :class:`QgsCoordinateReferenceSystem` object.
    
    Raises
    ------
    TypeError
        If `crs` is unexpected type.
    EPSGError
        If `epsg` code is invalid.
    
    Attributes
    ----------
    ref : :class:`osgeo.osr.SpatialReference`
        Spatial reference.
    srs : :class:`qgis.core.QgsCoordinateReferenceSystem`
        Coordinate reference system.
    '''
    
    def __init__(self, crs: Union[int, QgsCoordinateReferenceSystem]) -> None:
        validate(crs)
        if type(crs) is int:
            self.epsg = crs
            self.srs = QgsCoordinateReferenceSystem.fromEpsgId(self.epsg)
        elif isinstance(crs, QgsCoordinateReferenceSystem):
            self.epsg = int(crs.authid().lstrip('EPSG:'))
            self.srs = crs
        self.ref = SpatialReference()
        self.ref.ImportFromEPSG(self.epsg)
    
    def __repr__(self):
        return f'CRS(epsg={self.epsg})'
    
    def __str__(self):
        return f'EPSG:{self.epsg}'
    
    @property
    def bounds(self):
        '''Tuple[float, float, float, float]: Rectangular bounds (xmin, ymin,
        xmax, ymax) in degrees.'''
        area = self.ref.GetAreaOfUse()
        return (area.west_lon_degree, area.south_lat_degree,
            area.east_lon_degree, area.north_lat_degree)
    
    def info(self):
        '''
        Prints CRS info to the console.
        '''
        bounds = self.bounds
        
        print(f'EPSG:{self.epsg}',
              f'Name: {self.name!r}',
              f'Reference type: {self.reftype!r}',
              f'Length units: {self.units!r}',
              f'Longitude range: ({bounds[0]}, {bounds[2]})',
              f'Latitude range: ({bounds[1]}, {bounds[3]})',
              sep='\n')
        
        if self.reftype == 'projected':
            bounds = transform(np.array(bounds).reshape(2,2), 4326, self.epsg)
            bounds = bounds.flatten()
            print(f'x range: ({bounds[0]}, {bounds[2]})',
                  f'y range: ({bounds[1]}, {bounds[3]})',
                  sep='\n')
    
    @property
    def name(self):
        '''str: CRS name.'''
        return self.ref.GetName()
    
    @property
    def reftype(self):
        '''str: CRS type, either 'geographic' or 'projected'.
        
        References
        ----------
        https://www.esri.com/arcgis-blog/products/arcgis-pro/mapping/gcs_vs_pcs/
        '''
        if self.ref.IsGeographic():
            return 'geographic'
        elif self.ref.IsProjected():
            return 'projected'
        else:
            raise TypeError('unknown reference type')
    
    def transform(self, coords: np.ndarray, dst: int) -> np.ndarray:
        '''
        Returns coordinates transformed to another CRS.
        
        Parameters
        ----------
        coords : :obj:`numpy.ndarray`, shape (N, 2)
            Array containing the :math:`(x, y)` coordinates of N points.
        dst : int
            EPSG code of the destination CRS.
        
        Returns
        -------
        transformed : :obj:`numpy.ndarray`, shape (N, 2)
            Array containing the transformed :math:`(x, y)` coordinates.
        
        Raises
        ------
        ValueError
            If `coords` are not two-dimensional.
        EPSGError
            If `dst` code is invalid.
        
        See also
        --------
        :func:`transform`
        '''
        return transform(coords, self.epsg, dst)
    
    def transform_task(self, coords: np.ndarray, dst: int,
                       on_finished: Callable[[np.ndarray], None] = lambda x: None
                       ) -> None:
        create_and_queue(TransformTask, self, coords, dst, on_finished)
    
    @property
    def units(self):
        '''str: CRS unit name.'''
        reftype = self.reftype
        if reftype == 'geographic':
            return self.ref.GetAngularUnitsName()
        elif reftype == 'projected':
            return self.ref.GetLinearUnitsName()


class TransformTask(QgsTask):
    '''
    Task for transforming coordinates.
    
    Parameters
    ----------
    crs : :class:`CRS`
        :class:`CRS` instance.
    coords : :class:`numpy.ndarray`, shape (N, 2)
        Array containing the :math:`(x, y)` coordinates to transform.
    dst : int
        EPSG code of destination CRS.
    on_finished : Callable[[np.ndarray], None]
        Function to call when task is finished. Takes array of transformed
        coordinates as the argument.
    '''
    
    def __init__(self, crs: CRS, coords: np.ndarray, dst: int,
                 on_finished: Callable[[np.ndarray], None]) -> None:
        super().__init__('Transforming coordinates')
        self.crs = crs
        self.coords = coords
        self.dst = dst
        self.on_finished = on_finished
    
    def run(self) -> bool:
        self.result = self.crs.transform(self.coords, self.dst)
        return True
    
    def finished(self, success: bool) -> None:
        if success:
            self.on_finished(self.result)


@dataclass
class Coordinates:
    '''
    Class representing two- or three-dimensional point coordinates.
    
    Parameters
    ----------
    coords : :class:`numpy.ndarray`, shape (N, 2) or (N, 3)
        Array containing :math:`(x, y)` or :math:`(x, y, z)` coordinates.
    crs : :class:`CRS`
        CRS in which coordiantes are represented.
    '''
    
    coords: np.ndarray = field(repr=False)
    crs: CRS
    N: int = field(init=False)
    dims: int = field(init=False)
    
    def __post_init__(self):
        self.N, self.dims = self.coords.shape
    
    def transform(self, dst: int) -> None:
        self.coords = self.crs.transform(self.coords[:,[0,1]], dst)
