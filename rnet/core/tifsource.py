import os
from typing import Tuple

import numpy as np

try:
    from osgeo import gdal
except:
    pass

from rnet.core.source import Source
from rnet.utils import abspath, gdal_merge


__all__ = ['TifSource']


class TifSource(Source):
    '''
    Class for representing GeoTIFF data.
    
    Parameters
    ----------
    fp : str
        Path to TIF file.
    
    Examples
    --------
    Instantiate from a single TIF file:
        
        >>> data = TifData.from_tif(<path/to/tif>)
    
    or from multiple TIF files using the :meth:`from_tifs` method:
        
        >>> data = TifData.from_tifs(<path/to/tif_1>, <path/to/tif_2>)
    '''
    
    ext = '.tif'
    
    def __post_init__(self):
        super().__post_init__()
        self.source = gdal.Open(self.fp)
    
    @classmethod
    def from_tifs(cls, *tif_paths: str, directory: str = None,
                  filename: str = 'merged.tif') -> 'TifSource':
        '''
        Instantiate based on data taken from multiple TIF files.
        
        Parameters
        ----------
        *tif_paths : tuple[str]
            Paths to TIF files.
        
        Keyword arguments
        -----------------
        directory : :obj:`str`, optional
            Output directory. If None, then the common path among `tif_paths`
            is used. The default is None.
        filename : :obj:`str`, optional
            Output file name. The default is 'merged.tif'.
        
        Returns
        -------
        :class:`TifData`
        
        See also
        --------
        :meth:`merge_tifs()`
            Merges multiple TIF files.
        '''
        fp = cls.merge_tifs(*tif_paths, directory=directory, filename=filename)
        return cls(fp)
    
    @property
    def geotransform(self):
        '''
        Tuple[float, float, float, float, float, float]: Geotransform of
        source.
        
        References
        ----------
        https://gdal.org/tutorials/geotransforms_tut.html
        '''
        return self.source.GetGeoTransform()
    
    @staticmethod
    def merge_tifs(*tif_paths: str, directory: str, filename: str,
                   overwrite: bool = True) -> str:
        '''
        Merges multiple TIF files.
        
        Parameters
        ----------
        *tif_paths : tuple[str]
            Paths to TIF files.
        
        Keyword arguments
        -----------------
        directory : str
            Output directory. If None, then the common path among `tif_paths`
            is used.
        filename : str
            Output file name.
        overwrite : :obj:`bool`, optional
            Whether to overwrite existing files. The default is True.
        
        Returns
        -------
        output_path : str
            Path to merged TIF file.
        
        Raises
        ------
        FileExistsError
            If a file with the `filename` already exists in `directory`.
        FileNotFoundError
            If any of the `tif_paths` does not exist.
        TypeError
            If any of the `tif_paths` does not have the ``.tif`` extension.
        '''
        if directory is None:
            directory = os.path.commonpath(tif_paths)
        output_path = os.path.join(directory, filename)
        
        if os.path.isfile(output_path):
            if overwrite:
                os.remove(output_path)
            else:
                raise FileExistsError(output_path)
        
        paths = []
        for tif_path in tif_paths:
            paths.append(abspath(tif_path))
            if os.path.splitext(tif_path)[1] != '.tif':
                raise TypeError('expected TIF file')
        gdal_merge.main(['', '-o', output_path, *paths])
        return output_path
    
    @property
    def nx(self):
        '''int: Number of pixels in horizontal direction.'''
        return self.source.RasterXSize
    
    @property
    def ny(self):
        '''int: Number of pixels in vertical direction.'''
        return self.source.RasterYSize
    
    def origin(self) -> Tuple[float, float]:
        '''
        Returns :math:`(x, y)` coordinates of top-left of source area.
        
        Returns
        -------
        tuple[float, float]
        '''
        transform = self.geotransform
        return (transform[0], transform[3])
    
    def pixelsize(self) -> Tuple[float, float]:
        '''
        Returns pixel size.
        
        Returns
        -------
        tuple[float, float]
            Pixel size, :math:`(dx, dy)`.
        '''
        transform = self.geotransform
        return (transform[1], transform[5])
    
    def x(self) -> np.ndarray:
        '''
        Returns `x`-coordinates from left to right.
        
        Returns
        -------
        :class:`numpy.ndarray`, shape (nx,)
        '''
        x0, _ = self.origin()
        dx, _ = self.pixelsize()
        return np.arange(x0, x0 + self.nx * dx, dx)
    
    def y(self) -> np.ndarray:
        '''
        Returns `y`-coordinates from top to bottom.
        
        Returns
        -------
        :class:`numpy.ndarray`, shape(ny,)
        '''
        _, y0 = self.origin()
        _, dy = self.pixelsize()
        return np.arange(y0, y0 + self.ny * dy, dy)
    
    def z(self) -> np.ndarray:
        '''
        Returns array containing elevation data.
        
        Returns
        -------
        :class:`numpy.ndarray`, shape (ny, nx)
        '''
        return self.source.GetRasterBand(1).ReadAsArray()
