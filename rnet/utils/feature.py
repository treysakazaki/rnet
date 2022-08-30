from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsPointXY
    )


def create_feature(geom, attrs):
    '''
    Returns :class:`qgis.core.QgsFeature` with given geometry and attributes.
    
    Parameters
    ----------
    geom : :class:`qgis.core.QgsGeometry`
        Feature geometry.
    attrs : List[Any]
        Feature attributes.
    
    Returns
    -------
    qgis.core.QgsFeature
    '''
    feat = QgsFeature()
    feat.setGeometry(geom)
    feat.setAttributes(attrs)
    return feat


def point_geometry(x, y):
    '''
    Returns point geometry.
    
    Parameters
    ----------
    x : float
        `x`-coordinate.
    y : float
        `y`-coordinate.
    
    Returns
    -------
    qgis.core.QgsGeometry
    '''
    return QgsGeometry.fromPointXY(QgsPointXY(x, y))


def point_feature(x, y, attrs):
    '''
    Returns feature with point geometry and given attributes.

    Parameters
    ----------
    x : float
        `x`-coordinate.
    y : float
        `y`-coordinate.
    attrs : List[Any]
        Feature attributes.
    
    Returns
    -------
    qgis.core.QgsFeature
    '''
    feat = QgsFeature()
    feat.setGeometry(point_geometry(x, y))
    feat.setAttributes(attrs)
    return feat


def line_geometry(points):
    '''
    Returns line geometry.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 2) where N is the number of points in the geometry.
    
    Returns
    -------
    qgis.core.QgsGeometry
    '''
    return QgsGeometry.fromPolylineXY([QgsPointXY(*p) for p in points])
    

def line_feature(points, attrs):
    '''
    Returns feature with line geometry and given attributes.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 2) where N is the number of points in the geometry.
    attrs : List[Any]
        Feature attributes.
    
    Returns
    -------
    qgis.core.QgsFeature
    '''
    feat = QgsFeature()
    feat.setGeometry(line_geometry(points))
    feat.setAttributes(attrs)
    return feat


def polygon_geometry(points):
    '''
    Returns polygon geometry.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 2) where N is the number of points in the geometry.
    
    Returns
    -------
    qgis.core.QgsGeometry
    '''
    return QgsGeometry.fromPolygonXY([[QgsPointXY(*p) for p in points]])


def polygon_feature(points, attrs):
    '''
    Returns feature with polygon geometry and given attributes.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (N, 2) where N is the number of points in the geometry.
    attrs : List[Any]
        Feature attributes.
    
    Returns
    -------
    qgis.core.QgsFeature
    '''
    feat = QgsFeature()
    feat.setGeometry(polygon_geometry(points))
    feat.setAttributes(attrs)
    return feat

