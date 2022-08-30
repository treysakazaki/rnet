import numpy as np
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor
from qgis.core import (
    QgsFillSymbol,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsProperty,
    QgsSimpleMarkerSymbolLayerBase
    )


def qcolor(r, g, b):
    '''
    Returns color based on RGB definition.
    
    Parameters
    ----------
    r : int
        Red color component.
    g : int
        Green color component.
    b : int
        Blue color component.
    
    Returns
    -------
    PyQt5.QtGui.QColor
    '''
    return QColor.fromRgb(r, g, b)


def random_color():
    '''
    Returns a random RGB tuple.
    
    Returns
    -------
    Tuple[int, int, int]
    '''
    return tuple(np.random.choice(range(256), size=3))


def marker_symbol(*, color=None, shape='circle', size=1.0, anglename=None):
    '''
    Returns marker symbol.
    
    Keyword arguments
    -----------------
    color : :obj:`Tuple[int, int, int]`, optional
        RGB definition for marker fill color. If None, then a color is
        randomly generated. The default is None.
    shape : :obj:`str`, optional
        Marker shape. The default is 'circle'.
    size : :obj:`float`, optional
        Marker size. The default is 1.0.
    anglename : :obj:`str`, optional
        Field name for data defined angle. If None, then no rotation is
        applied. The default is None.

    Returns
    -------
    symbol : qgis.core.QgsMarkerSymbol
    '''
    if color is None:
        color = random_color()
    symbol = QgsMarkerSymbol()
    symbol.setColor(qcolor(*color))
    shape = QgsSimpleMarkerSymbolLayerBase.decodeShape(shape)[0]
    symbol.symbolLayer(0).setShape(shape)
    if shape == QgsSimpleMarkerSymbolLayerBase.HalfSquare:
        symbol.symbolLayer(0).setOffset(QPointF(size/4, 0))
    symbol.setSize(size)
    if type(anglename) is str:
        symbol.setDataDefinedAngle(QgsProperty.fromExpression(f'"{anglename}"'))
    return symbol


def line_symbol(*, color=None, width=0.5):
    '''
    Returns line symbol.
    
    Keyword arguments
    -----------------
    color : Tuple[int, int, int], optional
        RGB definition for line color. If None, then a color is randomly
        generated. The default is None.
    width : float, optional
        Line width. The default is 0.5.
    
    Returns
    -------
    symbol : qgis.core.QgsLineSymbol
    '''
    if color is None:
        color = random_color()
    symbol = QgsLineSymbol()
    symbol.setColor(qcolor(*color))
    symbol.setWidth(width)
    return symbol


def fill_symbol(*, color=None, opacity=0.4):
    '''
    Returns fill symbol.
    
    Keyword arguments
    -----------------
    color : Tuple[int, int, int], optional
        RGB definition for line color. If None, then a color is randomly
        generated. The default is None.
    opacity : float, optional
        Fill opacity. The default is 0.4.
    
    Returns
    -------
    symbol : qgis.core.QgsFillSymbol
    '''
    if color is None:
        color = random_color()
    symbol = QgsFillSymbol()
    symbol.setColor(qcolor(*color))
    symbol.setOpacity(opacity)
    return symbol
