import numpy as np
from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsSingleSymbolRenderer
    )
from rnet.utils.symbols import (
    marker_symbol,
    line_symbol,
    fill_symbol
    )


def single_marker_renderer(**kwargs):
    '''
    Returns single symbol renderer with marker symbol.
    
    Parameters
    ----------
    **kwargs : dict, optional
        See keyword arguments for :func:`rnet.utils.symbols.marker_symbol`.
    
    Returns
    -------
    qgis.core.QgsSingleSymbolRenderer
    '''
    return QgsSingleSymbolRenderer(marker_symbol(**kwargs))


def single_line_renderer(**kwargs):
    '''
    Returns single symbol renderer with line symbol.
    
    Parameters
    ----------
    **kwargs : dict, optional
        See keyword arguments for :func:`rnet.utils.symbols.line_symbol`.
    
    Returns
    -------
    qgis.core.QgsSingleSymbolRenderer
    '''
    return QgsSingleSymbolRenderer(line_symbol(**kwargs))


def categorized_renderer(fieldname):
    '''
    Returns categorized symbol renderer.
    
    Parameters
    ----------
    fieldname : str
        Name of the field which the categories will be matched against.

    Returns
    -------
    qgis.core.QgsCategorizedSymbolRenderer
    '''
    return QgsCategorizedSymbolRenderer(fieldname)


def point_category(value, **kwargs):
    '''
    Adds a new cateogry with point geometry to an existing categorized
    renderer.

    Parameters
    ----------
    value : str
        The value corresponding to the new category.
    **kwargs : dict, optional
        See keyword arguments for :func:`rnet.utils.symbols.marker_symbol`.

    Returns
    -------
    qgis.core.QgsRendererCategory
    '''
    return QgsRendererCategory(value, marker_symbol(**kwargs), value)


def line_category(value, **kwargs):
    '''
    Returns a renderer category with line geometry.
    
    Parameters
    ----------
    value : str
        The value corresponding to the new category.
    **kwargs : dict, optional
        See keyword arguments for :func:`rnet.utils.symbols.line_symbol`.
    
    Returns
    -------
    qgis.core.QgsRendererCategory
    '''
    return QgsRendererCategory(value, line_symbol(**kwargs), value)


def fill_category(value, **kwargs):
    '''
    Returns a renderer category with polygon geometry.
    
    Parameters
    ----------
    value : str
        The value corresponding to the new category.
    **kwargs : dict, optional
        See keyword arguments for :func:`rnet.utils.symbols.fill_symbol`.

    Returns
    -------
    qgis.core.QgsRendererCategory
    '''
    return QgsRendererCategory(value, fill_symbol(**kwargs), value)


def categorized_road_renderer(tags, *, color1=(210,210,210),
                              color2=(30,30,30), width1=0.2, width2=0.5):
    '''
    Returns a categorized renderer for rendering roads.
    
    Parameters
    ----------
    tags : List[str]
        List of road tags for minor to major.
    
    Keyword arguments
    -----------------
    color1 : Tuple[int, int, int], optional
        RGB definition for roads with the tag at index 0. The default is
        (210, 210, 210).
    color2 : Tuple[int, int, int], optional
        RGB definition for roads with the tag at index -1. The default is
        (30, 30, 30).
    width1 : float, optional
        Line width for roads with the tag at index 0. The default is 0.2.
    width2 : float, optional
        Line width for roads with the tag at index -1. The default is 0.6.

    Returns
    -------
    qgis.core.QgsCategorizedSymbolRenderer
    '''
    N = len(tags)
    colors = iter(np.linspace(color1, color2, N).astype(int).tolist())
    widths = iter(np.linspace(width1, width2, N).astype(float).tolist())
    renderer = categorized_renderer('tag')
    for tag in tags:
        color = next(colors)
        width = next(widths)
        renderer.addCategory(line_category(tag, color=color, width=width))
    return renderer

