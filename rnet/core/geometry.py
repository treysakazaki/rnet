from itertools import product
from typing import List, Tuple
import numpy as np


def edge_length(coords: np.ndarray, tr: Tuple[int, int] = None) -> float:
    '''
    Returns the length of the edge defined by `coords`.
    
    Parameters
    ----------
    coords : numpy.ndarray, shape (N,2) or (N,3)
        Array of size (N, 2) for a two-dimensional edge, or (N, 3) for a
        three-dimensional edge, where N is the number of points that define
        the edge geometry.
    tr : Tuple[int, int], optional
        2-tuple containing the EPSG codes of the source and destination CRSs,
        if the desired length units is of a different CRS than the one in
        which `coords` are represented. If None, no transformation is applied.
        The default is None.
            
    Returns
    -------
    float
    '''
    if tr is None:
        return float(np.sum(np.linalg.norm(coords[1:] - coords[:-1], axis=1)))


def circle_points(x: float, y: float, r: float, N: int = 720) -> np.ndarray:
    '''
    Returns points that form the perimeter of a circle.
    
    Parameters
    ----------
    x : float
        `x`-coordinate of cirlce center.
    y : float
        `y`-coordinate of circle center.
    r : float
        Circle radius.
    N : int
        Number of points along perimeter. The default is 720.
    
    Returns
    -------
    :class:`numpy.ndarray`, shape (N,2)
    '''
    angles = np.linspace(0, 2*np.pi, N)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def indices_in_circle(xdata: np.ndarray, ydata: np.ndarray, x: float,
                      y: float, r: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Parameters
    ----------
    xdata : :obj:`numpy.ndarray`, shape (nx,)
        `x`-coordinate data.
    ydata : :obj:`numpy.ndarray`, shape (ny,)
        `y`-coordinate data.
    x : float
        `x`-coordinate.
    y : float
        `y`-coordinate.
    r : float
        Circle radius.
    '''
    # Filter by square
    xindices = np.flatnonzero((x-r<=xdata) & (xdata<=x+r))
    yindices = np.flatnonzero((y-r<=ydata) & (ydata<=y+r))
    assert len(xindices) > 0
    assert len(yindices) > 0
    # Filter by circle
    coords = np.array(list(product(ydata[yindices], xdata[xindices])))
    coords[:,:] = coords[:,[1,0]]
    dists = np.linalg.norm(coords - np.array([x,y]), axis=1)
    indices = np.flatnonzero(dists <= r)
    xi, yi = np.unravel_index(indices, (len(xindices),len(yindices)), 'F')
    return xindices[xi], yindices[yi], dists[indices]


def segmentize(vertex_coords: np.ndarray, *indices: int) -> List[np.ndarray]:
    '''
    Breaks a path defined by a sequence of vertex coordinates at the specified
    break points.
    
    Parameters
    ----------
    vertex_coords : numpy.ndarray, shape (N,2)
        Array of shape (N, 2), where N is the total number of vertices.
    *indices : tuple[int]
        The break points.
    
    Returns
    -------
    list[numpy.ndarray]
        List of length :math:`M+2`, where :math:`M` is the number of `indices`. 
    '''
    indices = [0] + list(indices) + [len(vertex_coords)]
    coords = []
    for k in range(len(indices) - 1):
        coords.append(vertex_coords[indices[k]:indices[k+1]+1])
    return coords


def buffer(vertex_coords: np.ndarray, dist: float, *indices: int
           ) -> List[np.ndarray]:
    '''
    Returns the coordinates that form a buffer around the path described by
    the given vertex coordinates.
    
    Parameters
    ----------
    vertex_coords : numpy.ndarray
        Array of size (N, 2), where N is the number of vertices in the path.
    dist : float
        Buffer distance.
    *indices : :obj:`tuple[int]`, optional
        Break points at which the path is split.
    
    Returns
    -------
    buffers : List[numpy.ndarray]
        List of arrays, each of which contains the coordinates of the points
        which define the buffers.
    
    Raises
    ------
    AssertionError
        If coordinate pairs are not two-dimensional, or if less than two points
        are given.
    '''
    N, M = vertex_coords.shape
    assert N >= 2  # require at least two points
    assert M == 2  # require points are two-dimensional
    
    link_coords = np.column_stack([vertex_coords[:-1], vertex_coords[1:]])
    
    dx = link_coords[:,2] - link_coords[:,0]
    dy = link_coords[:,3] - link_coords[:,1]
    angles = np.arctan2(dy, dx)
    
    pointsl = []
    pointsr = []
    
    basis = np.array([-dy[0], dx[0]])
    basis = basis / np.linalg.norm(basis)
    pointsl.append(tuple(vertex_coords[0] + basis * dist))
    pointsr.append(tuple(vertex_coords[0] - basis * dist))
    
    for k in range(1, N-1):
        theta = np.mean([np.pi + angles[k-1], angles[k]])
        basis = -np.array([np.cos(theta), np.sin(theta)])
        pointsl.append(tuple(vertex_coords[k] + dist/np.sin(angles[k-1]-theta)*basis))
        pointsr.append(tuple(vertex_coords[k] - dist/np.sin(angles[k-1]-theta)*basis))
    
    basis = np.array([-dy[-1], dx[-1]])
    basis = basis / np.linalg.norm(basis)
    pointsl.append(tuple(vertex_coords[-1] + basis * dist))
    pointsr.append(tuple(vertex_coords[-1] - basis * dist))
    
    if len(indices) == 0:
        return [np.concatenate([pointsl, np.flip(pointsr, axis=0), [pointsl[0]]])]
    
    else:
        indices = [0] + list(indices) + [N]
        buffers = []
        for k in range(len(indices) - 1):
            buffers.append(np.concatenate([
                pointsl[indices[k]:indices[k+1]+1],
                np.flip(pointsr[indices[k]:indices[k+1]+1], axis=0),
                [pointsl[indices[k]]]
                ]))
        return buffers
