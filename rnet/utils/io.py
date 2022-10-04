import os
import pandas as pd
from typing import List, Union
try:
    from qgis.core import QgsProject
except:
    pass


__all__ = ['abspath', 'relpath', 'read_csv']


def cwd() -> str:
    '''
    Returns the current working directory.
    
    Returns
    -------
    str
    '''
    try:
        os.chdir(QgsProject.instance().homePath())
    except:
        pass
    finally:
        return os.getcwd()


def abspath(path: str, validate: bool = True) -> str:
    '''
    Returns the absolute path of `path`. If `path` is relative, then the
    current working directory is prepended.
    
    Parameters
    ----------
    path : str
        Path.
    validate : :obj:`bool`, optional
        If True, existence of the path is verified. The default is True.
    
    Returns
    -------
    path : str
        The absolute path.
    
    Raises
    ------
    FileNotFoundError
        If `validate` is True, but the path does not exist.
    '''
    abspath, _ = get_paths(path)
    if validate and not os.path.exists(abspath):
        raise FileNotFoundError(path)
    return abspath


def relpath(path: str, validate: bool = True) -> str:
    '''
    Returns the relative path of `path`. If `path` is absolute, then the
    current working directory is stripped.
    
    Parameters
    ----------
    path : str
        Path.
    validate : :obj:`bool`, optional
        If True, existence of the path is verified. The default is True.
    
    Returns
    -------
    relpath : str
        The relative path.
    
    Raises
    ------
    FileNotFoundError
        If `validate` is true, but the path does not exist.
    '''
    abspath, relpath = get_paths(path)
    if validate and not os.path.exists(abspath):
        raise FileNotFoundError(path)
    return relpath


def get_paths(path):
    '''
    Returns absolute and relative paths.
    
    Parameters
    ----------
    path : str
        Path.
    '''
    cwd_ = cwd()
    if os.path.isabs(path):
        abspath = path
        relpath = os.path.relpath(path, cwd_)
    else:
        abspath = os.path.abspath(path)
        relpath = path
    return abspath, relpath


def names_with_ext(ext, directory='') -> List[str]:
    '''
    Returns list of file names in `directory` with the specified extension.
    
    Parameters
    ----------
    ext : str
        File extension.
    directory : str
        Directory path.
    
    Returns
    -------
    list[str]
    '''
    filenames = os.listdir(abspath(directory))
    return list(filter(lambda path: os.path.splitext(path)[1] == ext, filenames))


def read_csv(path_to_csv: str, index: Union[str, List[str]],
             columns: List[str]) -> pd.DataFrame:
    """
    Read CSV file with column headers in the first row.
    
    Parameters
    ----------
    path_to_csv : str
        Path to CSV file.
    index : :obj:`str` or :obj:`List[str]`
        Name(s) of column(s) used for index.
    columns : List[str]
        List of column names.
    
    Returns
    -------
    :class:`pandas.DataFrame`
        Frame with single or multi index.
    """
    path_to_csv = abspath(path_to_csv)
    df = pd.read_csv(path_to_csv, header=0, index_col=index,
                     usecols=[index]+columns)
    df.index.name = index
    df = df[columns]
    return df
