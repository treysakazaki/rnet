from dataclasses import dataclass
try:
    from PyQt5.QtCore import QVariant
    from qgis.core import QgsField
except:
    pass


__all__ = ['Field', 'qgsfields']


FIELD_TYPES = {
    'str': QVariant.String,
    'int': QVariant.Int,
    'double': QVariant.Double,
    'float': QVariant.Double,
    'time': QVariant.Time
    }


@dataclass
class Field:
    '''
    Class for representing a field.
    
    Parameters
    ----------
    name : str
        Field name.
    type : {'str', 'int', 'double', 'float', 'time'}
        Field type
    '''
    
    name: str
    type: str
    
    def __post_init__(self):
        if type(self.name) is not str:
            raise TypeError("expected 'str' type for 'name' parameter")
        if self.type not in FIELD_TYPES:
            raise ValueError(f'unknown field type {self.type!r}')


def qgsfields(fields, rtype=list):
    '''
    Returns layer fields in a list or as a :class:`qgis.core.QgsFields`
    object.
    
    Parameters
    ----------
    fields : list[tuple[str, str]]
        List of 2-tuples containing field name and field type. Supported field
        types are {'str', 'int', 'double', 'float', 'time'}.
    rtype : {list, qgis.core.QgsFields}
        Return type.
    
    Returns
    -------
    List[:class:`lqgis.core.QgsField`] or :class:`qgis.core.QgsFields`
    '''
    result = rtype()
    result.append(QgsField('fid', QVariant.Int))
    for f in fields:
        result.append(QgsField(f.name, FIELD_TYPES[f.type]))
    return result
