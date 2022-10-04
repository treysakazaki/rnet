from abc import ABC, abstractmethod
from typing import List
try:
    from qgis.core import QgsFeature, QgsGeometry
except:
    pass
from rnet.core.field import Field
from rnet.utils import create_feature


class Element(ABC):
    '''
    Base class for representing elements.
    '''
    
    def attributes(self) -> list:
        '''
        Returns list of element attributes.
        
        Returns
        -------
        list
        '''
        return [getattr(self, name)
                for name, field in self.__dataclass_fields__.items()
                if field.repr]

    def feature(self, fid: int) -> QgsFeature:
        '''
        Returns feature.
        
        Parameters
        ----------
        fid : int
            Feature ID.
        
        Returns
        -------
        :class:`qgis.core.QgsFeature`
        '''
        return create_feature(self.geometry(), [fid] + self.attributes())
    
    @classmethod
    def fields(cls) -> List[Field]:
        '''
        Returns list of dataclass fields.
        
        Returns
        -------
        List[:class:`Field`]
        '''
        return [Field(name, field.type.__name__)
                for name, field in cls.__dataclass_fields__.items()
                if field.repr]
    
    @classmethod
    def field_names(cls, exclude_id: bool = True) -> List[str]:
        names = [f.name for f in cls.fields()]
        return names[1:] if exclude_id else names
    
    @abstractmethod
    def geometry(self) -> QgsGeometry:
        pass
