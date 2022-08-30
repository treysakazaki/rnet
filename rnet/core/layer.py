from collections import namedtuple
from dataclasses import dataclass
import os
from typing import Callable, Generator, List, Tuple, Union
try:
    from PyQt5.QtCore import QVariant
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsDataProvider,
        QgsFeature,
        QgsFeatureRequest,
        QgsFields,
        QgsProject,
        QgsTask,
        QgsVectorFileWriter,
        QgsVectorLayer,
        QgsWkbTypes
        )
except:
    pass

from rnet.core.crs import CRS
from rnet.core.field import Field, qgsfields
from rnet.utils import (
    abspath,
    relpath,
    create_and_queue,
    simple_labeling
    )


__all__ = ['LayerGroup', 'GpkgData']


Reporter = Callable[[float], None]
FeatureGenerator = Generator[QgsFeature, None, None]


class Location(namedtuple('Location', ['group', 'index'],
                          defaults=(None, None))):
    '''
    Class for representing the location of a layer in the layer tree.
    
    Parameters
    ----------
    group : :class:`LayerGroup`
        Layer group.
    index : int
        Index.
    '''
    __slots__ = ()


def locate(layer_id: str) -> Location:
    '''
    Returns location of layer in the layer tree.
    
    Parameters
    ----------
    layer_id : str
        Layer ID.
    
    Returns
    -------
    :class:`Location`

    Raises
    ------
    LayerNotFoundError
        If layer with given ID is not found in the layer tree.
    '''
    root = QgsProject.instance().layerTreeRoot()
    layer = root.findLayer(layer_id)
    if layer is None:
        raise LayerNotFoundError(f'{layer_id!r}')
    parent = layer.parent()
    group = parent.name()
    index = parent.children().index(layer)
    return Location(LayerGroup(group), index)


class LayerNotFoundError(Exception):
    '''Raised when a layer is not found.
    
    Parameters
    ----------
    name : str, optional
        Layer name.
    '''
    def __init__(self, name: str):
        super().__init__(name)


@dataclass
class Layer:
    '''
    Class representing a vector layer.
    
    Parameters
    ----------
    vl : :class:`qgis.core.QgsVectorLayer`
        Vector layer.
    loc : :class:`Location`, optional
        Named tuple representing location of layer in the layer tree.
    '''
    
    vl: QgsVectorLayer
    
    def __post_init__(self):
        if not isinstance(self.vl, QgsVectorLayer):
            raise TypeError("expected type 'qgis.core.QgsVectorLayer'")
    
    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name!r}>'
    
    def add(self, group: Union[str, 'LayerGroup'] = '', index: int = 0
            ) -> None:
        '''
        Add the layer to the project.
        
        Parameters
        ----------
        group : str or :class:`LayerGroup`, optional
            Group name or :class:`LayerGroup` object, specifying the group to
            which the layer is added. If '', then the layer is added to the 
            layer tree root. The default is ''.
        index : int, optional
            Index at which the layer is added. The default is 0.
        '''
        if type(group) is str:
            group = LayerGroup(group)
        elif isinstance(group, LayerGroup):
            pass
        else:
            raise TypeError("arg 'group' expected 'str' or 'LayerGroup'")
        
        group.insert(self, index)
        self.collapse()
    
    @property
    def added(self) -> bool:
        '''bool: True if layer is in the project layer tree.'''
        try:
            locate(self.id)
        except LayerNotFoundError:
            return False
        else:
            return True
    
    def collapse(self) -> None:
        '''Collapses the layer in the map legend.'''
        self.layer.setExpanded(False)
    
    @classmethod
    def create(cls, geometry: str, crs: int, name: str, fields: List[Field]
               ) -> 'Layer':
        '''
        Returns a temporary layer.
        
        Parameters
        ----------
        geometry : {'point', 'linestring', 'polygon'}
            Layer geometry.
        crs : int
            EPSG code of layer CRS.
        name : str
            Layer name.
        fields : List[Field]
            List of fields.
        
        Returns
        -------
        :class:`Layer`
        
        Raises
        ------
        TypeError
            If an unknown field type is given.
        '''
        vl = QgsVectorLayer(f"{geometry}?crs=epsg:{crs}", name, 'memory')
        vl.dataProvider().addAttributes(qgsfields(fields))
        vl.updateFields()
        return cls(vl)
    
    @property
    def crs(self):
        ''':class:`CRS`: Layer CRS.'''
        return CRS(self.vl.crs())
    
    def expand(self):
        '''Expands the layer in the legend.'''
        self.layer.setExpanded(True)
    
    def features(self, geometry=False):
        num_fields = len(self.field_names)
        req = QgsFeatureRequest()
        req.setFlags(QgsFeatureRequest.NoGeometry)
        req.setSubsetOfAttributes(list(range(1, num_fields + 1)))
        return self.vl.getFeatures(req)
    
    def fields(self) -> List[Field]:
        FIELD_TYPES = {QVariant.String: 'str',
                       QVariant.Int: 'int',
                       QVariant.LongLong: 'int',
                       QVariant.Double: 'float',
                       QVariant.Time: 'time'}
        fields = [f for f in self.vl.fields()]
        return [Field(f.name(), FIELD_TYPES[f.type()]) for f in fields[1:]]
    
    @property
    def field_names(self):
        '''List[str]: List of field names.'''
        return self.vl.fields().names()
    
    @classmethod
    def from_gpkg(cls, path_to_gpkg, layername):
        '''
        Instantiates :class:`Layer` from a sublayer of an existing GPKG.
        
        Parameters:
            path_to_gpkg (str): Path to GPKG.
            layername (str): Layer name.
        
        Returns:
            Layer:
        
        Raises:
            FileNotFoundError: If `path_to_gpkg` does not exist.
            ValueError: If layer with `layername` does not exist.
        '''
        gpkg = GpkgData(path_to_gpkg)
        return cls(gpkg.sublayer(layername))
    
    @classmethod
    def from_map_layer(cls, layername):
        '''
        Instantiates :class:`Layer` from a map layer with the given layer name.
        
        Parameters:
            layername (str): Layer name.
        
        Returns:
            Layer:
        
        Raises:
            ValueError: If a layer with the given name does not exist, or if
                multiple layers have the given name.
        '''
        map_layers = QgsProject.instance().mapLayersByName(layername)
        if len(map_layers) == 1:
            vl = map_layers[0]
            return cls(vl)
        else:
            raise ValueError(f'multiple layers found with name {layername!r}')
    
    @classmethod
    def from_ml(cls, layername):
        '''Alias for the :meth:`from_map_layer` method.'''
        return cls.from_map_layer(layername)
    
    @property
    def group(self):
        ''':class:`LayerGroup`: Group in which the layer is rendered.'''
        return self.loc[0]
    
    def hide(self):
        '''Unchecks the layer in the legend.'''
        self.layer.setItemVisibilityChecked(False)
    
    @property
    def id(self):
        '''str: Layer ID.'''
        return self.vl.id()
    
    @property
    def index(self):
        '''int: Index within :attr:`group` at which the layer is rendered.'''
        return self.loc[1]
    
    def info(self):
        '''
        Prints layer information to the console.
        '''
        print(self,
              f'Name: {self.name!r}',
              f'Source: {relpath(self.vl.dataProvider().dataSourceUri(), False)}',
              f'Geometry: {QgsWkbTypes.geometryDisplayString(self.vl.geometryType())}',
              f'CRS: {self.crs}',
              f'Storage: {self.storage}',
              f'Added to project: {self.added}',
              f'Layer ID: {self.id!r}',
              f'Feature count: {self.vl.featureCount():,}',
              f'Group: {self.loc.group}',
              f'Index: {self.loc.index}',
              sep='\n')
    
    def label(self, ids='all', **kwargs):
        '''
        Parameters
        ----------
        ids : int, List[int], or 'all'
            IDs of the features to label.
            
        Keyword arugments
        -----------------
        **kwargs : dict, optional
            See keyword arguments for :func:`rnet.utils.labeling.simple_labeling`.
        '''
        if ids == 'all':
            self.vl.setLabeling(simple_labeling(**kwargs))
            self.vl.setLabelsEnabled(True)
    
    @property
    def layer(self):
        '''qgis.core.LayerTreeLayer: Layer tree layer.'''
        root = QgsProject.instance().layerTreeRoot()
        return root.findLayer(self.id)
    
    @property
    def loc(self):
        ''':class:`Location`: Location of layer in the project layer tree.'''
        return locate(self.id)
    
    @property
    def name(self):
        '''str: Layer name in the legend.
        
        Note:
            This name may differ from the layer name within the GPKG.
        '''
        return self.vl.name()
    
    @name.setter
    def name(self, name):
        pass
    
    def populate(self, generator: Callable[[Reporter], FeatureGenerator],
                 truncate: bool = True) -> None:
        '''
        Adds to the task manager queue a new task in charge of populating the
        vector layer with features generated by `generator`.
        
        Parameters
        ----------
        generator : Callable[[Reporter], FeatureGenerator]
            Generator that yields features to be added to the vector layer.
        truncate : bool, optional
            If True, then existing features are cleared from the vector layer
            before new ones are added.
        
        See also
        --------
        :class:`PopulateTask`
        '''
        if truncate:
            self.vl.dataProvider().truncate()
        create_and_queue(PopulateTask, self, generator)
    
    def remove(self):
        if self.added:
            self.loc.group.remove(self)
    
    def render(self, *args, **kwargs):
        '''
        Sets the renderer for the vector layer.
        
        Parameters
        ----------
        *args : tuple, optional
            Arguments passed to :meth:`renderer`.
        **kwargs : dict, optional
            Keyword arguments passed to :meth:`renderer`.
        '''
        self.vl.setRenderer(self.renderer(*args, **kwargs))
    
    def save(self, gpkg, layername=None):
        '''
        Saves layer features as a GPKG layer. If the layer has beed added to
        the project legend, then the legend node is replaced with the saved
        GPKG layer.
        
        Parameters
        ----------
        gpkg : :class:`GpkgData` or str
            :class:`GpkgData` object or path specifying the GPKG layer to which
            vertices will be saved.
        layername : str, optional
            Layer name. If None, then the :attr:`name` attribute is used. The
            default is None.
        
        See also
        --------
        :meth:`GpkgData.write_layer`
            Writes the contents of a :class:`Layer` to the GeoPackage.
        '''
        # Save destination
        if type(gpkg) is str:
            gpkg = GpkgData(gpkg)
        elif isinstance(gpkg, GpkgData):
            pass
        else:
            TypeError(r"expected 'str' or 'GpkgData' for argument 'gpkg'")
        
        # Layer name
        if layername is None:
            layername = self.name
        
        # Write to disk
        gpkg.write_layer(self, layername)
        
        # If layer is in the legend, then replace it with the saved GPKG layer
        if self.added:
            group, index = self.loc
            renderer = self.vl.renderer().clone()
            if self.vl.labelsEnabled():
                labeling = self.vl.labeling().clone()
                labels = True
            else:
                labels = False
            self.remove()
            self.vl = QgsVectorLayer(gpkg.layerpath(layername), layername, 'ogr')
            self.vl.setRenderer(renderer)
            if labels:
                self.vl.setLabeling(labeling)
                self.vl.setLabelsEnabled(True)
            group.insert(self, index)
            self.collapse()
        else:
            pass
    
    def show(self):
        '''
        Checks the layer in the legend.
        '''
        self.layer.setItemVisibilityChecked(True)
    
    @property
    def storage(self):
        '''str: Storage type.'''
        return self.vl.dataProvider().storageType()


# class RuleBasedLayer(Layer):
#     '''
#     Rule-based renderer used for rendering data containers.
#     '''
#     def add_rule(self, label, **kwargs):
#         '''
#         Parameters:
#             name
        
#         Keyword arugments:
#             **kwargs: Keyword arguments passed to the :meth:`symbol` method.
#         '''
#         if type(label) is int:
#             self.vl.renderer().rootRule().appendChild(
#                 QgsRuleBasedRenderer.Rule(self.symbol(**kwargs),
#                                           filterExp=f'"id" = {label}',
#                                           label=str(label), description=str(label))
#                 )
#         elif type(label) is str:
#             self.vl.renderer().rootRule().appendChild(
#                 QgsRuleBasedRenderer.Rule(self.symbol(**kwargs),
#                                           filterExp=f'"id" = {label!r}',
#                                           label=label, description=label)
#                 )
    
#     @staticmethod
#     def renderer():
#         return QgsRuleBasedRenderer(QgsRuleBasedRenderer.Rule(None))
    
#     def rules(self):
#         '''
#         Prints information about the rule set to the console.
#         '''
#         print(self.vl.renderer().rootRule().dump())


class PopulateTask(QgsTask):
    '''
    Task for populating a layer.
    
    Parameters
    ----------
    layer : :class:`Layer`
        Layer to populate.
    generate : Callable[[Reporter], FeatureGenerator]
        Function that takes reporter function as argument and returns feature
        generator.
    '''
    
    __slots__ = ['layer', 'generate']
    
    def __init__(self, layer: Layer,
                 generate: Callable[[Reporter], FeatureGenerator]) -> None:
        super().__init__(f'Populating vector layer {layer.name!r}')
        self.layer = layer
        self.generate = generate
    
    def run(self) -> bool:
        layer, generate = self.layer, self.generate
        layer.vl.dataProvider().addFeatures(list(generate(self.setProgress)))
        layer.vl.updateExtents()
        return True
    
    def finished(self, success: bool) -> None:
        pass


class LayerGroup:
    '''
    Class for representing a group in the layer tree. If a group with the
    specified `name` doesn't alreay exist, then it is created on instantiation.
    
    Parameters
    ----------
    name : :obj:`str`, optional
        Layer group name. The default is ''.
    
    Note
    ----
    Empty string denotes layer tree root.
    '''
    
    __slots__ = ['group']
    
    def __init__(self, name=''):
        root = QgsProject.instance().layerTreeRoot()
        if name == '':
            self.group = root
        else:
            self.group = root.findGroup(name)
            if self.group is None:
                self.group = root.insertGroup(0, name)
    
    def __repr__(self):
        return f'<LayerGroup: {self.name!r}>'
    
    @property
    def children(self):
        '''List[qgis.core.QgsLayerTreeNode]: List of child nodes.
        
        Note:
            :class:`qgis.core.QgsLayerTreeNode` serves as the base for
            :class:`qgis.core.QgsLayerTreeGroup` and
            :class:`qgis.core.QgsLayerTreeLayer`.
        '''
        return self.group.children()
    
    def collapse(self):
        '''Collapses the group.'''
        self.group.setExpanded(False)
    
    def expand(self):
        '''Expands the group.'''
        self.group.setExpanded(True)
    
    def hide(self):
        '''Hides the group layers from the map canvas.'''
        self.group.setItemVisibilityChecked(False)
    
    @property
    def ids(self):
        '''List[str]: Child IDs.'''
        return [lyr.layerId() for lyr in self.children]
    
    def info(self):
        '''Print group info to the console.'''
        print(self)
        for i, child in enumerate(self.group.children()):
            print('{}: {!r} <{}>'.format(i, child.name(), child.layerId()))
    
    def insert(self, layer, index=0):
        '''
        Inserts `layer` to the group at the specified `index`.
        
        Parameters:
            layer (Layer): Layer to be inserted.
            index (:obj:`int`, optional): Index to which the layer is inserted.
                Default: 0.
        '''
        QgsProject.instance().addMapLayer(layer.vl, False)
        self.group.insertLayer(index, layer.vl)
    
    def layer(self, layername):
        layers = [lyr.layer() for lyr in self.children if lyr.name() == layername]
        if len(layers) == 0:
            raise LayerNotFoundError(f'layer {layername!r} not in group {self.name!r}')
        elif len(layers) == 1:
            return layers[0]
        else:
            return layers
    
    @property
    def layers(self):
        '''List[qgis.core.QgsVectorLayer]: List of map layers.'''
        return [lyr.layer() for lyr in self.children]
    
    @property
    def name(self):
        '''str: Group name.'''
        return self.group.name()
    
    @property
    def names(self):
        '''List[str]: Child names.'''
        return [lyr.name() for lyr in self.children]
    
    def remove(self, layer):
        '''
        Removes `layer` from the group.
        
        Parameters:
            layer (Layer): Layer to be removed.
        '''
        QgsProject.instance().removeMapLayer(layer.id)
    
    def show(self):
        '''Shows the group layers in the map canvas.'''
        self.group.setItemVisibilityChecked(True)


class GpkgData:
    '''
    Class for representing GPKG data.
    
    Parameters
    ----------
    source : :obj:`str` or :class:`GpkgData`
        Path to GPKG file or another :class:`GpkgData` object.
    '''
    
    __slots__ = ['_path', '_action']
    
    def __init__(self, source):
        if type(source) is str:
            self._path = abspath(source, False)
        elif isinstance(source, GpkgData):
            self._path = abspath(source.path, False)
        if os.path.isfile(self.path):
            self._action = 1
        else:
            self._action = 0
    
    def __repr__(self):
        return f'<GpkgData: {self.path}>'
    
    @property
    def action(self):
        '''int: Action on existing file, used when writing features to the
        disk. 0 indicates CreateOrOverwriteFile, 1 indicates
        CreateOrOverwriteLayer.'''
        return self._action
    
    @property
    def crs(self):
        '''int: EPSG code of CRS.'''
        
        return int(self.layer.crs().authid().split(':')[1])
    
    def exists(self, layername: str = None, raise_error: bool = True):
        '''
        Returns True if layer exists, False otherwise. If `raise_error` is
        True, then :exc:`LayerNotFoundError` is raised instead of returning
        False when a layer does not exist.
        
        Parameters
        ----------
        layername : :obj:`str`, optional
            Layer name. If None, then existence of the GeoPackage is returned.
            The default is None.
        raise_error : :obj:`bool`, optional
            Whether to raise an error for non-existent layers. The default is
            True.
        
        Returns
        -------
        bool
        
        Raises
        ------
        LayerNotFoundError
            If the layer does not exist and `raise_error` is True.
        '''
        if os.path.isfile(self.path):
            if layername is None:
                b = True
            else:
                b = layername in self.layernames
        else:
            b = False
        if raise_error and not b:
            raise LayerNotFoundError(layername)
        else:
            return b
    
    @property
    def layer(self):
        '''qgis.core.QgsVectorLayer: Main layer.'''
        if self.exists():
            return QgsVectorLayer(self.path, self.name, 'ogr')
    
    @property
    def layercount(self):
        '''int: Number of sublayers.'''
        vl = self.layer
        return vl.dataProvider().subLayerCount()
    
    @property
    def layernames(self):
        '''List[str]: Names of sublayers.'''
        vl = self.layer
        return [sublayer.split(QgsDataProvider.SUBLAYER_SEPARATOR)[1]
                for sublayer in vl.dataProvider().subLayers()]
    
    def layerpath(self, layername):
        '''
        Returns the absolute path to the layer with `layername`.
        
        Parameters
        ----------
        name : str
            Layer name.
        
        Returns
        -------
        str
        '''
        return f'{self.path}|layername={layername}'
    
    @property
    def name(self):
        '''str: GPKG file name.'''
        return os.path.basename(self.path)
    
    @property
    def path(self):
        '''str: Absolute path to the GPKG.'''
        return self._path
    
    def sublayer(self, layername):
        '''
        Returns the layer with `layername`.
        
        Parameters
        ----------
        layername : str
            Layer name.
        
        Returns
        -------
        :class:`qgis.core.QgsVectorLayer`
        '''
        if self.exists(layername):
            return QgsVectorLayer(self.layerpath(layername), layername, 'ogr')
    
    def write_features(self, layername, generate, crs, fields, geometrytype):
        '''
        Writes the features generated by `generate` to the GeoPackage.
        
        Parameters
        ----------
        layername : str
            Layer name.
        generate : Generator[qgis.core.QgsFeature, None, None]
            Generator that yields features to be written.
        crs : int
            EPSG code of CRS in which geometry coordinates are represented.
        fields : list of tuple[str, str]
            List of 2-tuples containing field name and field type. Supported
            field types are {'int', 'double', 'str'}.
        geometrytype : {'point', 'linestring', 'polygon'}
            Geometry type.
        
        See also
        --------
        :class:`WriteTask`
        '''
        create_and_queue(WriteTask, self, layername, generate, crs, fields,
                         geometrytype)
    
    def write_layer(self, layer: Layer, layername: str) -> None:
        '''
        Writes the contents of a :class:`Layer` to the GeoPackage.
        
        Parameters
        ----------
        layer : :class:`Layer`
            Layer from which features are taken.
        layername : str
            Layer name.
        '''
        context = QgsProject.instance().transformContext()
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.actionOnExistingFile = self.action
        options.layerName = layername
        
        error = QgsVectorFileWriter.writeAsVectorFormatV2(
            layer.vl, self.path, context, options)
        
        if error[0] == QgsVectorFileWriter.NoError:
            self._action = 1
            return
        else:
            print(error)


class WriteTask(QgsTask):
    '''
    Task for writing features from a data source to a GPKG layer.
    
    Parameters
    ----------
    gpkg : :class:`GpkgData`
        Path to GPKG.
    layername : str
        Layer name.
    generate : Callable[Callable[float, None], Generator[qgis.core.QgsFeature, None, None]]
        Function that takes task progress as an argument and yields features
        to be written.
    crs : int
        EPSG code of CRS in which geometries are represented.
    fields : List[Tuple[str, str]]
        List of 2-tuples containing field name and field type. Supported
        fields types are {'int', 'double', 'str'}.
    geometrytype : {'point', 'linestring', 'polygon'}
        Geometry type.
    
    Raises
    ------
    TypeError
        If any of the field types are not supported.
    ValueError
        If the `geometrytype` is none of the supported types.
    '''
    
    def __init__(self, gpkg: GpkgData, layername: str,
                 generate,
                 crs: int, fields: List[Tuple[str, str]], geometrytype: str
                 ) -> None:
        super().__init__('Writing features to GPKG')
        self.gpkg = gpkg
        self.layername = layername
        self.generate = generate
        self.crs = crs
        self.fields = qgsfields(fields, QgsFields)
        if geometrytype == 'point':
            self.geometrytype = 1
        elif geometrytype == 'linestring':
            self.geometrytype = 2
        elif geometrytype == 'polygon':
            self.geometrytype = 3
        else:
            raise ValueError(f'unknown geomtery type {geometrytype!r}')
    
    def run(self) -> bool:
        srs = QgsCoordinateReferenceSystem.fromEpsgId(self.crs)
        context = QgsProject.instance().transformContext()
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.actionOnExistingFile = self.gpkg.action
        options.layerName = self.layername
        writer = QgsVectorFileWriter.create(
            fileName=self.gpkg.path,
            fields=self.fields,
            geometryType=self.geometrytype,
            srs=srs,
            transformContext=context,
            options=options
            )
        writer.addFeatures(list(self.generate(self.setProgress)))
        del writer  # flush to disk
        self.gpkg._action = 1
        return True
    
    def finished(self, success: bool) -> None:
        pass
