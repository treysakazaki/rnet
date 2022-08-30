from PyQt5.QtGui import QFont
from qgis.core import (
    QgsPalLayerSettings,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorLayerSimpleLabeling
    )


def simple_labeling(*, fontfamily: str = 'Arial', fontsize: int = 10
                    ) -> QgsVectorLayerSimpleLabeling:
    '''
    Returns labeler.

    Keyword arguments
    -----------------
    fontfamily : str, optional
        Font family. The default is 'Arial'.
    fontsize : int, optional
        Font size. The default is 10.
    
    Returns
    -------
    :class:`qgis.core.QgsVectorLayerSimpleLabeling`
    '''
    settings = QgsPalLayerSettings()
    settings.fieldName = 'fid'
    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    fmt = QgsTextFormat()
    fmt.setBuffer(buffer)
    fmt.setFont(QFont(fontfamily))
    fmt.setSize(fontsize)
    settings.setFormat(fmt)
    # if 'maxscale' in kwargs:
    #     settings.scaleVisibility = True
    #     settings.maximumScale = kwargs['maxscale']
    # if 'minscale' in kwargs:
    #     settings.scaleVisibility = True
    #     settings.minimumScale = kwargs['minscale']
    return QgsVectorLayerSimpleLabeling(settings)
