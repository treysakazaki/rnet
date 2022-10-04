from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsVectorLayer,
    QgsProject
    )
from PyQt5.QtGui import QColor


def render_path(coords):
    f = QgsFeature()
    f.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*p) for p in coords]))
    vl = QgsVectorLayer('linestring?crs=epsg:6677', 'path', 'memory')
    vl.dataProvider().addFeatures([f])
    vl.renderer().symbol().symbolLayers()[0].setWidth(0.5)
    vl.renderer().symbol().symbolLayers()[0].setColor(QColor.fromRgb(255,0,0))
    QgsProject.instance().addMapLayer(vl)
