from qgis.core import QgsProject


def project_info():
    proj = QgsProject.instance()
    print(f'Home path: {proj.homePath()}',
          f'CRS: {proj.crs().authid()}',
          sep='\n')
