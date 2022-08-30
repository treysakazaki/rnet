RNet 
====

Tools for modeling and analyzing road networks in QGIS.

Installation
------------

To install the development version, do::

   $ git clone https://github.com/treysakazaki/rnet.git

To make RNet discoverable in QGIS, append the RNet directory to the ``PYTHONPATH`` environment variable through the QGIS ``Options`` dialog. Steps for doing so may be found in the `QGIS documentation <https://docs.qgis.org/3.22/en/docs/user_manual/introduction/qgis_configuration.html#system-settings>`_. The RNet library may then be accessed through the `QGIS Python console <https://docs.qgis.org/3.22/en/docs/user_manual/plugins/python_console.html>`_.

The recommended way for importing RNet is::

   >>> import rnet as rn

Getting started
---------------

Obtain street map data from the `OpenStreetMap dataset <https://www.openstreetmap.org/>`_. Then, create and visualize the road network model::

   >>> G = rn.GraphData.from_osm(path/to/osm)
   >>> G.render()

Save the model in GeoPackage format::

   >>> G.to_gpkg("model.gpkg")
