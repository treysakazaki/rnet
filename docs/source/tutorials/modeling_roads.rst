Modeling a Road Network
=======================

In this tutorial, we will create a two-dimensional road network model of Ichihara City in Chiba Prefecture, Japan.

Materials
---------

This tutorial requires the following materials:

    - street map data of Ichihara City.

Procedure
---------

1. Create a new QGIS project.

2. Set the project CRS to EPSG:6677 - JGD2011 / Japan Plane Rectangular CS IX. This may done from the ``CRS`` tab of the ``Project ► Properties...`` dialog (see `QGIS documentation <https://docs.qgis.org/3.22/en/docs/user_manual/working_with_projections/working_with_projections.html#project-coordinate-reference-systems>`_), or from the Python console with the following command::

    >>> QgsProject.instance().setCrs(QgsCoordinateReferenceSystem("EPSG:6677"))

3. Use the :meth:`GraphData.from_osm` method to construct the road network model from the street map data of Ichihara City::

    >>> import rnet as rn
    >>> G = rn.GraphData.from_osm("ichihara.osm")

4. Transform coordinates from EPSG:4326 to EPSG:6677 using the :meth:`transform` method::

    >>> G.transform(6677)

5. Call the :meth:`render` method to display the road network model on the map canvas::

    >>> G.render()

   The model is visualized using two point layers, ``vertices`` and ``nodes``, and two linestring layers, ``links`` and ``edges``. These layers are all temporary scratch layers, so they will be discarded when QGIS is closed.

6. To save the model for future reference, call the :meth:`to_gpkg` method::

    >>> G.to_gpkg("ichihara2d.gpkg")

   The model will be saved in a GeoPackage file containing four sublayers. The temporary scratch layers generated in the previous step will automatically be replaced with the newly saved GeoPackage layers.
