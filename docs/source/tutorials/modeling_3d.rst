Three-Dimensional Modeling
==========================

In this tutorial, we will create a three-dimensional model of the road network in Ichihara City, Japan.

Materials
---------

This tutorial requires the following materials:

    - street map data of Ichihara City (``ichihara.osm``), and
    - elevation data of Ichihara City (``n35_e139_1arc_v3.tif`` and ``n35_e140_1arc_v3.tif``).

Procedure
---------

1. Follow the steps 1 through 4 in the previous tutorial to first construct a two-dimensional road network model.

2. Instantiate :class:`ElevationQueryEngine` using the :meth:`~ElevationQueryEngine.from_tifs` method::

    >>> engine = rn.ElevationQueryEngine.from_tifs("n35_e139_1arc_v3.tif", "n35_e140_1arc_v3.tif")

3. Convert the two-dimensional model to a three-dimensional model using the :meth:`expand` method::

    >>> G.expand(engine)

   Confirm that the vertices and nodes in the model now have three-dimensional coordinates::

    >>> G.vdata.df
                     x          y          z
    id                                      
    0       139.933095  35.361260  27.660686
    1       139.933177  35.361303  27.700376
    2       139.933514  35.361481  28.943817
    3       139.934409  35.361953  33.050092
    4       139.935344  35.362447  36.824534
                            ...
    177148  140.313908  35.360385  12.376692
    177149  140.314289  35.360574  12.497454
    177150  140.314741  35.360745  12.481726
    177151  140.314892  35.360778  12.502719
    177152  140.315844  35.361007  12.407990
    [177153 rows x 3 columns]
    >>> G.ndata.df
                     x          y           z
    id                                       
    0       139.933095  35.361260   27.660686
    12      139.937906  35.259384  192.452257
    16      139.938034  35.438496    2.568035
    40      139.939477  35.386629    7.228554
    59      139.940661  35.248059  278.511355
                            ...
    176940  140.275111  35.374252   19.552842
    177020  140.284766  35.525735   85.109855
    177034  140.286765  35.436393   11.129895
    177142  140.307335  35.360429   14.269132
    177152  140.315844  35.361007   12.407990
    [37953 rows x 3 columns]

4. Call the :meth:`~GraphData.render` method to display the road network model on the map canvas::

    >>> G.render()

5. To save the model for future reference, call the :meth:`~GraphData.to_gpkg` method::

    >>> G.to_gpkg("ichihara3d.gpkg")
