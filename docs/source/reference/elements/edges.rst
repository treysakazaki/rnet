Edges
=====

.. currentmodule:: rnet

Constructors
------------

.. autosummary::
    :toctree: generated/

    EdgeData
    ~EdgeData.from_gpkg

Descriptions
------------

Edges are defined by a sequence of vertices whose IDs are stored in the :attr:`EdgeData.df` frame. The following methods provide further descriptions of each edge in the dataset.

.. autosummary::
    :toctree: generated/

    ~EdgeData.coords
    ~EdgeData.lengths
    ~EdgeData.tags

Iteration
---------

.. autosummary::
    :toctree: generated/

    ~EdgeData.edges
    ~EdgeData.generate

Manipulation
------------

.. autosummary::
    :toctree: generated/

    ~EdgeData.masked

Output
------

.. autosummary::
    :toctree: generated/

    ~EdgeData.render
    ~EdgeData.to_gpkg
