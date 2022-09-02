Elements
========

.. currentmodule:: rnet

To model road networks, RNet employs undirected weighted graphs defined by a set of nodes, :math:`\mathcal{N} = \{i \mid i \in \mathbb{N}\}`, connected by a set of edges, :math:`\mathcal{E} = \{\{i, j\} \mid \{i, j\} \in \mathcal{N}^2, i \neq j\}`. Nodes represent intersections and dead ends. Edges provide connections between nodes to represent the roads in the network. An additional set of vertices, :math:`\mathcal{V} \subset \mathcal{N}`, is used to define edge geometries. A set of links, :math:`\mathcal{L}`, provides connections between neighboring vertices.


Basic element types
-------------------

.. toctree::
    :maxdepth: 1

    edges
    vertices

Additional element types
------------------------

.. toctree::
    :maxdepth: 1

    places
