{{ fullname | escape | underline }}

.. currentmodule:: rnet

.. autoclass:: {{ objname }}
    
    {% if methods %}   
    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
        {% if item != "__init__" %}{{ item }}{% endif %}
    {%- endfor %}
    {% endif %}

    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
    {% for item in attributes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
