{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

{% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

{% for item in attributes %}
      ~{{ name }}.{{ item }}
{% endfor %}
{% endif %}
{% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

{% for item in methods %}
      ~{{ name }}.{{ item }}
{% endfor %}
{% endif %}
