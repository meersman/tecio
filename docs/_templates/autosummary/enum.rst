:orphan:

{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :exclude-members: {{ members | join(', ') }}
