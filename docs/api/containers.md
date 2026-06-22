# Accessing Zones and Variables

{class}`~tecio.ZoneList` and {class}`~tecio.VariableList` are the
format-agnostic container types returned by ``Read.zone`` and
``ReadZone.variable`` for every supported format ({mod}`tecio.szl`,
{mod}`tecio.plt`, and {mod}`tecio.dat`). See [Accessing Variable
Data](index.md#accessing-variable-data) for the access model and ``get_array``
examples.

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: ZoneList
   :members:
   :show-inheritance:

.. autoclass:: VariableList
   :members:
   :show-inheritance:
```
