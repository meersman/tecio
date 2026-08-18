# Zones

Every zone reader shares the {class}`~tecio.TecplotZoneReader` interface,
split into two concrete shapes depending on the zone's topology:
{class}`~tecio.TecplotOrderedZoneReader` for IJK-ordered zones, and
{class}`~tecio.TecplotFEZoneReader` for finite-element zones (classic
element types, plus FEPOLYGON/FEPOLYHEDRON/FEMIXED for everything except
connectivity, which no format can read for those types yet).

Only properties that mean the same thing for every zone, regardless of
topology, title, zone type, solution time, strand ID, the variable list,
auxiliary data, live on the base class. Dimensions and connectivity don't:
an ordered zone has no node map, and an FE zone has no ``I``/``J``/``K``, so
each lives only on its own subclass. Accessing the wrong one raises
``AttributeError``:

```python
with tecio.open("flow.szplt") as r:
    zone = r.zone[0]
    zone.title, zone.zone_type  # always available
    if isinstance(zone, tecio.TecplotOrderedZoneReader):
        zone.dimensions  # (I, J, K)
    elif isinstance(zone, tecio.TecplotFEZoneReader):
        zone.node_map  # ndarray | None
```

```{eval-rst}
.. currentmodule:: tecio

.. autoclass:: TecplotZoneReader
   :members:
   :show-inheritance:

.. autoclass:: TecplotOrderedZoneReader
   :members:
   :show-inheritance:

.. autoclass:: TecplotFEZoneReader
   :members:
   :show-inheritance:
```
