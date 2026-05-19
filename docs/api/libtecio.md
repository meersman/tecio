# tecio.libtecio

```{eval-rst}
.. automodule:: tecio.libtecio
```

The {mod}`libtecio` module provides the direct interface with the
TecIO C library functions. Each are wrapped in an equivalent Python
function which also handles conversions from Python data objects to C
compatible objects, and vis versa for the read functions.

These are the basis of the {mod}`tecio` Python package, and each
submodule provides a higher level interface with these
functions. However, the the Python wrapper functions in this submodule
are kept in the public API to allow users directly call the TecIO
library if desired.

## Enums

See {doc}`libtecio_enums` for full documentation.

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:
   
   FileFormat
   FileType
   ZoneType
   DataType
   ValueLocation
   FaceNeighborMode
   FeCellShape
   DataFormat
   VarStatus
   Boolean
```



## Exceptions

```{eval-rst}
.. autoclass:: tecio.libtecio.TecioError
```

## SZL Read Functions

See {doc}`libtecio_szl_read` for full documentation.

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tec_file_reader_open
   tec_file_reader_close
   tec_file_get_type
   tec_data_set_get_title
   tec_data_set_get_num_vars
   tec_data_set_get_num_zones
   tec_zone_get_ijk
   tec_zone_get_title
   tec_zone_get_type
   tec_zone_is_enabled
   tec_zone_get_solution_time
   tec_zone_get_strand_id
   is_64bit
   tec_zone_node_map_get_64
   tec_zone_node_map_get
   tec_var_get_name
   tec_var_is_enabled
   tec_zone_var_get_type
   tec_zone_var_get_value_location
   tec_zone_var_is_passive
   tec_zone_var_get_shared_zone
   tec_zone_var_get_num_values
   tec_zone_var_get_float_values
   tec_zone_var_get_double_values
   tec_zone_var_get_int32_values
   tec_zone_var_get_int16_values
   tec_zone_var_get_uint8_values
   tec_data_set_aux_data_get_num_items
   tec_data_set_aux_data_get_item
   tec_var_aux_data_get_num_items
   tec_var_aux_data_get_item
   tec_zone_aux_data_get_num_items
   tec_zone_aux_data_get_item
```

## SZL Write Functions

See {doc}`libtecio_szl_write` for full documentation.

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tec_file_writer_open
   tec_file_writer_close
   tec_zone_create_ijk
   tec_zone_create_fe
   tec_zone_set_unsteady_options
   tec_zone_var_write_double_values
   tec_zone_var_write_float_values
   tec_zone_var_write_int32_values
   tec_zone_var_write_int16_values
   tec_zone_var_write_uint8_values
   tec_zone_node_map_write32
   tec_zone_node_map_write64
   tec_zone_face_nbr_write_connections32
   tec_zone_face_nbr_write_connections64
   tec_data_set_add_aux_data
   tec_var_add_aux_data
   tec_zone_add_aux_data
```

## Classic API Functions

See {doc}`libtecio_classic` for full documentation.

```{eval-rst}
.. currentmodule:: tecio.libtecio

.. autosummary::
   :nosignatures:

   tecini142
   tecend142
   tecflush142
   tecfil142
   tecforeign142
   teczne142
   tecpolyzne142
   tecznefemixed142
   tecdat142
   tecnode142
   tecface142
   tecpolyface142
   tecpolybconn142
   tecauxstr142
   tecvauxstr142
   teczauxstr142
   tecusr142
```

```{toctree}
:hidden:

libtecio_enums
libtecio_szl_read
libtecio_szl_write
libtecio_classic
```
