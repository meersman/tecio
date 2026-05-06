# libtecio — C Library Bindings

Low-level Python wrappers for the TecIO C library functions.

## Enums

```{eval-rst}
.. autoclass:: tecio.libtecio.FileFormat
   :members:
   :undoc-members:

.. autoclass:: tecio.libtecio.FileType
   :members:
   :undoc-members:

.. autoclass:: tecio.libtecio.ZoneType
   :members:
   :undoc-members:

.. autoclass:: tecio.libtecio.DataType
   :members:
   :undoc-members:

.. autoclass:: tecio.libtecio.ValueLocation
   :members:
   :undoc-members:

.. autoclass:: tecio.libtecio.FaceNeighborMode
   :members:
   :undoc-members:

.. autoclass:: tecio.libtecio.FeCellShape
   :members:
   :undoc-members:

.. autoclass:: tecio.libtecio.DataFormat
   :members:
   :undoc-members:
```

## Exceptions

```{eval-rst}
.. autoclass:: tecio.libtecio.TecioError
```

## SZL Read Functions

```{eval-rst}
.. autofunction:: tecio.libtecio.tec_file_reader_open
.. autofunction:: tecio.libtecio.tec_file_get_type
.. autofunction:: tecio.libtecio.tec_data_set_get_title
.. autofunction:: tecio.libtecio.tec_data_set_get_num_vars
.. autofunction:: tecio.libtecio.tec_data_set_get_num_zones
.. autofunction:: tecio.libtecio.tec_zone_get_ijk
.. autofunction:: tecio.libtecio.tec_zone_get_title
.. autofunction:: tecio.libtecio.tec_zone_get_type
.. autofunction:: tecio.libtecio.tec_zone_is_enabled
.. autofunction:: tecio.libtecio.tec_zone_get_solution_time
.. autofunction:: tecio.libtecio.tec_zone_get_strand_id
.. autofunction:: tecio.libtecio.is_64bit
.. autofunction:: tecio.libtecio.tec_zone_node_map_get_64
.. autofunction:: tecio.libtecio.tec_zone_node_map_get
.. autofunction:: tecio.libtecio.tec_var_get_name
.. autofunction:: tecio.libtecio.tec_var_is_enabled
.. autofunction:: tecio.libtecio.tec_zone_var_get_type
.. autofunction:: tecio.libtecio.tec_zone_var_get_value_location
.. autofunction:: tecio.libtecio.tec_zone_var_is_passive
.. autofunction:: tecio.libtecio.tec_zone_var_get_shared_zone
.. autofunction:: tecio.libtecio.tec_zone_var_get_num_values
.. autofunction:: tecio.libtecio.tec_zone_var_get_float_values
.. autofunction:: tecio.libtecio.tec_zone_var_get_double_values
.. autofunction:: tecio.libtecio.tec_zone_var_get_int32_values
.. autofunction:: tecio.libtecio.tec_zone_var_get_int16_values
.. autofunction:: tecio.libtecio.tec_zone_var_get_uint8_values
```

## SZL Auxiliary Data Functions

```{eval-rst}
.. autofunction:: tecio.libtecio.tec_data_set_aux_data_get_num_items
.. autofunction:: tecio.libtecio.tec_data_set_aux_data_get_item
.. autofunction:: tecio.libtecio.tec_var_aux_data_get_num_items
.. autofunction:: tecio.libtecio.tec_var_aux_data_get_item
.. autofunction:: tecio.libtecio.tec_zone_aux_data_get_num_items
.. autofunction:: tecio.libtecio.tec_zone_aux_data_get_item
```

## SZL Write Functions

```{eval-rst}
.. autofunction:: tecio.libtecio.tec_file_writer_open
.. autofunction:: tecio.libtecio.tec_file_writer_close
.. autofunction:: tecio.libtecio.tec_zone_create_ijk
.. autofunction:: tecio.libtecio.tec_zone_create_fe
.. autofunction:: tecio.libtecio.tec_zone_set_unsteady_options
.. autofunction:: tecio.libtecio.tec_data_set_add_aux_data
.. autofunction:: tecio.libtecio.tec_var_add_aux_data
.. autofunction:: tecio.libtecio.tec_zone_add_aux_data
.. autofunction:: tecio.libtecio.tec_zone_var_write_double_values
.. autofunction:: tecio.libtecio.tec_zone_var_write_float_values
.. autofunction:: tecio.libtecio.tec_zone_var_write_int32_values
.. autofunction:: tecio.libtecio.tec_zone_var_write_int16_values
.. autofunction:: tecio.libtecio.tec_zone_var_write_uint8_values
.. autofunction:: tecio.libtecio.tec_zone_node_map_write32
.. autofunction:: tecio.libtecio.tec_zone_node_map_write64
.. autofunction:: tecio.libtecio.tec_zone_face_nbr_write_connections32
.. autofunction:: tecio.libtecio.tec_zone_face_nbr_write_connections64
```

## Classic PLT API Functions

```{eval-rst}
.. autofunction:: tecio.libtecio.tecini142
.. autofunction:: tecio.libtecio.tecend142
.. autofunction:: tecio.libtecio.tecflush142
.. autofunction:: tecio.libtecio.tecfil142
.. autofunction:: tecio.libtecio.tecforeign142
.. autofunction:: tecio.libtecio.teczne142
.. autofunction:: tecio.libtecio.tecpolyzne142
.. autofunction:: tecio.libtecio.tecznefemixed142
.. autofunction:: tecio.libtecio.tecdat142
.. autofunction:: tecio.libtecio.tecnode142
.. autofunction:: tecio.libtecio.tecface142
.. autofunction:: tecio.libtecio.tecpolyface142
.. autofunction:: tecio.libtecio.tecpolybconn142
.. autofunction:: tecio.libtecio.tecauxstr142
.. autofunction:: tecio.libtecio.tecvauxstr142
.. autofunction:: tecio.libtecio.teczauxstr142
.. autofunction:: tecio.libtecio.tecusr142
```
