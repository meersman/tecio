# SZL (``.szplt``) Files

Read and write Tecplot SZL subzone-loadable binary files.

## Reading

```{eval-rst}
.. autoclass:: tecio.szl.Read
   :members:

.. autoclass:: tecio.szl._read.ReadZone
   :members:

.. autoclass:: tecio.szl._read.ReadVariable
   :members:

.. autoclass:: tecio.szl._read.ReadAuxData
   :members:
```

## Writing

```{eval-rst}
.. autoclass:: tecio.szl.Write
   :members:

.. autofunction:: tecio.szl._write.write_data

.. autofunction:: tecio.szl._write.write_connectivity

.. autofunction:: tecio.szl._write.write_zone_aux_data

.. autofunction:: tecio.szl._write.write_dataset_aux_data

.. autofunction:: tecio.szl._write.write_variable_aux_data
```
