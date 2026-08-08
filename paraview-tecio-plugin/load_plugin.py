"""Scripted load of Paraview Tecio Plugin."""
from paraview.simple import LoadPlugin, OpenDataFile, Show, Render

LoadPlugin("./TecplotTecioReader.py", remote=False, ns=globals())
reader = OpenDataFile("/Users/jmeersman/tecio-dev/tests/Onera.szplt")
Show(reader)
Render()
