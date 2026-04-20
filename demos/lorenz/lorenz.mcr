#!MC 1410
$!ReadDataSet  '"STANDARDSYNTAX" "1.0" "FILENAME_FILE" "lorenz.szplt"'
  DataSetReader = 'Tecplot Subzone Data Loader'
  ReadDataOption = New
  ResetStyle = No
  AssignStrandIDs = No
  InitialPlotType = Automatic
  InitialPlotFirstZoneOnly = No
  AddZonesToExistingStrands = No
  VarLoadMode = ByName
$!PlotType = Cartesian3D
$!FrameLayout ShowBorder = No
$!FrameLayout BackgroundColor = Black
$!FrameLayout Height = 5.5
$!FieldLayers ShowShade = No
$!FieldLayers ShowScatter = Yes
$!ActiveFieldMaps -= [1]
$!FieldMap [3]  Mesh{Show = No}
$!FieldMap [2]  Mesh{LineThickness = 0.3}
$!GlobalScatter Var = 4
$!FieldMap [1]  Scatter{Show = No}
$!FieldMap [2]  Scatter{Show = No}
$!FieldMap [3]  Scatter{SymbolShape{GeomShape = Sphere}}
$!FieldMap [3]  Scatter{Color = Yellow}
$!FieldMap [3]  Scatter{FrameSize = 1}
$!SetContourVar 
  Var = 4
  ContourGroup = 1
  LevelInitMode = ResetToNice
$!FieldMap [2]  Mesh{Color = Multi}
$!FieldLayers ShowMesh = Yes
$!SetContourVar 
  Var = 5
  ContourGroup = 1
  LevelInitMode = ResetToNice
$!GlobalContour 1  ColorMapName = 'cmocean - thermal'
$!ContourLevels New
  ContourGroup = 1
  RawData
51
-10
-9.8
-9.6
-9.4
-9.2
-9
-8.8
-8.6
-8.4
-8.2
-8
-7.8
-7.6
-7.4
-7.2
-7
-6.8
-6.6
-6.4
-6.2
-6
-5.8
-5.6
-5.4
-5.2
-5
-4.8
-4.6
-4.4
-4.2
-4
-3.8
-3.6
-3.4
-3.2
-3
-2.8
-2.6
-2.4
-2.2
-2
-1.8
-1.6
-1.4
-1.2
-1
-0.8
-0.6
-0.4
-0.2
0
$!GlobalContour 1  Legend{Show = No}
$!ThreeDAxis FrameAxis{Show = No}
$!ThreeDView 
  PSIAngle = 146.2
  ThetaAngle = -58.8823
  AlphaAngle = 0.774591
    ViewerPosition
    {
    X = 215.9691212281154
    Y = -128.5208142457311
    Z = -342.8116583671255
    }
  ViewWidth = 67.0335
$!PrintSetup Palette = Color
$!ExportSetup ExportFormat = MPEG4
$!ExportSetup ImageWidth = 1080
$!ExportSetup UseSuperSampleAntiAliasing = Yes
$!ExportSetup AnimationSpeed = 45
$!ExportSetup ExportFName = 'lorenz.mp4'
$!AnimateTime 
  StartTime = 0
  EndTime = 49.9
  Skip = 1
  CreateMovieFile = Yes
