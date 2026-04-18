#!MC 1410
$!ReadDataSet  '"STANDARDSYNTAX" "1.0" "FILENAME_FILE" "gravity_waves.szplt"'
  DataSetReader = 'Tecplot Subzone Data Loader'
  ReadDataOption = New
  ResetStyle = No
  AssignStrandIDs = No
  InitialPlotType = Automatic
  InitialPlotFirstZoneOnly = No
  AddZonesToExistingStrands = No
  VarLoadMode = ByName
$!PlotType = Cartesian3D
$!FieldLayers ShowShade = No
$!FieldLayers ShowContour = Yes
$!GlobalContour 1  ColorMapName = 'Two Color'
$!GlobalContour 1  ColorMapFilter{ColorMapDistribution = Continuous}
$!GlobalContour 1  ColorMapFilter{ContinuousColor{CMin = -0.950000000000000178}}
$!GlobalContour 1  ColorMapFilter{ContinuousColor{CMax = -0.0500000000000001277}}
$!ThreeDAxis DepXToZRatio = 2
$!FrameLayout ShowBorder = No
$!FrameLayout Height = 3.5
$!FrameLayout BackgroundColor = Black
$!GlobalContour 1  Legend{Show = No}
$!ThreeDAxis FrameAxis{Show = No}
$!ThreeDView 
  PSIAngle = 111.718
  ThetaAngle = 27.6387
  AlphaAngle = 178.663
  ViewerPosition
    {
    X = -8.7038630448356
    Y = -16.62160872259596
    Z = -14.94649982448934
    }
  ViewWidth = 2.74777
$!PrintSetup Palette = Color
$!ExportSetup ExportFormat = MPEG4
$!ExportSetup ImageWidth = 1080
$!ExportSetup AnimationSpeed = 100
$!ExportSetup ExportFName = 'gravity_waves.mp4'
$!AnimateTime 
  StartTime = 0.000625
  EndTime = 9.994375
  Skip = 1
  CreateMovieFile = Yes
