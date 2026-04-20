#!MC 1410
$!ReadDataSet  '"STANDARDSYNTAX" "1.0" "FILENAME_FILE" "simple_spectral.szplt"'
  DataSetReader = 'Tecplot Subzone Data Loader'
  ReadDataOption = New
  ResetStyle = No
  AssignStrandIDs = No
  InitialPlotType = Automatic
  InitialPlotFirstZoneOnly = No
  AddZonesToExistingStrands = No
  VarLoadMode = ByName
$!FrameLayout ShowBorder = No
$!FrameLayout Height = 4.6
$!FrameLayout BackgroundColor = Black
$!FieldLayers ShowShade = No
$!FieldLayers ShowContour = Yes
$!GlobalContour 1  ColorMapFilter{ColorMapDistribution = Continuous}
$!GlobalContour 1  ColorMapFilter{ContinuousColor{CMin = -60}}
$!GlobalContour 1  ColorMapFilter{ContinuousColor{CMax = 60}}
$!ContourLevels New
  ContourGroup = 1
  RawData
7
-60
-40
-20
0
20
40
60
$!TwoDAxis XDetail{Title{TextShape{FontFamily = 'Times'}}}
$!TwoDAxis XDetail{Title{TextShape{IsBold = No}}}
$!TwoDAxis XDetail{Title{TextShape{SizeUnits = Point}}}
$!TwoDAxis XDetail{Title{TextShape{Height = 24}}}
$!TwoDAxis XDetail{Title{Offset = 12}}
$!TwoDAxis XDetail{TickLabel{TextShape{FontFamily = 'Times'}}}
$!TwoDAxis XDetail{TickLabel{TextShape{SizeUnits = Point}}}
$!TwoDAxis XDetail{TickLabel{TextShape{Height = 18}}}
$!TwoDAxis XDetail{AutoGrid = No}
$!TwoDAxis XDetail{Ticks{NumMinOrTicks = 1}}
$!TwoDAxis XDetail{Title{Color = White}}
$!TwoDAxis XDetail{TickLabel{Color = White}}
$!TwoDAxis XDetail{AxisLine{Color = White}}
$!TwoDAxis XDetail{Title{TitleMode = UseText}}
$!TwoDAxis XDetail{Title{Text = '<i>x</i>'}}
$!TwoDAxis YDetail{Title{TextShape{FontFamily = 'Times'}}}
$!TwoDAxis YDetail{Title{TextShape{IsBold = No}}}
$!TwoDAxis YDetail{Title{TextShape{SizeUnits = Point}}}
$!TwoDAxis YDetail{Title{TextShape{Height = 24}}}
$!TwoDAxis YDetail{Title{Offset = 10}}
$!TwoDAxis YDetail{TickLabel{TextShape{FontFamily = 'Times'}}}
$!TwoDAxis YDetail{TickLabel{TextShape{SizeUnits = Point}}}
$!TwoDAxis YDetail{TickLabel{TextShape{Height = 18}}}
$!TwoDAxis YDetail{AxisLine{Color = White}}
$!TwoDAxis YDetail{Title{Color = White}}
$!TwoDAxis YDetail{TickLabel{Color = White}
$!TwoDAxis YDetail{Title{TitleMode = UseText}}
$!TwoDAxis YDetail{Title{Text = '<i>y</i>'}}
$!TwoDAxis YDetail{AutoGrid = No}
$!TwoDAxis YDetail{Ticks{NumMinOrTicks = 1}}
$!TwoDAxis ViewportPosition{Y1 = 18}
$!TwoDAxis ViewportPosition{X1 = 12}
$!TwoDAxis ViewportPosition{X2 = 86}
$!TwoDAxis ViewportPosition{Y2 = 90}
$!TwoDAxis AxisMode = Independent
$!TwoDAxis YDetail{RangeMax = 1}
$!TwoDAxis XDetail{RangeMax = 2}
$!GlobalContour 1  Legend{XYPos{X = 100}}
$!GlobalContour 1  Legend{XYPos{Y = 100}}
$!GlobalContour 1  Legend{TextColor = White}
$!GlobalContour 1  Legend{NumberTextShape{FontFamily = 'Times'}}
$!GlobalContour 1  Legend{NumberTextShape{SizeUnits = Point}}
$!GlobalContour 1  Legend{NumberTextShape{Height = 18}}
$!GlobalContour 1  Legend{Header{TextShape{FontFamily = 'Times'}}}
$!GlobalContour 1  Legend{Header{TextShape{SizeUnits = Point}}}
$!GlobalContour 1  Legend{Header{TextShape{Height = 18}}}
$!GlobalContour 1  Legend{Box{BoxType = None}}
$!GlobalContour 1  Legend{RowSpacing = 2.18}
$!GlobalContour 1  Labels{AutoLevelSkip = 1}
$!GlobalContour 1  Legend{Header{UseCustomText = Yes}}
$!GlobalContour 1  Legend{Header{TextType = Regular}}
$!GlobalContour 1  Legend{Header{CustomText = 'Vorticity'}}
$!AttachText 
  AnchorPos
    {
    X = 72.5
    Y = 91
    }
  TextShape
    {
    FontFamily = 'Times'
    IsBold = No
    Height = 18
    }
  Color = White
  Text = '<i>t</i> = &(SOLUTIONTIME%7.4f)'
$!PrintSetup Palette = Color
$!ExportSetup ExportFormat = MPEG4
$!ExportSetup ImageWidth = 1080
$!ExportSetup UseSuperSampleAntiAliasing = Yes
$!ExportSetup AnimationSpeed = 50
$!ExportSetup ExportFName = 'simple_spectral.mp4'
$!AnimateTime 
  StartTime = 1E-10
  EndTime = 29.9500000001
  Skip = 1
  CreateMovieFile = Yes
