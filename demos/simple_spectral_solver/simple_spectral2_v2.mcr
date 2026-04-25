#!MC 1410
### Frame Number 1 ###
$!FrameLayout 
  ShowBorder = No
  ShowHeader = No
  BackgroundColor = Black
  HeaderColor = Red
  XYPos
    {
    X = 1
    Y = 0.25
    }
  Width = 9
  Height = 9.75
$!PlotType  = Cartesian2D
$!FrameName  = 'trace'
$!ActiveFieldMaps  =  [1]
$!GlobalContour  1
  Var = 9
  ColorMapName = 'GrayScale'
$!GlobalContour 1  Legend{Show = No}
$!GlobalContour 1  ColorMapFilter{ColorMapDistribution = Continuous}
$!ContourLevels New
  ContourGroup = 1
  RawData
1
0.5
$!View Fit
$!TwoDAxis 
  DepXToYRatio = 1
  ViewportPosition
    {
    X1 = 12
    Y1 = 10
    X2 = 99
    Y2 = 90.25
    }
  ViewportTopSnapTarget = 90.25
$!TwoDAxis 
  XDetail
    {
    CoordScale = Linear
    RangeMin = 0
    RangeMax = 1.0007189072609632419
    AutoGrid = No
    GRSpacing = 0.2
    Ticks
      {
      NumMinOrTicks = 1
      }
    TickLabel
      {
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        SizeUnits = Point
        Height = 18
        }
      }
    Title
      {
      TitleMode = UseText
      Text = '<i>x</i>'
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        IsBold = No
        SizeUnits = Point
        Height = 24
        }
      Offset = 7
      }
    AxisLine
      {
      Color = White
      }
    }
$!TwoDAxis 
  YDetail
    {
    CoordScale = Linear
    RangeMin = 0
    RangeMax = 1
    AutoGrid = No
    GRSpacing = 0.2
    Ticks
      {
      NumMinOrTicks = 1
      }
    TickLabel
      {
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        SizeUnits = Point
        Height = 18
        }
      }
    Title
      {
      TitleMode = UseText
      Text = '<i>y</i>'
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        IsBold = No
        SizeUnits = Point
        Height = 24
        }
      Offset = 10
      }
    AxisLine
      {
      Color = White
      }
    }
$!FieldLayers 
  ShowMesh = No
  ShowContour = Yes
  ShowEdge = No
$!Linking 
  BetweenFrames
    {
    LinkSolutionTime = Yes
    }
$!AttachText 
  AnchorPos
    {
    X = 98.5
    Y = 91
    }
  TextShape
    {
    FontFamily = 'Times'
    IsBold = No
    Height = 24
    }
  Color = White
  Anchor = Right
  Text = '<i>t</i> = &(SOLUTIONTIME%05.2f)'
$!AttachText 
  AnchorPos
    {
    X = 54.59101818610913
    Y = 99
    }
  TextShape
    {
    FontFamily = 'Times'
    IsBold = No
    Height = 24
    }
  Color = White
  Anchor = HeadCenter
  Text = 'Kelvin-Helmholtz Instability Test\n(<i>N<sub>x</sub></i> <math>4</math> <i>N<sub>y </sub></i>) = (&(AUXDATASET:Nx) <math>4</math> &(AUXDATASET:Nx))'
### Frame Number 2 ###
$!CreateNewFrame 
$!FrameLayout 
  ShowBorder = No
  ShowHeader = No
  BackgroundColor = Black
  HeaderColor = Red
  XYPos
    {
    X = 10
    Y = 0.25
    }
  Width = 4
  Height = 9.75
$!PlotType  = Cartesian2D
$!FrameName  = 'uvel'
$!GlobalTime 
  SolutionTime = 14.9000000001000004
$!ActiveFieldMaps  =  [1]
$!FieldLayers ShowMesh = Yes
$!FieldMap [1]  Mesh{Color = Custom8}
$!TwoDAxis 
  XDetail
    {
    VarNum = 3
    }
  YDetail
    {
    VarNum = 2
    }
$!View Fit
$!TwoDAxis 
  AxisMode = Independent
  DepXToYRatio = 1
  ViewportPosition
    {
    X1 = 14
    Y1 = 10
    X2 = 96
    Y2 = 90.25
    }
  ViewportTopSnapTarget = 90.25
$!TwoDAxis 
  XDetail
    {
    CoordScale = Linear
    RangeMin = -1
    RangeMax = 1
    AutoGrid = No
    GRSpacing = 1
    Ticks
      {
      NumMinOrTicks = 1
      }
    TickLabel
      {
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        SizeUnits = Point
        Height = 18
        }
      Offset = 3
      }
    Title
      {
      TitleMode = UseText
      Text = '<i>u</i>'
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        IsBold = No
        SizeUnits = Point
        Height = 24
        }
      Offset = 7
      }
    AxisLine
      {
      Color = White
      }
    }
$!TwoDAxis 
  YDetail
    {
    CoordScale = Linear
    RangeMin = 0
    RangeMax = 1
    AutoGrid = No
    GRSpacing = 0.2
    Ticks
      {
      NumMinOrTicks = 1
      }
    TickLabel
      {
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        SizeUnits = Point
        Height = 18
        }
      }
    Title
      {
      ShowOnAxisLine = No
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        IsBold = No
        SizeUnits = Point
        Height = 24
        }
      Offset = 10
      }
    AxisLine
      {
      Color = White
      }
    }
$!FieldLayers 
  ShowShade = No
  ShowEdge = No
$!Linking 
  BetweenFrames
    {
    LinkSolutionTime = Yes
    }
### Frame Number 3 ###
$!CreateNewFrame 
$!FrameLayout 
  ShowBorder = No
  ShowHeader = No
  BackgroundColor = Black
  HeaderColor = Red
  XYPos
    {
    X = 1
    Y = 10
    }
  Width = 13
  Height = 3.75
$!ThreeDAxis 
  AspectRatioLimit = 25
  BoxAspectRatioLimit = 25
$!PlotType  = Cartesian2D
$!FrameName  = 'vvel'
$!GlobalTime 
  SolutionTime = 14.9000000001000004
$!ActiveFieldMaps  =  [1]
$!FieldLayers ShowMesh = Yes
$!FieldMap [1]  Mesh{Color = Custom7}
$!TwoDAxis 
  XDetail
    {
    VarNum = 1
    }
  YDetail
    {
    VarNum = 4
    }
$!View Fit
$!TwoDAxis 
  AxisMode = Independent
  DepXToYRatio = 1
  ViewportPosition
    {
    X1 = 8.3
    Y1 = 12
    X2 = 68.5
    Y2 = 99
    }
  ViewportTopSnapTarget = 99
$!TwoDAxis 
  XDetail
    {
    CoordScale = Linear
    RangeMin = 0
    RangeMax = 1
    AutoGrid = No
    GRSpacing = 0.2
    Ticks
      {
      NumMinOrTicks = 1
      }
    TickLabel
      {
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        SizeUnits = Point
        Height = 18
        }
      Offset = 3
      }
    Title
      {
      ShowOnAxisLine = No
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        IsBold = No
        SizeUnits = Point
        Height = 24
        }
      Offset = 15
      }
    AxisLine
      {
      Color = White
      }
    }
$!TwoDAxis 
  YDetail
    {
    CoordScale = Linear
    RangeMin = -1
    RangeMax = 1
    AutoGrid = No
    GRSpacing = 1
    Ticks
      {
      NumMinOrTicks = 1
      }
    TickLabel
      {
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        SizeUnits = Point
        Height = 18
        }
      }
    Title
      {
      TitleMode = UseText
      Text = '<i>v</i>'
      Color = White
      TextShape
        {
        FontFamily = 'Times'
        IsBold = No
        SizeUnits = Point
        Height = 24
        }
      Offset = 7
      }
    AxisLine
      {
      Color = White
      }
    }
$!FieldLayers 
  ShowShade = No
  ShowEdge = No
$!Linking 
  BetweenFrames
    {
    LinkSolutionTime = Yes
    }
