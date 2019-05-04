import sys
if sys.version.startswith("2.7"):
	from MapMaker import mapmaker
	from MappingClass import mapclass
	from ReplaceColor import ReplaceColor
	from GitMetadata import gitmetadata
	from HistogramMaker import histmaker
	from ReplaceHexColors import ReplaceHexColor
elif sys.version.startswith("3.7"):
	from myfunctions.PlotFunctions.MapMaker import mapmaker
	from myfunctions.PlotFunctions.MappingClass import mapclass
	from myfunctions.PlotFunctions.ReplaceColor import ReplaceColor
	from myfunctions.PlotFunctions.GitMetadata import gitmetadata
	from myfunctions.PlotFunctions.HistogramMaker import histmaker
	from myfunctions.PlotFunctions.ReplaceHexColors import ReplaceHexColor

__all__ = ['mapmaker', 'mapclass', 'ReplaceColor', 'gitmetadata', 'histmaker', "ReplaceHexColor"]