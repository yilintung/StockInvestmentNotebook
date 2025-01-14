#
# @file __init__.py
#

import myutils.TurningPoints
import myutils.DirectionalChange
import myutils.PerceptuallyImportant
import myutils.RankEvaluate
import myutils.TrendlineAutomation
import myutils.HeadShoulders
import myutils.StockGPT

import myutils.Eval


FindingTurningPoints = TurningPoints.FindingTurningPoints
FindingDirectionalChangePoints = DirectionalChange.FindingDirectionalChangePoints
FindingPerceptuallyImportantPoints = PerceptuallyImportant.FindingPerceptuallyImportantPoints
StockRankEvaluate = RankEvaluate.StockRankEvaluate
FitTrendlines = TrendlineAutomation.FitTrendlines
FindingHeadShoulderPatterns = HeadShoulders.FindingHeadShoulderPatterns
StockGPT = StockGPT.StockGPT

DetectTurningPoints = Eval.DetectTurningPoints
TrendlineAutomation = Eval.TrendlineAutomation
test_DetectTurningPoints = Eval.test_DetectTurningPoints
test_TrendlineAutomation = Eval.test_TrendlineAutomation