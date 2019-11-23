import os
DIRNAME = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import unittest
from ddt import ddt, data, unpack
from dataAnalysis.dataAnalysis import calculateAvoidCommitmentRatio, calculateFirstOutZoneRatio, calculateFirstIntentionRatio,calculateFirstIntention


@ddt
class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        self.testParameter = 0

    @data(([(1, 2), [3, 2], [4, 2], [5, 5], [5, 2]], [(3, 2), (4, 2), (5, 2), (5, 3), (5, 4)], 0.75)
          )
    @unpack
    def testCalculateAvoidCommitmnetZone(self, trajectory, zone, groundTruthRatio):
        avoidCommitmentRatio = calculateAvoidCommitmentRatio(trajectory, zone)
        truthValue = np.array_equal(avoidCommitmentRatio, groundTruthRatio)
        self.assertTrue(truthValue)

    @data(([(1, 2), [3, 2], [4, 2], [5, 5], [5, 2]], [(3, 2), (4, 2), (5, 2), (5, 3), (5, 4)], 0.5))
    @unpack
    def testCalculateFirstOutZoneRatio(self, trajectory, zone, groundTruthRatio):
        avoidCommitmentRatio = calculateFirstOutZoneRatio(trajectory, zone)
        truthValue = np.array_equal(avoidCommitmentRatio, groundTruthRatio)
        self.assertTrue(truthValue)

    @data(([0, 0, 0, 0, 2, 0, 0, 2, 2, 2], 0.5))
    @unpack
    def testCalculateFirstIntentionRatio(self, goalList, groundTruthRatio):
        avoidCommitmentRatio = calculateFirstIntentionRatio(goalList)
        truthValue = np.array_equal(avoidCommitmentRatio, groundTruthRatio)
        self.assertTrue(truthValue)

    @data(([0, 0, 0, 0, 2, 0, 0, 2, 2, 2], 2),
        ([0, 0, 0, 0,1, 2, 0, 0, 2, 2, 2], 1),
        ([0, 0, 0, 0, 0, 0, 0, 0], 0))
    @unpack
    def testCalculateFirstIntention(self, goalList, groundTruthGoal):
        firstIntention = calculateFirstIntention(goalList)
        truthValue = np.array_equal(firstIntention, groundTruthGoal)
        self.assertTrue(truthValue)

if __name__ == '__main__':
    unittest.main()
