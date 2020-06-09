import os
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import pygame as pg
import collections as co
import numpy as np
import pickle
from itertools import permutations
from collections import namedtuple
import random
import pandas as pd

from src.writer import WriteDataFrameToCSV
from src.visualization import InitializeScreen, DrawBackground, DrawNewState, DrawImage, DrawText
from src.controller import SampleSoftmaxAction, ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary, backToZoneNoise, SampleToZoneNoise, AimActionWithNoise, InferGoalPosterior, ModelControllerWithGoal, ModelControllerOnlineReward, SoftmaxRLPolicy, SoftmaxGoalPolicy
from src.simulationTrial import NormalTrialOnline, SpecialTrialOnline, NormalTrialWithGoal, SpecialTrialWithGoal, NormalTrialRewardOnline, SpecialTrialRewardOnline
from src.experiment import ModelExperiment
from src.design import CreatExpCondition, SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue
from machinePolicy.valueIteration import RunVI


def main():
    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    dataPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'conditionData'))
    df = pd.read_csv(os.path.join(dataPath, 'DesignConditionForAvoidCommitmentZone.csv'))
    df['intentionedDisToTargetMin'] = df.apply(lambda x: x['minDis'] - x['avoidCommitmentZone'], axis=1)

    gridSize = 15

    screenWidth = 600
    screenHeight = 600
    fullScreen = False
    renderOn = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    pg.mouse.set_visible(False)

    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

    width = [3, 4, 5]
    height = [3, 4, 5]
    intentionDis = [2, 4, 6]
    direction = [45, 135, 225, 315]

    distanceDiffList = [0, 2, 4]
    minDisList = range(5, 15)
    intentionedDisToTargetList = [2, 4, 6]
    rectAreaSize = [6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 36]
    lineAreaSize = [4, 5, 6, 7, 8, 9, 10]

    condition = namedtuple('condition', 'name areaType distanceDiff minDis areaSize intentionedDisToTarget')

    expCondition = condition(name='expCondition', areaType='rect', distanceDiff=[0], minDis=minDisList, areaSize=rectAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    rectCondition = condition(name='controlRect', areaType='rect', distanceDiff=[2, 4], minDis=minDisList, areaSize=rectAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    straightLineCondition = condition(name='straightLine', areaType='straightLine', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=lineAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    midLineCondition = condition(name='MidLine', areaType='midLine', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=lineAreaSize, intentionedDisToTarget=intentionedDisToTargetList)
    noAreaCondition = condition(name='noArea', areaType='none', distanceDiff=distanceDiffList, minDis=minDisList, areaSize=[0], intentionedDisToTarget=intentionedDisToTargetList)

    Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGird15_policy.pkl"), "rb"))
    # policy = None
    actionSpace = ((1, 0), (0, 1), (-1, 0), (0, -1))
    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    noiseActionSpace = [(0, -2), (0, 2), (-2, 0), (2, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    normalNoise = NormalNoise(noiseActionSpace, gridSize)
    sampleToZoneNoise = SampleToZoneNoise(noiseActionSpace)

    # goal_Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))

    initPrior = [0.5, 0.5]

    # softmaxBeta = 2.5
    # goalPolicy = SoftmaxGoalPolicy(goal_Q_dict, softmaxBeta)
    priorBeta = 5

    commitBetaList = np.arange(1, 10, 1)

    rewardVarianceList = [50]
    softmaxBetaList = np.round(np.arange(0.4, 0.5, 0.01), decimals=2)
    print(softmaxBetaList)
    softmaxBetaList = [-1, 1, 2]
    for softmaxBeta in softmaxBetaList:
        # policy = SoftmaxRLPolicy(Q_dict, softmaxBeta)
        # for commitBeta in commitBetaList:
        for i in range(33):
            print(i)
            expDesignValues = [[b, h, d] for b in width for h in height for d in intentionDis]
            numExpTrial = len(expDesignValues)
            random.shuffle(expDesignValues)
            expDesignValues.append(random.choice(expDesignValues))
            createExpCondition = CreatExpCondition(direction, gridSize)
            samplePositionFromCondition = SamplePositionFromCondition(df, createExpCondition, expDesignValues)
            numExpTrial = len(expDesignValues) - 1
            numControlTrial = int(numExpTrial * 2 / 3)
            expTrials = [expCondition] * numExpTrial
            conditionList = list(expTrials + [rectCondition] * numExpTrial + [straightLineCondition] * numControlTrial + [midLineCondition] * numControlTrial + [noAreaCondition] * numControlTrial)
            numNormalTrials = len(conditionList)

            random.shuffle(conditionList)
            conditionList.append(expCondition)

            numTrialsPerBlock = 3
            noiseCondition = list(permutations([1, 2, 0], numTrialsPerBlock))
            noiseCondition.append((1, 1, 1))
            blockNumber = int(numNormalTrials / numTrialsPerBlock)
            noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)

    # deubg
            # conditionList = [expCondition] * 27
            # noiseDesignValues = ['special'] * 27

    # model simulation
            noise = 0.1
            gamma = 0.99
            goalReward = 10
            runModel = RunVI(gridSize, actionSpace, noiseActionSpace, noise, gamma, goalReward)
            sampleAction = SampleSoftmaxAction(softmaxBeta)
            normalTrial = NormalTrialOnline(sampleAction, drawNewState, drawText, normalNoise, checkBoundary)
            specialTrial = SpecialTrialOnline(sampleAction, drawNewState, drawText, sampleToZoneNoise, checkBoundary)

            experimentValues = co.OrderedDict()
            experimentValues["name"] = "softmaxBeta" + str(softmaxBeta) + '_' + str(i)
            resultsDirPath = os.path.join(resultsPath, "softmaxBeta" + str(softmaxBeta))

            if not os.path.exists(resultsDirPath):
                os.makedirs(resultsDirPath)
            writerPath = os.path.join(resultsDirPath, experimentValues["name"] + '.csv')
            writer = WriteDataFrameToCSV(writerPath)

            experiment = ModelExperiment(runModel, sampleAction, normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath)
            experiment(noiseDesignValues, conditionList)


if __name__ == "__main__":
    main()
