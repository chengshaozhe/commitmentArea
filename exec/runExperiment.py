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
from src.visualization import DrawBackground, DrawNewState, DrawImage, DrawText
from src.controller import HumanController, ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary, backToZoneNoise
from src.trial import NormalTrial, SpecialTrial
from src.experiment import Experiment
from src.design import CreatExpCondition, SamplePositionFromCondition, createNoiseDesignValue, createExpDesignValue


def main():
    gridSize = 15

    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    dataPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'conditionData'))
    df = pd.read_csv(os.path.join(dataPath, 'DesignConditionForAvoidCommitmentZone.csv'))
    df['intentionedDisToTargetMin'] = df.apply(lambda x: x['minDis'] - x['avoidCommitmentZone'], axis=1)

    width = [3, 4, 5]
    height = [3, 4, 5]
    intentionDis = [2, 4, 6]
    direction = [45, 135, 225, 315]

    expDesignValues = [[b, h, d] for b in width for h in height for d in intentionDis]
    expDesignValues.append(random.choice(expDesignValues))
    random.shuffle(expDesignValues)
    numExpTrial = len(expDesignValues)

    createExpCondition = CreatExpCondition(direction, gridSize)
    samplePositionFromCondition = SamplePositionFromCondition(df, createExpCondition, expDesignValues)

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

    numControlTrial = int(numExpTrial * 2 / 3)
    expTrials = [expCondition] * numExpTrial
    random.shuffle(expTrials)
    conditionList = list(expTrials[:-1] + [rectCondition] * numExpTrial + [straightLineCondition] * numControlTrial + [midLineCondition] * numControlTrial + [noAreaCondition] * numControlTrial)
    random.shuffle(conditionList)
    numNormalTrials = len(conditionList)
    conditionList.append(expCondition)

    noiseCondition = list(permutations([1, 2, 0], 3))
    noiseCondition.append((1, 1, 1))
    blockNumber = int(numNormalTrials / 3)
    noiseDesignValues = np.array([random.choice(noiseCondition) for _ in range(blockNumber)]).flatten().tolist()
    noiseDesignValues.append('special')

    pg.init()
    screenWidth = 600
    screenHeight = 600
    screen = pg.display.set_mode((screenWidth, screenHeight))
    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    experimentValues = co.OrderedDict()
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    experimentValues["name"] = 'test'
    writerPath = os.path.join(resultsPath, experimentValues["name"] + '.csv')
    writer = WriteDataFrameToCSV(writerPath)
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)

    # policy = pickle.load(open(machinePolicyPath + "noise0.1SingleWolfTwoSheepsGrid15allPosition.pkl", "rb"))
    # softmaxBeta = -1
    # modelController = ModelController(policy, gridSize, softmaxBeta)

    pygameActionDict = {pg.K_UP: (0, -1), pg.K_DOWN: (0, 1), pg.K_LEFT: (-1, 0), pg.K_RIGHT: (1, 0)}
    humanController = HumanController(pygameActionDict)
    controller = humanController

    checkBoundary = CheckBoundary([0, gridSize - 1], [0, gridSize - 1])
    actionSpace = list(pygameActionDict.values())
    noiseActionSpace = [(0, -2), (0, 2), (-2, 0), (2, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    normalNoise = NormalNoise(noiseActionSpace, gridSize)
    awayFromTheGoalNoise = AwayFromTheGoalNoise(actionSpace, gridSize)
    normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
    specialTrial = SpecialTrial(controller, drawNewState, drawText, backToZoneNoise, checkBoundary)
    experiment = Experiment(normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath)
    drawImage(introductionImage)
    experiment(noiseDesignValues, conditionList)
    drawImage(finishImage)


if __name__ == "__main__":
    main()
