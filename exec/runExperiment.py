import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import pygame as pg
import collections as co
import numpy as np
import pickle
from itertools import permutations

from src.visualization import DrawBackground, DrawNewState, DrawImage, DrawText
from src.controller import HumanController, ModelController, NormalNoise, AwayFromTheGoalNoise, CheckBoundary
from src.updateWorld import createNoiseDesignValue, createShapeDesignValue, UpdateWorld
from src.writer import WriteDataFrameToCSV
from src.trial import NormalTrial, SpecialTrial
from src.experiment import Experiment


def main():
    dimension = 15
    minDistanceBetweenGrids = 5
    blockNumber = 5
    noiseCondition = list(permutations([1, 2, 0], 3))
    noiseCondition.append((1, 1, 1))
    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    width = [4, 5, 6]
    height = [4, 5, 6]
    distance = [4, 5, 6]
    direction = [45, 135, 225, 315]
    noiseDesignValues = createNoiseDesignValue(noiseCondition, blockNumber)
    shapeDesignValues = createShapeDesignValue(width, height, distance)
    # controlDesignValues = createControlDesignValue()

    updateWorld = UpdateWorld(direction, dimension)
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
    softmaxBeta = -1
    experimentValues = co.OrderedDict()
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    experimentValues["name"] = 'test'
    writerPath = resultsPath + experimentValues["name"] + '.csv'
    writer = WriteDataFrameToCSV(writerPath)
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawText = DrawText(screen, drawBackground)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    # policy = pickle.load(open(machinePolicyPath + "noise0.1SingleWolfTwoSheepsGrid15allPosition.pkl", "rb"))
    # modelController = ModelController(policy, dimension, softmaxBeta)
    humanController = HumanController(dimension)
    checkBoundary = CheckBoundary([0, dimension - 1], [0, dimension - 1])
    controller = humanController
    normalNoise = NormalNoise(controller)
    awayFromTheGoalNoise = AwayFromTheGoalNoise(controller)
    normalTrial = NormalTrial(controller, drawNewState, drawText, normalNoise, checkBoundary)
    specialTrial = SpecialTrial(controller, drawNewState, drawText, awayFromTheGoalNoise, checkBoundary)
    experiment = Experiment(normalTrial, specialTrial, writer, experimentValues, updateWorld, drawImage, resultsPath,
                            minDistanceBetweenGrids)
    drawImage(introductionImage)
    experiment(noiseDesignValues, shapeDesignValues)
    drawImage(finishImage)


if __name__ == "__main__":
    main()
