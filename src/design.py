import numpy as np
import random
from visualization import DrawBackground, DrawNewState, DrawImage, DrawText
import itertools as it


def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle


def findQuadrant(vector):
    quadrant = 0
    if vector[0] > 0 and vector[1] > 0:
        quadrant = 4
    elif vector[0] < 0 and vector[1] > 0:
        quadrant = 1
    elif vector[0] < 0 and vector[1] < 0:
        quadrant = 2
    elif vector[0] > 0 and vector[1] < 0:
        quadrant = 3
    else:
        quadrant = 0
    return quadrant


# class CreatMidLineCondition():
#     def __init__(self, direction, dimension):
#         self.direction = direction
#         self.dimension = dimension

#     def __call__(self, width, height, distanceDiff):
#         direction = random.choice(self.direction)
#         distanceDiff = random.choice([distanceDiff, -distanceDiff])
#         if direction == 0:
#             pacmanPosition = (floor(self.dimension / 2), random.randint(height, self.dimension - 1))
#             bean1Position = (pacmanPosition[0] - floor(width / 2), pacmanPosition[1] - height)
#             bean2Position = (pacmanPosition[0] + floor(width / 2), pacmanPosition[1] - height)
#         elif direction == 180:

#         elif direction == 90:
#         else:

#         return pacmanPosition, bean1Position, bean2Position, direction


class CreatStraightLineCondition():
    def __init__(self, direction, dimension):
        self.direction = direction
        self.dimension = dimension

    def __call__(self, width, height, distance, distanceDiff):
        direction = random.choice(self.direction)
        distanceDiff = random.choice([distanceDiff, -distanceDiff])
        if direction == 0:
            pacmanPosition = (random.randint(1, self.dimension - (distance + distanceDiff) - width - 2), random.randint(1, self.dimension - (distance + distanceDiff) - height - 2))
            bean1Position = (pacmanPosition[0] + width + (distance + distanceDiff), pacmanPosition[1] + height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] + height + distance)

        elif direction == 90:
            pacmanPosition = (random.randint(1 + (distance + distanceDiff) + width, self.dimension - 2), random.randint(1, self.dimension - 2 - (distance + distanceDiff) - height))
            bean1Position = (pacmanPosition[0] - width, pacmanPosition[1] + height + (distance + distanceDiff))
            bean2Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] + height)

        elif direction == 180:
            pacmanPosition = (random.randint(1 + (distance + distanceDiff) + width, self.dimension - 2), random.randint(1 + (distance + distanceDiff) + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] - width - (distance + distanceDiff), pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] - width, pacmanPosition[1] - height - distance)
        else:
            pacmanPosition = (random.randint(1, self.dimension - (distance + distanceDiff) - width - 2), random.randint(1 + (distance + distanceDiff) + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] + width + (distance + distanceDiff), pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] - height - distance)

        return bean1Position, bean2Position, pacmanPosition


class CreatExpCondition():
    def __init__(self, direction, dimension):
        self.direction = direction
        self.dimension = dimension

    def __call__(self, width, height, distance):
        direction = random.choice(self.direction)
        if direction == 45:
            pacmanPosition = (random.randint(1, self.dimension - distance - width - 2), random.randint(1, self.dimension - distance - height - 2))
            bean1Position = (pacmanPosition[0] + width + distance, pacmanPosition[1] + height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] + height + distance)

        elif direction == 135:
            pacmanPosition = (random.randint(1 + distance + width, self.dimension - 2), random.randint(1, self.dimension - 2 - distance - height))
            bean1Position = (pacmanPosition[0] - width, pacmanPosition[1] + height + distance)
            bean2Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] + height)

        elif direction == 225:
            pacmanPosition = (random.randint(1 + distance + width, self.dimension - 2), random.randint(1 + distance + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] - width, pacmanPosition[1] - height - distance)
        else:
            pacmanPosition = (random.randint(1, self.dimension - distance - width - 2), random.randint(1 + distance + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] + width + distance, pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] - height - distance)

        return pacmanPosition, bean1Position, bean2Position, direction


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    dis1 = np.linalg.norm(np.array(playerGrid) - np.array(target1), ord=1)
    dis2 = np.linalg.norm(np.array(playerGrid) - np.array(target2), ord=1)
    distanceDiff = dis1 - dis2
    rect1 = creatRect(playerGrid, target1)
    rect2 = creatRect(playerGrid, target2)
    avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
    avoidCommitmentZone.remove(tuple(playerGrid))
    return avoidCommitmentZone, distanceDiff


def isZoneALine(zone):
    zoneArr = np.array(zone).T
    if len(set(zoneArr[0])) == 1 or len(set(zoneArr[1])) == 1:
        return True
    else:
        return False


def createNoiseDesignValue(condition, blockNumber):
    noiseDesignValuesIndex = random.sample(list(range(len(condition))), blockNumber)
    noiseDesignValues = np.array(condition)[noiseDesignValuesIndex].flatten().tolist()
    noiseDesignValues.append('special')
    return noiseDesignValues


def createExpDesignValue(width, height, distance):
    shapeDesignValues = [[b, h, d] for b in width for h in height for d in distance]
    random.shuffle(shapeDesignValues)
    shapeDesignValues.append([random.choice(width), random.choice(height), random.choice(distance)])
    return shapeDesignValues


def createControlDesignValue(areaType, distanceDiff):

    return controlDesignValue


if __name__ == '__main__':
    dimension = 15
    direction = [0, 90, 180, 270]
    import pygame
    pygame.init()
    screenWidth = 600
    screenHeight = 600
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    textColorTuple = (255, 50, 50)

    drawBackground = DrawBackground(screen, dimension, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

    creatStraightLineCondition = CreatStraightLineCondition(direction, dimension)
    width, height, distance, distanceDiff = [2, 2, 2, 0]

    target1, target2, playerGrid = creatStraightLineCondition(width, height, distance, distanceDiff)

    import pandas as pd
    df = pd.DataFrame(columns=('playerGrid', 'target1', 'target2'))
    coordinations = tuple(it.product(range(1, dimension - 1), range(1, dimension - 1)))
    stateAll = list(it.combinations(coordinations, 3))
    # print(len(stateAll))
    # condition = []
    # index = 0
    distanceDiffList = [0, 2, 4]
    intentionedDisToTargetList = [2, 4, 6]
    areaSize = [[3, 4, 5, 6], [3, 4, 5, 6]]

    for diff in distanceDiffList:
        for state in stateAll:
            playerGrid, target1, target2 = state
            avoidCommitmentZone, distanceDiff = calculateAvoidCommitmnetZone(playerGrid, target1, target2)
            dis1 = np.linalg.norm(np.array(playerGrid) - np.array(target1), ord=1)
            dis2 = np.linalg.norm(np.array(playerGrid) - np.array(target2), ord=1)
            minDis = min(dis1, dis2)

            if len(avoidCommitmentZone) > 3 and distanceDiff == diff and minDis > 4 and isZoneALine(avoidCommitmentZone) == True:
                df = df.append(pd.DataFrame({'index': [index], 'distanceDiff': distanceDiff, 'playerGrid': [playerGrid], 'target1': [target1], 'target2': [target2]}))
                index += 1

                fileName =
    df.to_csv('NoAvoidCommitmentZone.csv')

    # pause = True
    # while pause:
    #     drawNewState(target1, target2, playerGrid)
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             pause = False
    # pygame.quit()

    from collections import namedtuple
    condition = namedtuple('condition', 'name areaType distanceDiff areaSize intentionedDisToTarget')

    expCondition = condition(name='expCondition', areaType='rect', distanceDiff=[0], areaSize=[[3, 4, 5, 6], [3, 4, 5, 6]], intentionedDisToTarget=[2, 4, 6])

    rectCondition = condition(name='control', areaType='rect', distanceDiff=[2, 4], areaSize=[[3, 4, 5, 6], [3, 4, 5, 6]], intentionedDisToTarget=[2, 4, 6])
    straightLineCondition = condition(name='control', areaType='line', distanceDiff=[0, 2, 4], areaSize=[3, 4, 5, 6, 7, 8], intentionedDisToTarget=[2, 4, 6])
    midLineCondition = condition(name='control', areaType='line', distanceDiff=[0, 2, 4], areaSize=[3, 4, 5, 6, 7, 8], intentionedDisToTarget=[2, 4, 6])
    noAreaCondition = condition(name='control', areaType='none', distanceDiff=[0, 2, 4], areaSize=[3, 4, 5, 6, 7, 8], intentionedDisToTarget=[2, 4, 6])

    print(noAreaCondition.areaSize)
