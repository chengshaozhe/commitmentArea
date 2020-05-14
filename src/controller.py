import numpy as np
import pygame as pg
import random


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


def inferGoal(originGrid, aimGrid, targetGridA, targetGridB):
    pacmanBean1aimDisplacement = calculateGridDis(targetGridA, aimGrid)
    pacmanBean2aimDisplacement = calculateGridDis(targetGridB, aimGrid)
    pacmanBean1LastStepDisplacement = calculateGridDis(targetGridA, originGrid)
    pacmanBean2LastStepDisplacement = calculateGridDis(targetGridB, originGrid)
    bean1Goal = pacmanBean1LastStepDisplacement - pacmanBean1aimDisplacement
    bean2Goal = pacmanBean2LastStepDisplacement - pacmanBean2aimDisplacement
    if bean1Goal > bean2Goal:
        goal = 1
    elif bean1Goal < bean2Goal:
        goal = 2
    else:
        goal = 0
    return goal


def calculateSoftmaxProbability(acionValues, beta):
    expont = [min(700, i) for i in np.multiply(beta, acionValues)]
    newProbabilityList = list(np.divide(np.exp(expont), np.sum(np.exp(expont))))

    return newProbabilityList


class NormalNoise():
    def __init__(self, actionSpace, gridSize):
        self.actionSpace = actionSpace
        self.gridSize = gridSize

    def __call__(self, playerGrid, action, noiseStep, stepCount):
        if stepCount in noiseStep:
            realAction = random.choice(self.actionSpace)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, realAction


class AimActionWithNoise():
    def __init__(self, actionSpace, gridSize):
        self.actionSpace = actionSpace
        self.gridSize = gridSize

    def __call__(self, playerGrid, action, noiseStep, stepCount):
        if stepCount in noiseStep:
            actionSpace = self.actionSpace.copy()
            actionSpace.remove(action)
            actionList = [str(action) for action in actionSpace]
            actionStr = np.random.choice(actionList)
            realAction = eval(actionStr)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, realAction


def backToZoneNoise(playerGrid, trajectory, zone, noiseStep, firstIntentionFlag):
    realPlayerGrid = None

    if playerGrid not in zone and tuple(trajectory[-2]) in zone and not firstIntentionFlag:
        realPlayerGrid = trajectory[-3]
        noiseStep = len(trajectory)
        firstIntentionFlag = True
    return realPlayerGrid, noiseStep, firstIntentionFlag


class SampleToZoneNoise:
    def __init__(self, noiseActionSpace):
        self.noiseActionSpace = noiseActionSpace

    def __call__(self, playerGrid, trajectory, zone, noiseStep, firstIntentionFlag):
        realPlayerGrid = None
        if playerGrid not in zone and tuple(trajectory[-2]) in zone and not firstIntentionFlag:
            possibleGrid = (tuple(np.add(playerGrid, action)) for action in self.noiseActionSpace)
            realPlayerGrids = list(filter(lambda x: x in zone, possibleGrid))
            realPlayerGrid = random.choice(realPlayerGrids)
            noiseStep = len(trajectory)
            firstIntentionFlag = True
        return realPlayerGrid, noiseStep, firstIntentionFlag


def isGridsNotALine(playerGrid, bean1Grid, bean2Grid):
    line = np.array((playerGrid, bean1Grid, bean2Grid)).T
    if len(set(line[0])) != len(line[0]) or len(set(line[1])) != len(line[1]):
        return False
    else:
        return True


class SampleToZoneNoiseNoLine:
    def __init__(self, noiseActionSpace):
        self.noiseActionSpace = noiseActionSpace

    def __call__(self, playerGrid, bean1Grid, bean2Grid, trajectory, zone, noiseStep, firstIntentionFlag):
        realPlayerGrid = None
        if playerGrid not in zone and tuple(trajectory[-2]) in zone and not firstIntentionFlag:
            possibleGrid = (tuple(np.add(playerGrid, action)) for action in self.noiseActionSpace)
            realPlayerGrids = tuple(filter(lambda x: x in zone, possibleGrid))
            noLineGrids = list(filter(lambda x: isGridsNotALine(x, bean1Grid, bean2Grid), realPlayerGrids))
            if noLineGrids:
                realPlayerGrid = random.choice(noLineGrids)
            else:
                realPlayerGrid = random.choice(realPlayerGrids)
            noiseStep = len(trajectory)
            firstIntentionFlag = True
        return realPlayerGrid, noiseStep, firstIntentionFlag


def selectActionMinDistanceFromTarget(goal, playerGrid, bean1Grid, bean2Grid, actionSpace):
    allPosiibilePlayerGrid = [np.add(playerGrid, action) for action in actionSpace]
    allActionGoal = [inferGoal(playerGrid, possibleGrid, bean1Grid, bean2Grid) for possibleGrid in allPosiibilePlayerGrid]
    if goal == 1:
        realActionIndex = allActionGoal.index(2)
    else:
        realActionIndex = allActionGoal.index(1)
    realAction = actionSpace[realActionIndex]
    return realAction


class AwayFromTheGoalNoise():
    def __init__(self, actionSpace, gridSize):
        self.actionSpace = actionSpace
        self.gridSize = gridSize

    def __call__(self, playerGrid, bean1Grid, bean2Grid, action, goal, firstIntentionFlag, noiseStep, stepCount):
        if goal != 0 and not firstIntentionFlag:
            noiseStep.append(stepCount)
            firstIntentionFlag = True
            realAction = selectActionMinDistanceFromTarget(goal, playerGrid, bean1Grid, bean2Grid, self.actionSpace)
        else:
            realAction = action
        realPlayerGrid = tuple(np.add(playerGrid, realAction))
        return realPlayerGrid, firstIntentionFlag, noiseStep


class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        adjustedX, adjustedY = position
        if position[0] >= self.xMax:
            adjustedX = self.xMax
        if position[0] <= self.xMin:
            adjustedX = self.xMin
        if position[1] >= self.yMax:
            adjustedY = self.yMax
        if position[1] <= self.yMin:
            adjustedY = self.yMin
        checkedPosition = (adjustedX, adjustedY)
        return checkedPosition


class HumanController():
    def __init__(self, actionDict):
        self.actionDict = actionDict

    def __call__(self, playerGrid, targetGrid1, targetGrid2):
        action = [0, 0]
        pause = True
        while pause:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key in self.actionDict.keys():
                        action = self.actionDict[event.key]
                        aimePlayerGrid = tuple(np.add(playerGrid, action))
                        pause = False
                    if event.key == pg.K_ESCAPE:
                        exit()
        return aimePlayerGrid, action


class ModelController():
    def __init__(self, policy, gridSize, softmaxBeta):
        self.policy = policy
        self.gridSize = gridSize
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2):
        try:
            policyForCurrentStateDict = self.policy[(playerGrid, (targetGrid1, targetGrid2))]
        except KeyError as e:
            policyForCurrentStateDict = self.policy[(playerGrid, (targetGrid2, targetGrid1))]
        if self.softmaxBeta < 0:
            actionMaxList = [action for action in policyForCurrentStateDict.keys() if
                             policyForCurrentStateDict[action] == np.max(list(policyForCurrentStateDict.values()))]
            action = random.choice(actionMaxList)
        else:

            actionValue = list(policyForCurrentStateDict.values())
            actionValue = [min(100, v) for v in actionValue]
            softmaxProbabilityList = calculateSoftmaxProbability(actionValue, self.softmaxBeta)
            action = list(policyForCurrentStateDict.keys())[
                list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
        aimePlayerGrid = tuple(np.add(playerGrid, action))
        # pg.time.delay(500)
        return aimePlayerGrid, action


def chooseMaxAcion(actionDict):
    actionMaxList = [action for action in actionDict.keys() if
                     actionDict[action] == np.max(list(actionDict.values()))]
    action = random.choice(actionMaxList)
    return action


def chooseSoftMaxAction(actionDict, softmaxBeta):
    actionValue = list(actionDict.values())
    softmaxProbabilityList = calculateSoftmaxProbability(actionValue, softmaxBeta)
    action = list(actionDict.keys())[
        list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
    return action


def calBasePolicy(posteriorList, actionProbList):
    basePolicyList = [np.multiply(goalProb, actionProb) for goalProb, actionProb in zip(posteriorList, actionProbList)]
    basePolicy = np.sum(basePolicyList, axis=0)
    return basePolicy


class SoftmaxPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1):
        actionDict = self.Q_dict[(playerGrid, target1)]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict

class InferGoalPosterior:
    def __init__(self, goalPolicy, commitBeta):
        self.goalPolicy = goalPolicy
        self.commitBeta = commitBeta

    def __call__(self, playerGrid, action, target1, target2, priorList):
        targets = list([target1, target2])

        likelihoodList = [self.goalPolicy.get(action) for goal in targets]

        posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]

        evidence = sum(posteriorUnnormalized)
        posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]

        # diff = abs(posteriorList[0] - posteriorList[1])
        # if diff < self.commitBeta / 10:
        #     posteriorUnnormalized = calculateSoftmaxProbability(posteriorUnnormalized, 1 / self.commitBeta)
        # else:
        #     posteriorUnnormalized = calculateSoftmaxProbability(posteriorUnnormalized, self.commitBeta)
        # evidence = sum(posteriorUnnormalized)
        # posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]

        # diff = abs(posteriorList[0] - posteriorList[1])
        # diifScale = sigmoidScale(diff, commitBeta)
        # if diff < self.commitBeta / 10:
        #     posteriorList = calculateSoftmaxProbability(posteriorUnnormalized, 1 / self.commitBeta)
        # else:
        #     posteriorList = calculateSoftmaxProbability(posteriorUnnormalized, self.commitBeta)


        diff = abs(posteriorList[0] - posteriorList[1])
        if diff < self.commitBeta/150:
            posteriorList = calculateSoftmaxProbability(posteriorList, 1 / (self.commitBeta))
        else:
            posteriorList = calculateSoftmaxProbability(posteriorList, self.commitBeta*150)

        return posteriorList


def sigmoid(x):
    return 1 / (1 + np.exp(- x))


def sigmoidScale(diff, commitBeta):
    return 1 / (1 + np.exp(- 10 * (x - commitBeta/10)))


def normalizeProb(unnormProb):
    prob = [p / sum(unnormProb) for p in unnormProb]
    return prob


class ModelControllerWithGoal:
    def __init__(self, gridSize, softmaxBeta, policy, goalPolicy, commitBeta):
        self.gridSize = gridSize
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.softmaxBeta = softmaxBeta
        self.policy = policy
        self.goalPolicy = goalPolicy
        self.commitBeta = commitBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2, priorList):
        targets = list([targetGrid1, targetGrid2])
        actionProbList = [list(self.goalPolicy.values()) for goal in targets]
        actionKeys = self.goalPolicy[playerGrid, targetGrid1].keys()
        actionProbs = calBasePolicy(priorList, actionProbList)
        actionDict = dict(zip(actionKeys, actionProbs))

        # softPriorList = calculateSoftmaxProbability(priorList, self.commitBeta)
        # softPriorList = priorList

        # if max(softPriorList) > sigmoid(self.commitBeta):
        #     softPriorList = calculateSoftmaxProbability(priorList, self.commitBeta * 100)
        #     actionProbs = calBasePolicy(softPriorList, actionProbList)
        #     # goal = targets[np.argmax(priorList)]
        #     # actionProbs = self.goalPolicy[playerGrid, goal].values()
        #     actionDict = dict(zip(actionKeys, actionProbs))
        # else:
        #     priorList = calculateSoftmaxProbability(priorList, 100 / self.commitBeta)
        #     actionProbs = calBasePolicy(priorList, actionProbList)
        #     actionDict = dict(zip(actionKeys, actionProbs))
        #     # actionDict = self.policy[(playerGrid, tuple(sorted(targets)))]

        # diff = abs(priorList[0] - priorList[1])
        # if diff < self.commitBeta:
        #     actionProbs = calBasePolicy([0.5, 0.5], actionProbList)
        # else:
        #     priorList = np.array(priorList) ** (10*self.commitBeta) / sum(np.array(priorList) ** (10*self.commitBeta))
        #     actionProbs = calBasePolicy(priorList, actionProbList)
        # actionDict = dict(zip(actionKeys, actionProbs))

        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


class ModelControllerOnlineReward:
    def __init__(self, gridSize, softmaxBeta, runVI):
        self.gridSize = gridSize
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.softmaxBeta = softmaxBeta
        self.runVI = runVI

    def __call__(self, playerGrid, targetGrid1, targetGrid2, goalRewardList):
        QDict = self.runVI((targetGrid1, targetGrid2), goalRewardList)
        actionDict = QDict[(playerGrid, (targetGrid1, targetGrid2))]

        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


if __name__ == "__main__":
    pg.init()
    screenWidth = 720
    screenHeight = 720
    screen = pg.display.set_mode((screenWidth, screenHeight))
    gridSize = 20
    leaveEdgeSpace = 2
    lineWidth = 2
    backgroundColor = [188, 188, 0]
    lineColor = [255, 255, 255]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    targetGridA = [5, 5]
    targetGridB = [15, 5]
    playerGrid = [10, 15]
    currentScore = 5
    textColorTuple = (255, 50, 50)
    stopwatchEvent = pg.USEREVENT + 1
    stopwatchUnit = 10
    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    finishTime = 90000
    currentStopwatch = 32000

    drawBackground = Visualization.DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor,
                                                  lineWidth, textColorTuple)
    drawNewState = Visualization.DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius,
                                              playerRadius)

    getHumanAction = HumanController(gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)
    import pickle

    policy = pickle.load(open("SingleWolfTwoSheepsGrid15.pkl", "rb"))
    getModelAction = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)

    # [playerNextPosition,action,newStopwatch]=getHumanAction(targetGridA, targetGridB, playerGrid, currentScore, currentStopwatch)
    [playerNextPosition, action, newStopwatch] = getModelAction(targetGridA, targetGridB, playerGrid, currentScore, currentStopwatch)
    print(playerNextPosition, action, newStopwatch)

    pg.quit()
