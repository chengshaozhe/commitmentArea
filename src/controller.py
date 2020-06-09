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

    expont = np.multiply(beta, acionValues)
    # expont = [min(500, i) for i in np.multiply(beta, acionValues)]
    newProbabilityList = list(np.divide(np.exp(expont), np.sum(np.exp(expont))))

    return newProbabilityList


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


class SoftmaxGoalPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1):
        actionDict = self.Q_dict[(playerGrid, target1)]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class SoftmaxRLPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1, target2):
        actionDict = self.Q_dict[(playerGrid, tuple(sorted((target1, target2))))]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class SampleSoftmaxAction:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, Q_dict, playerGrid):
        actionKeys = list(Q_dict[playerGrid].keys())
        actionValues = list(Q_dict[playerGrid].values())

        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        action = actionKeys[
            list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
        return action


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
        actionProbs = self.policy(playerGrid, targetGrid1, targetGrid2).values()
        actionKeys = self.policy(playerGrid, targetGrid1, targetGrid2).keys()
        actionDict = dict(zip(actionKeys, actionProbs))
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = sampleAction(actionDict)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        # pg.time.delay(500)
        return aimePlayerGrid, action


class GoalModelController():
    def __init__(self, policy, gridSize, softmaxBeta):
        self.policy = policy
        self.gridSize = gridSize
        self.actionSpace = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, targetGrid):
        actionDict = self.policy(playerGrid, targetGrid)
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = sampleAction(actionDict)

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


def sampleAction(actionDict):
    actionProbs = list(actionDict.values())
    action = list(actionDict.keys())[
        list(np.random.multinomial(1, actionProbs)).index(1)]
    return action


# def goalCommited(probList, commitBeta):
#     a, b = probList
#     if a > 0.5:
#         aNew = sigmoidScale(a, commitBeta)
#         bNew = 1 - aNew
#     else:
#         bNew = sigmoidScale(b, commitBeta)
#         aNew = 1 - bNew
#     return [aNew, bNew]

def commitSigmoid(x, commitBeta):  # [1,5,10,20]
    return 1 / (1 + np.exp(- commitBeta * (x - 0.5)))


def goalCommit(intention, commitBeta):
    commitedIntention = [commitSigmoid(x, commitBeta) for x in intention]
    return commitedIntention


class InferGoalPosterior:
    def __init__(self, goalPolicy, commitBeta):
        self.goalPolicy = goalPolicy
        self.commitBeta = commitBeta

    def __call__(self, playerGrid, action, target1, target2, priorList):
        targets = list([target1, target2])

        priorList = goalCommit(priorList, self.commitBeta)
        likelihoodList = [self.goalPolicy(playerGrid, goal).get(action) for goal in targets]
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]

        evidence = sum(posteriorUnnormalized)
        posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]

        return posteriorList


def sigmoid(x):
    return 1 / (1 + np.exp(- x))


def sigmoidScale(diff, commitBeta):
    return 1 / (1 + np.exp(- 10 * (x - commitBeta / 10)))


def normalizeProb(unnormProb):
    prob = [p / sum(unnormProb) for p in unnormProb]
    return prob


def calBasePolicy(posteriorList, actionProbList):
    basePolicyList = [np.multiply(goalProb, actionProb) for goalProb, actionProb in zip(posteriorList, actionProbList)]
    basePolicy = np.sum(basePolicyList, axis=0)
    return basePolicy


class ModelControllerWithGoal:
    def __init__(self, gridSize, softmaxBeta, goalPolicy, Q_dict, commitBeta):
        self.gridSize = gridSize
        self.softmaxBeta = softmaxBeta
        self.goalPolicy = goalPolicy
        self.Q_dict = Q_dict
        self.commitBeta = commitBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2, priorList):
        targets = list([targetGrid1, targetGrid2])

        actionProbList = [list(self.goalPolicy(playerGrid, goal).values()) for goal in targets]
        actionKeys = self.Q_dict[playerGrid, targetGrid1].keys()

        # actionProbs = calBasePolicy(priorList, actionProvList)
        # actionDict = dict(zip(actionKeys, actionProbs))

        goal = list(np.random.multinomial(1, priorList)).index(1)
        actionProb = actionProbList[goal]
        actionDict = dict(zip(actionKeys, actionProb))

        # softPriorList = calculateSoftmaxProbability(priorList, self.commitBeta)
        # softPriorList = priorList

        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = sampleAction(actionDict)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


class ModelControllerOnlineReward:
    def __init__(self, softmaxBeta, goalPolicy):
        self.softmaxBeta = softmaxBeta
        self.goalPolicy = goalPolicy

    def __call__(self, playerGrid, targetGrid1, targetGrid2, goalRewardList):
        actionDict = runVI((targetGrid1, targetGrid2), goalRewardList)
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action


class ModelControllerOnline:
    def __init__(self, softmaxBeta):
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, targetGrid1, targetGrid2, obstacles):
        actionDict = runVI((targetGrid1, targetGrid2, obstacles))
        if self.softmaxBeta < 0:
            action = chooseMaxAcion(actionDict)
        else:
            action = chooseSoftMaxAction(actionDict, self.softmaxBeta)

        aimePlayerGrid = tuple(np.add(playerGrid, action))
        return aimePlayerGrid, action
