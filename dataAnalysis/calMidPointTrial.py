import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import ttest_ind, entropy
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score as KL
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone


def calculateSoftmaxProbability(acionValues, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, acionValues)), np.sum(np.exp(np.multiply(beta, acionValues)))))
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


class BasePolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1, target2):
        actionDict = self.Q_dict[(playerGrid, tuple(sorted((target1, target2))))]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class GoalInfernce:
    def __init__(self, initPrior, goalPolicy):
        self.initPrior = initPrior
        self.goalPolicy = goalPolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        goalPosteriorList = []
        priorGoal = initPrior[0]

        goal = trajectory[-1]
        targets = list([target1, target2])
        noGoal = [target for target in targets if target != goal][0]
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(playerGrid, goal).get(action)
            likelihoodB = self.goalPolicy(playerGrid, noGoal).get(action)
            posteriorGoal = (priorGoal * likelihoodGoal) / ((priorGoal * likelihoodGoal) + (1 - priorGoal) * likelihoodB)
            goalPosteriorList.append(posteriorGoal)
            priorGoal = posteriorGoal
        goalPosteriorList.insert(0, initPrior[0])
        return goalPosteriorList


def calInformationGain(baseProb, conditionProb):

    return entropy(baseProb) - entropy(conditionProb)


def calPosterior(goalPosteriorList):
    x = np.divide(np.arange(len(goalPosteriorList) + 1), len(goalPosteriorList))
    goalPosteriorList.append(1)
    y = np.array(goalPosteriorList)
    f = interp1d(x, y, kind='nearest')
    xnew = np.linspace(0., 1., 30)
    goalPosterior = f(xnew)
    return goalPosterior


def calInfo(expectedInfoList):
    x = np.divide(np.arange(len(expectedInfoList)), len(expectedInfoList) - 1)
    y = np.array(expectedInfoList)
    f = interp1d(x, y, kind='nearest')
    xnew = np.linspace(0., 1., 30)
    goalPosterior = f(xnew)
    return goalPosterior


class CalFirstIntentionStep:
    def __init__(self, inferThreshold):
        self.inferThreshold = inferThreshold

    def __call__(self, goalPosteriorList):
        for index, goalPosteriori in enumerate(goalPosteriorList):
            if goalPosteriori > self.inferThreshold:
                return index + 1
                break
        return len(goalPosteriorList)


class CalFirstIntentionStepRatio:
    def __init__(self, calFirstIntentionStep):
        self.calFirstIntentionStep = calFirstIntentionStep

    def __call__(self, goalPosteriorList):
        firstIntentionStep = self.calFirstIntentionStep(goalPosteriorList)
        firstIntentionStepRatio = firstIntentionStep / len(goalPosteriorList)
        return firstIntentionStepRatio


def calMidPoints(trajectory, target1, target2):
    playerGrid = trajectory[0]
    zone = calculateAvoidCommitmnetZone(playerGrid, target1, target2)
    midpoints = list([(target1[0], target2[1]), (target2[0], target1[1])])
    midPoint = list(set(zone).intersection(set(midpoints)))
    return midPoint[0]


def isTrajHasMidPoints(trajectory, target1, target2):
    trajectory = list(map(tuple, trajectory))
    midPoint = calMidPoints(trajectory, target1, target2)
    hasMidPoint = 1 if midPoint in trajectory else 0

    return hasMidPoint


def sliceTraj(trajectory, midPoint):
    trajectory = list(map(tuple, trajectory))
    index = trajectory.index(midPoint) + 1
    return trajectory[: index]


if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))
    # Q_dict_base = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGird15_policy.pkl"), "rb"))
    softmaxBeta = 2.5
    softmaxPolicy = SoftmaxPolicy(Q_dict, softmaxBeta)
    # basePolicy = BasePolicy(Q_dict_base, softmaxBeta)
    initPrior = [0.5, 0.5]
    inferThreshold = 0.95
    goalInfernce = GoalInfernce(initPrior, softmaxPolicy)
    calFirstIntentionStep = CalFirstIntentionStep(inferThreshold)
    calFirstIntentionStepRatio = CalFirstIntentionStepRatio(calFirstIntentionStep)

    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'softmaxBeta2.5', 'commitBeta1', 'commitBeta3', 'commitBeta5', 'commitBeta7']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        df = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]

        # print(len(df))
        df['hasMidPoint'] = df.apply(lambda x: isTrajHasMidPoints(eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)

        # dfExpTrail = df[(df['areaType'] == 'rect')]

        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] == 'special')]

        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]

        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine') & (df['intentionedDisToTargetMin'] == 2)]

        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine') & (df['distanceDiff'] == 0)]
        # dfExpTrail = df[(df['areaType'] != 'none')]

        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['areaType'] != 'rect')]

        # dfExpTrail = df[df['noiseNumber'] != 'special']
        # dfExpTrail = df

        statDF = pd.DataFrame()
        # statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        # statDF['firstIntentionStepRatio'] = dfExpTrail.groupby('name')["firstIntentionStepRatio"].mean()
        # statDF['firstIntentionStepRatio'] = dfExpTrail.groupby('name')["firstIntentionStepRatio"].mean()
        # statDF['goalPosterior'] = dfExpTrail.groupby('name')["goalPosterior"].mean()

        statDF['midTriaPercent'] = df.groupby('name')["hasMidPoint"].sum() / (len(df) / len(df.groupby('name')["hasMidPoint"]))

        # print(df.groupby('name')["hasMidPoint"].head(6))

        # print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        stats = statDF.columns

        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    print(statsList)
    lables = participants
    # lables = ['Human', 'Agent']

    xlabels = ['midTriaPercent']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.1, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.ylim((0, 0.8))
    plt.legend(loc='best')

    plt.show()
