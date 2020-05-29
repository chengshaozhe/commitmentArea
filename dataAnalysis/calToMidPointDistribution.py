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
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calculateGridDis, calMidPoints


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


def sliceTraj(trajectory, midPoint):
    trajectory = list(map(tuple, trajectory))
    index = trajectory.index(midPoint) + 1
    return trajectory[: index]


def calDisToMidPointTraj(trajectory, target1, target2):
    trajectory = list(map(tuple, trajectory))
    playerGrid = trajectory[0]
    midPoint = calMidPoints(playerGrid, target1, target2)
    disToMidPointTraj = [calculateGridDis(grid, midPoint) for grid in trajectory]

    x = np.divide(np.arange(len(disToMidPointTraj)), len(disToMidPointTraj) - 1)
    y = np.array(disToMidPointTraj)

    f = interp1d(x, y, kind='nearest')
    xnew = np.linspace(0., 1., 30)
    posterior = f(xnew)
    return posterior


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
    statDFList = []
    participants = ['human', 'softmaxBeta2.5']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        nubOfSubj = len(df["name"].unique())
        print(participant, nubOfSubj)

        df['goalPosteriorList'] = df.apply(lambda x: goalInfernce(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

        # df['expectedInfoList'] = df.apply(lambda x: calculateActionInformation(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

        df['firstIntentionStep'] = df.apply(lambda x: calFirstIntentionStep(x['goalPosteriorList']), axis=1)

        df['firstIntentionStepRatio'] = df.apply(lambda x: calFirstIntentionStepRatio(x['goalPosteriorList']), axis=1)

        df['goalPosterior'] = df.apply(lambda x: calPosterior(x['goalPosteriorList']), axis=1)

        df = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]

        df['posterior'] = df.apply(lambda x: calDisToMidPointTraj(eval(x['trajectory']), eval(x['target1']), eval(x['target2'])), axis=1)

        # df['goalPosterior'] = df.apply(lambda x: calInfo(x['expectedInfoList']), axis=1)

        dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
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

        # print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        stats = statDF.columns
        # statsList.append([np.mean(statDF[stat]) for stat in stats])
        # stdList.append([calculateSE(statDF[stat]) for stat in stats])

        posterior = np.array(dfExpTrail['posterior'].tolist())
        posteriorMean = np.mean(posterior, axis=0)
        posteriorStd = np.divide(np.std(posterior, axis=0, ddof=1), np.sqrt(len(posterior) - 1))
        statsList.append(posteriorMean)
        stdList.append(posteriorStd)

        def arrMean(df, colnames):
            arr = np.array(df[colnames].tolist())
            return np.mean(arr, axis=0)
        grouped = pd.DataFrame(dfExpTrail.groupby('name').apply(arrMean, 'posterior'))
        statArr = np.array(grouped.iloc[:, 0].tolist()).T

        statDFList.append(statArr)

    # testDataSet = abs(np.array(statDFList[0]) - np.array(statDFList[1]))
    # testData = testDataSet
    # print(testData)
    # print(ttest_ind(testData, np.zeros(len(testData))))

    pvalus = np.array([ttest_ind(statDFList[0][i], statDFList[1][i])[1] for i in range(statDFList[0].shape[0])])
    # pvalus = np.array([mannwhitneyu(statDFList[0][i], statDFList[1][i])[1] for i in range(statDFList[0].shape[0])])

    sigArea = np.where(pvalus < 0.05)[0]
    print(sigArea)

    # print(mannwhitneyu(statDFList[0], statDFList[1]))
    # print(ranksums(statDFList[0], statDFList[1]))

    lables = participants
    # lables = ['Human', 'Agent']

    xnew = np.linspace(0., 1., 30)
    xnewSig = xnew[sigArea]
    ySig = [stats[sigArea] for stats in statsList]

    lineWidth = 1
    for i in range(len(statsList)):
        plt.plot(xnew, statsList[i], label=lables[i], linewidth=lineWidth)
    # plt.errorbar(xnew, statsList[i], yerr=stdList[i], label=lables[i])

 # sig area line
    # for sigLine in [xnewSig[0], xnewSig[-1]]:
    #     plt.plot([sigLine] * 10, np.linspace(0.5, 1., 10), color='black', linewidth=2, linestyle="--")

    # xlabels = ['firstIntentionStepRatio']
    # labels = participants
    # x = np.arange(len(xlabels))
    # totalWidth, n = 0.1, len(participants)
    # width = totalWidth / n
    # x = x - (totalWidth - width) / 2
    # for i in range(len(statsList)):
    #     plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    # plt.xticks(x, xlabels)
    # plt.ylim((0, 1.1))

    fontSize = 10
    plt.legend(loc='best', fontsize=fontSize)
    plt.xlabel('Time', fontsize=fontSize, color='black')
    plt.ylabel('Probability of Intention', fontsize=fontSize, color='black')

    plt.xticks(fontsize=fontSize, color='black')
    plt.yticks(fontsize=fontSize, color='black')

    plt.title('Inferred Intention Through Time', fontsize=fontSize, color='black')
    plt.show()