import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind

from dataAnalysis import calculateFirstIntention, calculateSE, calculateFirstIntentionRatio, calculateFirstIntentionStep


def isGridsALine(playerGrid, bean1Grid, bean2Grid):
    line = np.array((playerGrid, bean1Grid, bean2Grid)).T
    xcoors = line[0]
    ycoors = line[1]
    if len(set(xcoors)) != len(xcoors) or len(set(ycoors)) != len(ycoors):
        return True
    else:
        return False


def calFirstIntentionConsistAfterNoise(trajectory, noisePoints, target1, target2, goalList):
    trajectory = list(map(tuple, trajectory))

    afterNoiseGrid = trajectory[noisePoints]
    if isGridsALine(afterNoiseGrid, target1, target2):
        afterNoiseIntentionConsis = 1 if goalList[noisePoints] == calculateFirstIntention(goalList) else 0
    else:
        afterNoiseIntentionConsis = 1 if goalList[noisePoints + 1] == calculateFirstIntention(goalList) else 0
    return afterNoiseIntentionConsis


def calFirstIntentionInConsistAfterNoise(noisePoints, goalList):
    afterNoiseGoalList = goalList[noisePoints:]
    afterNoiseIntentionInConsis = 1 if calculateFirstIntention(afterNoiseGoalList) != calculateFirstIntention(goalList) else 0
    return afterNoiseIntentionInConsis


def calFirstIntentionDelayConsistAfterNoise(trajectory, noisePoints, target1, target2, goalList):
    afterNoiseIntentionDelayConsis = 1 if not calFirstIntentionConsistAfterNoise(trajectory, noisePoints, target1, target2, goalList) and not calFirstIntentionInConsistAfterNoise(noisePoints, goalList) else 0
    return afterNoiseIntentionDelayConsis


def calFirstIntentionStepRationAfterNoise(noisePoints, goalList):
    afterNoiseGoalList = goalList[noisePoints:]
    afterNoiseFirstIntentionStep = calculateFirstIntentionStep(afterNoiseGoalList)
    return afterNoiseFirstIntentionStep


class CalLikehood:
    def __init__(self, goalPolicy):
        self.goalPolicy = goalPolicy

    def __call__(self, trajectory, target1, target2, noisePoints, goalList):

        playerGrid = trajectory[noisePoints]
        targets = list([target1, target2])
        originGoal = targets[calculateFirstIntention(goalList)]
        noGoal = [target for target in targets if target != originGoal][0]

        likelihoodGoal = self.goalPolicy(playerGrid, originGoal).get(action)
        likelihoodNoGoal = self.goalPolicy(playerGrid, noGoal).get(action)

        return [likelihoodGoal, likelihoodNoGoal]


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    # participants = ['human', 'maxModelNoise0.1', 'softMaxBeta2.5ModelNoise0.1', 'softMaxBeta10Model', 'maxModelNoNoise']

    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))
    softmaxBeta = 0.5
    softmaxPolicy = SoftmaxPolicy(Q_dict, softmaxBeta)
    calLikehood = CalLikehood(softmaxPolicy)

    participants = ['human', 'softmaxBeta0.5']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)

        df = df[df['noisePoint'] != "[]"]
        dfSpecialTrail = df[df['noiseNumber'] == 'special']

        # dfSpecialTrail["afterNoiseIntentionConsis"] = dfSpecialTrail.apply(lambda x: calFirstIntentionConsistAfterNoise(eval(x['noisePoint']), eval(x['goal'])), axis=1)

        dfSpecialTrail["afterNoiseIntentionConsis"] = dfSpecialTrail.apply(lambda x: calFirstIntentionConsistAfterNoise(eval(x['trajectory']), eval(x['noisePoint']), eval(x['target1']), eval(x['target2']), eval(x['goal'])), axis=1)

        dfSpecialTrail["afterNoiseIntentionInConsis"] = dfSpecialTrail.apply(lambda x: calFirstIntentionInConsistAfterNoise(eval(x['noisePoint']), eval(x['goal'])), axis=1)

        dfSpecialTrail["afterNoiseIntentionConsisDelay"] = dfSpecialTrail.apply(lambda x: calFirstIntentionDelayConsistAfterNoise(eval(x['trajectory']), eval(x['noisePoint']), eval(x['target1']), eval(x['target2']), eval(x['goal'])), axis=1)

        # dfSpecialTrail["afterNoiseFirstIntentionStep"] = dfSpecialTrail.apply(lambda x: calFirstIntentionStepRationAfterNoise(eval(x['noisePoint']), eval(x['goal'])), axis=1)

        statDF = pd.DataFrame()
        statDF['afterNoiseIntentionConsis'] = dfSpecialTrail.groupby('name')["afterNoiseIntentionConsis"].mean()
        statDF['afterNoiseIntentionConsisDelay'] = dfSpecialTrail.groupby('name')["afterNoiseIntentionConsisDelay"].mean()
        # statDF['afterNoiseIntentionInConsis'] = dfSpecialTrail.groupby('name')["afterNoiseIntentionInConsis"].mean()

        # statDF['afterNoiseFirstIntentionStep'] = dfSpecialTrail.groupby('name')["afterNoiseFirstIntentionStep"].mean()

        # statDF.to_csv("statDF.csv")

        print('afterNoiseIntentionConsis', np.mean(statDF['afterNoiseIntentionConsis']))
        # print('afterNoiseFirstIntentionStep', np.mean(statDF['afterNoiseFirstIntentionStep']))

        print('')

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])
        print(statsList)

    xlabels = ['Commit to Intention with Least Actions', 'Commit to Intention with Delay Actions']
    # lables = participants
    lables = ['Human', 'RL Agent']

    # x = np.arange(len(xlabels))
    totalWidth, n = 0.8, len(xlabels)
    width = totalWidth / n
    # x = x - (totalWidth - width) / 3

    ind = np.arange(len(lables))

    consisInLeastSteps = [statsList[0][0], statsList[1][0]]
    consisWithDelaySteps = [statsList[0][1], statsList[1][1]]

    p1 = plt.bar(ind, consisInLeastSteps, width, yerr=[stdList[0][0], stdList[1][0]])
    p2 = plt.bar(ind, consisWithDelaySteps, width, bottom=consisInLeastSteps, yerr=[stdList[0][1], stdList[1][1]])

    plt.xticks(ind, lables)
    plt.legend((p1[0], p2[0]), xlabels)

    fontSize = 16
    plt.xticks(fontsize=fontSize, color='black')
    plt.yticks(fontsize=fontSize, color='black')

    plt.ylim((0, 1))
    # plt.legend(loc='best')
    plt.title('Intention Consistency', fontsize=fontSize, color='black')  # Intention Consistency
    plt.show()
