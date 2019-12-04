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


def calFirstIntentionStepRationAfterNoise(noisePoints, goalList):
    afterNoiseGoalList = goalList[noisePoints:]
    afterNoiseFirstIntentionStep = calculateFirstIntentionStep(afterNoiseGoalList)
    return afterNoiseFirstIntentionStep


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    # participants = ['human', 'maxModelNoise0.1', 'softMaxBeta2.5ModelNoise0.1', 'softMaxBeta10Model', 'maxModelNoNoise']
    participants = ['human', 'softMaxBeta100']

    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        nubOfSubj = len(df["name"].unique())
        print('participant', participant, nubOfSubj)
        dfSpecialTrail = df[df['noiseNumber'] == 'special']

        # dfSpecialTrail["afterNoiseIntentionConsis"] = dfSpecialTrail.apply(lambda x: calFirstIntentionConsistAfterNoise(eval(x['noisePoint']), eval(x['goal'])), axis=1)

        dfSpecialTrail["afterNoiseIntentionConsis"] = dfSpecialTrail.apply(lambda x: calFirstIntentionConsistAfterNoise(eval(x['trajectory']), eval(x['noisePoint']), eval(x['target1']), eval(x['target2']), eval(x['goal'])), axis=1)

        dfSpecialTrail["afterNoiseFirstIntentionStep"] = dfSpecialTrail.apply(lambda x: calFirstIntentionStepRationAfterNoise(eval(x['noisePoint']), eval(x['goal'])), axis=1)

        statDF = pd.DataFrame()
        statDF['afterNoiseIntentionConsisSpecail'] = dfSpecialTrail.groupby('name')["afterNoiseIntentionConsis"].mean()
        # statDF['afterNoiseFirstIntentionStep'] = dfSpecialTrail.groupby('name')["afterNoiseFirstIntentionStep"].mean()

        # statDF.to_csv("statDF.csv")

        print('afterNoiseIntentionConsisSpecail', np.mean(statDF['afterNoiseIntentionConsisSpecail']))
        # print('afterNoiseFirstIntentionStep', np.mean(statDF['afterNoiseFirstIntentionStep']))

        print('')

        stats = statDF.columns
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])

    xlabels = ['afterNoiseIntentionConsisSpecail']
    lables = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 3
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=lables[i])
    plt.xticks(x, xlabels)

    plt.ylim((0, 1))
    plt.legend(loc='best')
    plt.title('afterNoise first step intention consistency')  # Intention Consistency
    plt.show()
