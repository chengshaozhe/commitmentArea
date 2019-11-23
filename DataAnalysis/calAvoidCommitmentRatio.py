import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter
from dataAnalysis import calculateSE,calculateAvoidCommitmnetZone,calculateAvoidCommitmnetZoneAll, calculateFirstOutZoneRatio, calculateAvoidCommitmentRatio, calculateFirstIntentionStep, calculateFirstIntentionRatio


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'maxModelNoise0.1', 'softMaxBeta2.5ModelNoise0.1', 'softMaxBeta10Model', 'maxModelNoNoise']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)

        df['avoidCommitmentZone'] = df.apply(lambda x: calculateAvoidCommitmnetZone(eval(x['playerGrid']), eval(x['target1']), eval(x['target2'])), axis=1)
        df['avoidCommitmentRatio'] = df.apply(lambda x: calculateAvoidCommitmentRatio(eval(x['trajectory']), x['avoidCommitmentZone']), axis=1)
        df['firstIntentionStep'] = df.apply(lambda x: calculateFirstIntentionStep(eval(x['goal'])), axis=1)
        df['firstIntentionRatio'] = df.apply(lambda x: calculateFirstIntentionRatio(eval(x['goal'])), axis=1)
        # df.to_csv("all.csv")

        # df['firstIntentionRatio'] = df.apply(lambda x: calculateRatioInNonCommitment(x['goal']), axis=1)

        # print(df.head(6))
        nubOfSubj = len(df["name"].unique())
        statDF = pd.DataFrame()
        print(participant, nubOfSubj)

        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]

        dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'rect')]

        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine') & (df['distanceDiff'] == 0)]
        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['areaType'] != 'none')]
        # dfExpTrail = df[df['noiseNumber'] != 'special']
        # dfExpTrail = df

        statDF['avoidCommitmentRatio'] = dfExpTrail.groupby('name')["avoidCommitmentRatio"].mean()
        print('avoidCommitmentRatio', np.mean(statDF['avoidCommitmentRatio']))

        statDF['firstIntentionRatio'] = dfExpTrail.groupby('name')["firstIntentionRatio"].mean()
        print('firstIntentionRatio', np.mean(statDF['firstIntentionRatio']))

        statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        # statDF.to_csv("statDF.csv")

        # stats = statDF.columns
        stats = ['firstIntentionRatio','avoidCommitmentRatio']
        statsList.append([np.mean(statDF[stat]) for stat in stats])
        stdList.append([calculateSE(statDF[stat]) for stat in stats])


    xlabels = ['firstIntentionRatio', 'avoidCommitmentAreaRatio']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], yerr=stdList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.ylim((0, 1))
    plt.legend(loc='best')

    plt.title('avoidCommitment')
    plt.show()
