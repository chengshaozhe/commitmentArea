import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter

from dataAnalysis import calculateFirstIntentionStep


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    participants = ['human', 'maxModelNoise0.1', 'maxModelNoNoise']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)

        df['firstIntentionStep'] = df.apply(lambda x: calculateFirstIntentionStep(eval(x['goal'])), axis=1)
        # df.to_csv("all.csv")

        # print(df.head(6))
        nubOfSubj = len(df["name"].unique())
        statDF = pd.DataFrame()
        print(participant, nubOfSubj)

        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
        dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] != 'none')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'straightLine')]
        # dfExpTrail = df[(df['distanceDiff'] == 0) & (df['areaType'] == 'midLine')]

        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine') & (df['distanceDiff'] == 0)]
        # dfExpTrail = df[(df['areaType'] != 'none')]
        # dfExpTrail = df[(df['areaType'] == 'expRect') & (df['areaType'] != 'rect')]

        # dfExpTrail = df[df['noiseNumber'] != 'special']
        # dfExpTrail = df

        statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))
        print('')

        # statDF.to_csv("statDF.csv")
        statsList.append([np.mean(statDF['firstIntentionStep'])])

    xlabels = ['firstIntentionStep']
    labels = participants
    x = np.arange(len(xlabels))
    totalWidth, n = 0.1, len(participants)
    width = totalWidth / n
    x = x - (totalWidth - width) / 2
    for i in range(len(statsList)):
        plt.bar(x + width * i, statsList[i], width=width, label=labels[i])
    plt.xticks(x, xlabels)
    plt.ylim((0, 10))
    plt.legend(loc='best')

    plt.title('firstIntentionStep')
    plt.show()
