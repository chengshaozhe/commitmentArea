import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind

from dataAnalysis import calculateFirstIntentionConsistency, calculateFirstIntention


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statsList = []
    stdList = []
    participants = ['human', 'maxModelNoise0.1', 'softMaxBeta2.5ModelNoise0.1', 'softMaxBeta10Model', 'maxModelNoNoise']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")

        # print(df.head(6))
        nubOfSubj = len(df["name"].unique())
        statDF = pd.DataFrame()
        print('participant', participant, nubOfSubj)

        df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionConsistency(eval(x['goal'])), axis=1)
        dfNormailTrail = df[df['noiseNumber'] != 'special']
        dfSpecialTrail = df[df['noiseNumber'] == 'special']

        statDF['firstIntentionConsistFinalGoalNormal'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
        statDF['firstIntentionConsistFinalGoalSpecail'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()

        # statDF['firstIntentionConsistFinalGoalNormalSD'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].std()
        # statDF['firstIntentionConsistFinalGoalSpecailSD'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].std()

        # statDF.to_csv("statDF.csv")
        print('firstIntentionConsistFinalGoalNormal', np.mean(statDF['firstIntentionConsistFinalGoalNormal']))
        print('firstIntentionConsistFinalGoalSpecail', np.mean(statDF['firstIntentionConsistFinalGoalSpecail']))
        print('')

        statsList.append([np.mean(statDF['firstIntentionConsistFinalGoalNormal']), np.mean(statDF['firstIntentionConsistFinalGoalSpecail'])])

        stdList.append([np.std(statDF['firstIntentionConsistFinalGoalNormal']) / np.sqrt(len(statDF['firstIntentionConsistFinalGoalNormal']) - 1), np.std(statDF['firstIntentionConsistFinalGoalSpecail']) / np.sqrt(len(statDF['firstIntentionConsistFinalGoalSpecail']) - 1)])

    xlabels = ['normalTrial', 'specialTrial']
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
    plt.title('commit to goal ratio')  # Intention Consistency
    plt.show()
