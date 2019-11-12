import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind


def calculateFirstIntentionMatchFinalIntention(goalStr):
    goalList = eval(goalStr)
    firstGoal = calculateFirstIntention(goalList)
    finalGoal = calculateFirstIntention(list(reversed(goalList)))
    firstIntention = 1 if firstGoal == finalGoal else 0
    return firstIntention


def calculateFirstIntention(goalList):
    try:
        target1Goal = goalList.index(1)
    except ValueError as e:
        target1Goal = 999
    try:
        target2Goal = goalList.index(2)
    except ValueError as e:
        target2Goal = 999
    if target1Goal < target2Goal:
        firstGoal = 1
    elif target2Goal < target1Goal:
        firstGoal = 2
    else:
        firstGoal = 0
    return firstGoal


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    statVariable = []
    participants = ['human', 'maxModel', 'maxModelNoNoise']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")

        # print(df.head(6))
        nubOfSubj = len(df["name"].unique())
        statDF = pd.DataFrame()
        print('participant', participant, nubOfSubj)

        df["firstIntentionConsistFinalGoal"] = df.apply(lambda x: calculateFirstIntentionMatchFinalIntention(x['goal']), axis=1)
        dfNormailTrail = df[df['noiseNumber'] != 'special']
        dfSpecialTrail = df[df['noiseNumber'] == 'special']

        statDF['firstIntentionConsistFinalGoalNormal'] = dfNormailTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()
        statDF['firstIntentionConsistFinalGoalSpecail'] = dfSpecialTrail.groupby('name')["firstIntentionConsistFinalGoal"].mean()

       # statDF.to_csv("statDF.csv")
        print('firstIntentionConsistFinalGoalNormal', np.mean(statDF['firstIntentionConsistFinalGoalNormal']))
        print('firstIntentionConsistFinalGoalSpecail', np.mean(statDF['firstIntentionConsistFinalGoalSpecail']))
        print('')
        # statVariable.append(np.mean(statDF['firstIntentionConsistFinalGoalNormal']))
        statVariable.append(np.mean(statDF['firstIntentionConsistFinalGoalSpecail']))

    plt.bar([0, 0.3, 0.6], statVariable, width=0.1)
    plt.xticks([0, 0.3, 0.6], participants)
    plt.ylabel('frequency')
    plt.ylim((0, 1))
    plt.title('firstIntentionPredictFinalGoal')
    plt.show()