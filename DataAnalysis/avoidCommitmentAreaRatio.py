import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
from collections import Counter


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    playerGrid, target1, target2 = [eval(i) for i in [playerGrid, target1, target2]]
    dis1 = np.linalg.norm(np.array(playerGrid) - np.array(target1), ord=1)
    dis2 = np.linalg.norm(np.array(playerGrid) - np.array(target2), ord=1)
    if dis1 == dis2:
        rect1 = creatRect(playerGrid, target1)
        rect2 = creatRect(playerGrid, target2)
        avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
        avoidCommitmentZone.remove(tuple(playerGrid))
    else:
        avoidCommitmentZone = []
    return avoidCommitmentZone


def calculateFirstIntentionRatio(trajectory, zone):
    trajectory = eval(trajectory)
    avoidCommitmentPath = list()
    for point in trajectory:
        if tuple(point) not in zone and len(avoidCommitmentPath) != 0:
            break
        if tuple(point) in zone:
            avoidCommitmentPath.append(point)
    avoidCommitmentRatio = len(avoidCommitmentPath) / len(trajectory)
    return avoidCommitmentRatio


def calculateAvoidCommitmentRatio(trajectory, zone):
    trajectory = eval(trajectory)
    avoidCommitmentPath = list()
    avoidCommitmentSteps = 0
    for step in trajectory:
        if tuple(step) in zone:
            avoidCommitmentSteps += 1
    avoidCommitmentRatio = avoidCommitmentSteps / len(trajectory)
    return avoidCommitmentRatio


def calculateFirstIntentionStep(goalList):
    goal1Step = len(goalList)
    goal2Step = len(goalList)
    intentionList = eval(goalList)
    if 1 in intentionList:
        goal1Step = intentionList.index(1)
    if 2 in intentionList:
        goal2Step = intentionList.index(2)
    firstIntentionStep = min(goal1Step, goal2Step)
    return firstIntentionStep


if __name__ == '__main__':
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    avoidCommitmentRatio = []
    participants = ['human', 'max', 'maxNoNoise']
    for participant in participants:
        dataPath = os.path.join(resultsPath, participant)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
        # df.to_csv("all.csv")
        df['avoidCommitmentZone'] = df.apply(lambda x: calculateAvoidCommitmnetZone(x['playerGrid'], x['target1'], x['target2']), axis=1)

        df['avoidCommitmentRatio'] = df.apply(lambda x: calculateAvoidCommitmentRatio(x['trajectory'], x['avoidCommitmentZone']), axis=1)
        df['firstIntentionRatio'] = df.apply(lambda x: calculateFirstIntentionRatio(x['trajectory'], x['avoidCommitmentZone']), axis=1)

        df['firstIntentionStep'] = df.apply(lambda x: calculateFirstIntentionStep(x['goal']), axis=1)

        # print(df.head(6))
        nubOfSubj = len(df["name"].unique())
        statDF = pd.DataFrame()
        print(nubOfSubj)

        dfExpTrail = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
        # dfExpTrail = df[df['noiseNumber'] != 'special']
        # dfExpTrail = df[(df['areaType'] == 'straightLine') | (df['areaType'] == 'midLine')]
        # dfExpTrail = df[(df['areaType'] != 'none')]

        # df.to_csv("statDF.csv")
        statDF['avoidCommitmentRatio'] = dfExpTrail.groupby('name')["avoidCommitmentRatio"].mean()
        print('avoidCommitmentRatio', np.mean(statDF['avoidCommitmentRatio']))
        statDF['firstIntentionRatio'] = dfExpTrail.groupby('name')["firstIntentionRatio"].mean()
        print('firstIntentionRatio', np.mean(statDF['firstIntentionRatio']))
        statDF['firstIntentionStep'] = dfExpTrail.groupby('name')["firstIntentionStep"].mean()
        print('firstIntentionStep', np.mean(statDF['firstIntentionStep']))

        avoidCommitmentRatio.append(np.mean(statDF['firstIntentionRatio']))

    plt.bar([0, 0.3, 0.6], avoidCommitmentRatio, width=0.1)
    plt.xticks([0, 0.3, 0.6], participants)
    plt.xlabel('Consist And Inconsist')
    plt.ylabel('frequency')
    plt.ylim((0, 1))
    plt.title('firstIntentionPredictFinalGoal NormalTrial')
    plt.show()
