import numpy as np


def calculateSE(data):
    standardError = np.std(data, ddof=1) / np.sqrt(len(data) - 1)
    return standardError


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


def creatRect(coor1, coor2):
    vector = np.array(list(zip(coor1, coor2)))
    vector.sort(axis=1)
    rect = [(i, j) for i in range(vector[0][0], vector[0][1] + 1) for j in range(vector[1][0], vector[1][1] + 1)]
    return rect


def calculateAvoidCommitmnetZoneAll(playerGrid, target1, target2):
    rect1 = creatRect(playerGrid, target1)
    rect2 = creatRect(playerGrid, target2)
    avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
    avoidCommitmentZone.remove(tuple(playerGrid))
    return avoidCommitmentZone


def calculateAvoidCommitmnetZone(playerGrid, target1, target2):
    dis1 = calculateGridDis(playerGrid, target1)
    dis2 = calculateGridDis(playerGrid, target2)
    if dis1 == dis2:
        rect1 = creatRect(playerGrid, target1)
        rect2 = creatRect(playerGrid, target2)
        avoidCommitmentZone = list(set(rect1).intersection(set(rect2)))
        avoidCommitmentZone.remove(tuple(playerGrid))
    else:
        avoidCommitmentZone = []
    return avoidCommitmentZone


def calculateAvoidCommitmentRatio(trajectory, zone):
    avoidCommitmentSteps = 0
    for step in trajectory:
        if tuple(step) in zone:
            avoidCommitmentSteps += 1
    avoidCommitmentRatio = avoidCommitmentSteps / (len(trajectory) - 1)
    return avoidCommitmentRatio


def calculateFirstOutZoneRatio(trajectory, zone):
    avoidCommitmentPath = list()
    for point in trajectory:
        if tuple(point) not in zone and len(avoidCommitmentPath) != 0:
            break
        if tuple(point) in zone:
            avoidCommitmentPath.append(point)
    avoidCommitmentRatio = len(avoidCommitmentPath) / (len(trajectory) - 1)
    return avoidCommitmentRatio


def calculateFirstIntentionStep(goalList):
    maxSteps = len(goalList)
    goal1Step = goalList.index(1) + 1 if 1 in goalList else maxSteps
    goal2Step = goalList.index(2) + 1 if 2 in goalList else maxSteps
    firstIntentionStep = min(goal1Step, goal2Step)
    return firstIntentionStep


def calculateFirstIntentionRatio(goalList):
    firstIntentionStep = calculateFirstIntentionStep(goalList)
    firstIntentionRatio = firstIntentionStep / len(goalList)
    return firstIntentionRatio


def calculateFirstIntention(goalList):
    maxSteps = len(goalList) - 1
    goal1Index = goalList.index(1) if 1 in goalList else maxSteps
    goal2Index = goalList.index(2) if 2 in goalList else maxSteps
    firstIntentionIndex = min(goal1Index, goal2Index)
    firstIntention = goalList[firstIntentionIndex]
    return firstIntention


def calculateFirstIntentionConsistency(goalList):
    firstGoal = calculateFirstIntention(goalList)
    finalGoal = calculateFirstIntention(list(reversed(goalList)))
    firstIntention = 1 if firstGoal == finalGoal else 0
    return firstIntention


if __name__ == '__main__':
    playerGrid, target1, target2 = [(4, 4), (13, 8), (9, 12)]
    a = calculateAvoidCommitmnetZone(playerGrid, target1, target2)
    print(a)

    trajectory = [(4, 4), [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [5, 9], [6, 9], [7, 9], [7, 10], [7, 11], [8, 11], [8, 12], [9, 12]]
    b = calculateAvoidCommitmentRatio(trajectory, a)
    print(b)

    c = calculateFirstOutZoneRatio(trajectory, a)
    print(c)

    goalList = [0, 0, 0, 2, 2, 2, 2, 2, 0, 0]
    d = calculateFirstIntentionRatio(goalList)
    print(d)

    print(calculateFirstIntention(goalList))
