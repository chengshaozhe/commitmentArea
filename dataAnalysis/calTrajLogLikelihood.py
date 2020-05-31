import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import random
import pickle
from scipy.stats import entropy
from scipy.interpolate import interp1d
from dataAnalysis import calculateSE, calculateAvoidCommitmnetZoneAll, calculateAvoidCommitmnetZone, calculateGridDis, calMidPoints, calculateSoftmaxProbability


def calculateSoftmaxProbability(acionValues, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, acionValues)), np.sum(np.exp(np.multiply(beta, acionValues)))))
    return newProbabilityList


def KL(p, q):
    KLD = entropy(p, q)
    return KLD


class GoalPolicy:
    def __init__(self, goal_dict, softmaxBeta):
        self.goal_dict = goal_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1):
        actionDict = self.goal_dict[(playerGrid, target1)]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class SoftmaxRLPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1, target2):
        actionDict = self.Q_dict[(playerGrid, tuple(sorted((target1, target2))))]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class BasePolicy:
    def __init__(self, goalPolicy):
        self.goalPolicy = goalPolicy

    def __call__(self, playerGrid, priorList):
        actionProbList = [self.goalPolicy(playerGrid, goal).values() for goal in [target1, target2]]
        basePolicyList = [np.multiply(prior, actionProb) for prior, actionProb in zip(priorList, actionProbList)]
        basePolicy = np.sum(basePolicyList, axis=0)
        return basePolicy


class CalLikehood:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, state, action):
        if len(state) == 3:
            playerGrid, target1, target2 = state
            likelihood = self.policy(playerGrid, target1, target2).get(action)
        elif len(state) == 2:
            playerGrid, target = state
            likelihood = self.policy(playerGrid, target).get(action)
        else:
            raise ValueError('wrong input state')
        return likelihood


class CalTrajLikelihood():
    def __init__(self, calLikehood):
        self.calLikehood = calLikehood

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        likelihoodList = []
        for playerGrid, action in zip(trajectory, aimAction):
            state = [playerGrid, target1, target2]
            likelihood = self.calLikehood(state, action)
            likelihoodList.append(likelihood)
        likelihoodAll = np.prod(likelihoodList)
        return likelihoodAll


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


def calBasePolicy(priorList, actionProbList):
    basePolicyList = [np.multiply(goalProb, actionProb) for goalProb, actionProb in zip(priorList, actionProbList)]
    basePolicy = np.sum(basePolicyList, axis=0)
    return basePolicy


def isCommited(selfGrid, target1, target2):
    distanceDiff = abs(calculateGridDis(selfGrid, target1) - calculateGridDis(selfGrid, target2))
    pAvoidCommit = 1 / (distanceDiff + 1)
    return 1 - pAvoidCommit


class AvoidCommitWithMidpiontPolicy:
    def __init__(self, goalPolicy):
        self.goalPolicy = goalPolicy

    def __call__(self, selfGrid, target1, target2, trajectory):
        initGrid = trajectory[0]
        goal = trajectory[-1]
        midpoint = calMidPoints(selfGrid, target1, target2)
        zone = calculateAvoidCommitmnetZoneAll(initGrid, target1, target2)
        if midpoint:
            disToMidPoint = calculateGridDis(selfGrid, midpoint)
            disToTargets = [calculateGridDis(selfGrid, target) for target in[target1, target2]]
            isInDeliberationArea = 1 if disToMidPoint < min(disToTargets) else 0

            if isInDeliberationArea:
                actionDis = self.goalPolicy(selfGrid, midpoint)
            else:
                actionDis = self.goalPolicy(selfGrid, goal)
        else:
            actionDis = self.goalPolicy(selfGrid, goal)
        return actionDis


class AvoidCommitPolicy:
    def __init__(self, goalPolicy):
        self.goalPolicy = goalPolicy
        self.priorList = [0.5, 0.5]

    def __call__(self, selfGrid, target1, target2, trajectory):
        trajectory = list(map(tuple, trajectory))
        initGrid = trajectory[0]
        goal = trajectory[-1]
        actionProbList = [list(self.goalPolicy(selfGrid, goal).values()) for goal in [target1, target2]]
        actionDis = calBasePolicy(self.priorList, actionProbList)
        actionKeys = self.goalPolicy(selfGrid, goal).keys()
        actionDict = dict(zip(actionKeys, actionDis))
        return actionDict


class DeliberateIntentionModel:
    def __init__(self, goalPolicy, avoidCommitPolicy):
        self.goalPolicy = goalPolicy
        self.avoidCommitPolicy = avoidCommitPolicy

    def __call__(self, selfGrid, target1, target2, trajectory):
        trajectory = list(map(tuple, trajectory))
        initGrid = trajectory[0]
        goal = trajectory[-1]
        pCommit = isCommited(selfGrid, target1, target2)
        priorList = [pCommit, 1 - pCommit]
        actionProbList = [list(self.goalPolicy(selfGrid, goal).values()), list(self.avoidCommitPolicy(selfGrid, target1, target2, trajectory).values())]
        actionDis = calBasePolicy(priorList, actionProbList)
        actionKeys = self.goalPolicy(selfGrid, goal).keys()
        actionDict = dict(zip(actionKeys, actionDis))
        return actionDict


class DeliberateIntentionCommitModel:
    def __init__(self, goalPolicy, avoidCommitPolicy):
        self.goalPolicy = goalPolicy
        self.avoidCommitPolicy = avoidCommitPolicy

    def __call__(self, selfGrid, target1, target2, trajectory):
        trajectory = list(map(tuple, trajectory))
        initGrid = trajectory[0]
        goal = trajectory[-1]
        pCommit = isCommited(selfGrid, target1, target2)
        priorList = [pCommit, 1 - pCommit]
        actionProbList = [list(self.goalPolicy(selfGrid, goal).values()), list(self.avoidCommitPolicy(selfGrid, target1, target2, trajectory).values())]
        actionDis = calBasePolicy(priorList, actionProbList)
        actionKeys = self.goalPolicy(selfGrid, goal).keys()
        actionDict = dict(zip(actionKeys, actionDis))
        return actionDict


class InferGoalPosterior:
    def __init__(self, goalPolicy, commitBeta):
        self.goalPolicy = goalPolicy
        self.commitBeta = commitBeta

    def __call__(self, playerGrid, action, target1, target2, priorList):
        targets = list([target1, target2])

        priorList = goalCommit(priorList, self.commitBeta)
        likelihoodList = [self.goalPolicy(playerGrid, goal).get(action) for goal in targets]
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in zip(priorList, likelihoodList)]

        evidence = sum(posteriorUnnormalized)
        posteriorList = [posterior / evidence for posterior in posteriorUnnormalized]

        return posteriorList


def calCommitStep(initGrid, target1, target2, trajectory):
    initGrid = trajectory[0]
    distanceDiff = abs(calculateGridDis(initGrid, target1) - calculateGridDis(initGrid, target2))
    minDisToTarget = min([calculateGridDis(selfGrid, target) for target in[target1, target2]])
    currentSteps = trajectory.index(selfGrid)

    return commitStep


class StickWithDeliberatedIntentionModel:
    def __init__(self, goalPolicy, avoidCommitPolicy):
        self.goalPolicy = goalPolicy
        self.avoidCommitPolicy = avoidCommitPolicy
        self.commitFlag = None

    def __call__(self, selfGrid, target1, target2, trajectory):
        initGrid = trajectory[0]
        goal = trajectory[-1]

        if selfGrid == initGrid:
            self.commitFlag = 0

        pCommit = isCommited(selfGrid, target1, target2)
        priorList = [1 - pCommit, pCommit]

        if self.commitFlag:
            commited = 1
        else:
            commited = np.random.choice(2, p=priorList)

        if commited:
            actionDict = self.goalPolicy(selfGrid, goal)
            self.commitFlag = True
        else:
            actionDict = self.avoidCommitPolicy(selfGrid, target1, target2, trajectory)
        return actionDict


class CommitWithDeliberatedIntentionModel:
    def __init__(self, goalPolicy, avoidCommitPolicy):
        self.goalPolicy = goalPolicy
        self.avoidCommitPolicy = avoidCommitPolicy

    def __call__(self, selfGrid, target1, target2, trajectory):
        trajectory = list(map(tuple, trajectory))
        initGrid = trajectory[0]
        goal = trajectory[-1]
        pCommit = isCommited(selfGrid, target1, target2)
        priorList = [1 - pCommit, pCommit]
        commited = np.random.choice(2, p=priorList)
        if commited:
            actionDict = self.goalPolicy(selfGrid, goal)
        else:
            actionDict = self.avoidCommitPolicy(selfGrid, target1, target2, trajectory)
        return actionDict


class CalRLLikelihood():
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        likelihoodList = []
        for playerGrid, action in zip(trajectory, aimAction):
            likelihood = self.policy(playerGrid, target1, target2).get(action)
            likelihoodList.append(likelihood)
        # likelihoodList = list(filter(lambda x: x > 0.2, likelihoodList))
        likelihoodAll = np.prod(likelihoodList)
        return likelihoodAll


class CalImmediateIntentionLh:
    def __init__(self, goalPolicy):
        self.goalPolicy = goalPolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        likelihoodList = []
        goal = trajectory[-1]
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.goalPolicy(playerGrid, goal).get(action)
            likelihoodList.append(likelihoodGoal)

        # likelihoodList = list(filter(lambda x: x > 0.2, likelihoodList))
        logLikelihood = np.prod(likelihoodList)
        return logLikelihood


class CalDeliberateIntentionLh:
    def __init__(self, deliberatePolicy):
        self.deliberatePolicy = deliberatePolicy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        likelihoodList = []
        goal = trajectory[-1]
        for playerGrid, action in zip(trajectory, aimAction):
            likelihoodGoal = self.deliberatePolicy(playerGrid, target1, target2, trajectory).get(action)
            likelihoodList.append(likelihoodGoal)

        # likelihoodList = list(filter(lambda x: x > 0.2, likelihoodList))
        logLikelihood = np.prod(likelihoodList)
        return logLikelihood


class CalExpectedBayesFactor:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def __call__(self, trajectory, aimAction):
        trajectory = list(map(tuple, trajectory))
        bayesFactorList = []
        for playerGrid, action in zip(trajectory, aimAction):
            likelihood1 = self.model1(playerGrid, goal).get(action)
            likelihood2 = self.model2(playerGrid, goal).get(action)
            bayesFactor = likelihood1 * log(likelihood1 / likelihood2)
            bayesFactorList.append(bayesFactor)
        expectedBayesFactor = np.prod(bayesFactorList)
        return expectedBayesFactor


def calBICFull(logLikelihood, numOfObservations, numOfParas):
    bic = -2 * logLikelihood + np.log(numOfObservations) * numOfParas
    return bic


def calBIC(logLikelihood):
    bic = -2 * logLikelihood
    return bic


if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    softmaxBeta = 2.5
    goal_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))
    goalPolicy = GoalPolicy(goal_dict, softmaxBeta)

    # Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGird15_policy.pkl"), "rb"))
    # RLPolicy = SoftmaxRLPolicy(Q_dict, softmaxBeta)

    # playerGrid, target1, target2 = [(3, 3), (6, 4), (4, 6)]
    # p = list(goalPolicy(playerGrid, target1).values())
    # q = list(goalPolicy(playerGrid, target2).values())
    # actionProbList = [list(goalPolicy(playerGrid, target1)), list(goalPolicy(playerGrid, target2))]

    # print(KL(p, q))

    # trajectory = [(1, 7), [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [8, 9], [9, 9], [10, 9], [8, 9], [9, 9], [10, 9], [11, 9], [12, 9], [12, 8], [12, 7]]
    # aimAction = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, -1), (0, -1)]
    # target1, target2 = (6, 13), (12, 7)

    trajectory = [(9, 6), [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [8, 11], [7, 11], [6, 11], [5, 11], [6, 10], [6, 11], [6, 12], [6, 13]]

    aimAction = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (0, 1), (0, 1), (0, 1)]
    target1, target2 = (6, 13), (4, 11)
    # calLogLikelihood = CalRLLikelihood(RLPolicy)

    avoidCommitWithMidpiontPolicy = AvoidCommitWithMidpiontPolicy(goalPolicy)
    avoidCommitPolicy = AvoidCommitPolicy(goalPolicy)

    deliberateIntentionModel = DeliberateIntentionModel(goalPolicy, avoidCommitPolicy)
    stickWithDeliberatedIntentionModel = StickWithDeliberatedIntentionModel(goalPolicy, avoidCommitPolicy)
    commitWihtDeliberatedIntentionModel = CommitWithDeliberatedIntentionModel(goalPolicy, avoidCommitPolicy)

    calImmediateIntentionLh = CalImmediateIntentionLh(goalPolicy)

    calTrajLikelihood = CalDeliberateIntentionLh(deliberateIntentionModel)

    calStickToDeliberatedModel = CalDeliberateIntentionLh(stickWithDeliberatedIntentionModel)

    calCommitWithDeliberation = CalDeliberateIntentionLh(commitWihtDeliberatedIntentionModel)

    calTrajLikelihoodList = [calImmediateIntentionLh, calStickToDeliberatedModel, calCommitWithDeliberation]

    likehoodLists = [calLikelihood(trajectory, aimAction, target1, target2) for calLikelihood in calTrajLikelihoodList]

    bicResults = [calBIC(np.log(likehoodList)) for likehoodList in likehoodLists]
    print(bicResults)


##
    resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    participant = 'human'
    dataPath = os.path.join(resultsPath, participant)
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
    print(df.columns)

    # df = df[(df['areaType'] == 'expRect') & (df['noiseNumber'] != 'special')]
    # df = df[(df['areaType'] == 'expRect') | (df['areaType'] == 'rect')]

    # df['likelihood'] = df.apply(lambda x: calLogLikelihood(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

    df['likelihood2'] = df.apply(lambda x: calImmediateIntentionLh(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

    df['likelihood3'] = df.apply(lambda x: calTrajLikelihood(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

    df['likelihood4'] = df.apply(lambda x: calStickToDeliberatedModel(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

    df['likelihood5'] = df.apply(lambda x: calCommitWithDeliberation(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

    grouped = pd.DataFrame(df.groupby('name'))
    statDF = pd.DataFrame()
    statsList = []
    stdList = []

    import random
    random.seed(147)
    numOfSamples = 50

    def calBICDF(df, colnames):
        samples = random.sample(range(len(df[colnames])), numOfSamples)
        likelihoodList = np.array(df[colnames])[samples]
        likelihoodAll = np.prod(likelihoodList)
        bic = calBIC(np.log(likelihoodAll))
        return bic

    statDF['modelBic2'] = df.groupby('name').apply(calBICDF, 'likelihood2')
    statDF['modelBic3'] = df.groupby('name').apply(calBICDF, 'likelihood3')
    statDF['modelBic4'] = df.groupby('name').apply(calBICDF, 'likelihood4')
    statDF['modelBic5'] = df.groupby('name').apply(calBICDF, 'likelihood5')

    stats = statDF.columns
    # statsList.append([statDF[stat].tolist() for stat in stats])

    statsList.append([np.mean(statDF[stat]) for stat in stats])
    stdList.append([calculateSE(statDF[stat]) for stat in stats])

    print(statsList)
    xlabels = ['model1', 'model2', 'model3', 'model4']

    x = np.arange(len(xlabels))
    totalWidth, n = 0.6, len(xlabels)
    width = totalWidth / n
    x = x - (totalWidth - width) / 3
    plt.bar(x + width, statsList[0], yerr=stdList[0], width=width)
    plt.xticks(x, xlabels)

    # x = np.arange(1, len(df["name"].unique()) + 1)
    # for i in range(len(xlabels)):
    #     plt.plot(x, statsList[0][i], label=xlabels[i], linewidth=1)

    plt.ylabel('BIC')
    plt.legend(loc='best')
    plt.title('commit to goal ratio')  # Intention Consistency
    plt.show()

    samples = random.sample(range(len(df['likelihood2'])), numOfSamples)
    # likelihoodList = np.array(df['likelihood'])[samples]
    likelihoodList2 = np.array(df['likelihood2'])[samples]
    likelihoodList3 = np.array(df['likelihood3'])[samples]
    likelihoodList4 = np.array(df['likelihood4'])[samples]
    likelihoodList5 = np.array(df['likelihood5'])[samples]

    # likelihoodAll = np.prod(likelihoodList)
    likelihoodAll2 = np.prod(likelihoodList2)
    likelihoodAll3 = np.prod(likelihoodList3)
    likelihoodAll4 = np.prod(likelihoodList4)
    likelihoodAll5 = np.prod(likelihoodList5)

    __import__('ipdb').set_trace()
    # bic = calBIC(np.log(likelihoodAll))
    bic2 = calBIC(np.log(likelihoodAll2))
    bic3 = calBIC(np.log(likelihoodAll3))
    bic4 = calBIC(np.log(likelihoodAll4))
    bic5 = calBIC(np.log(likelihoodAll5))

    # print(bic)
    print(bic2)
    print(bic3)
    print(bic4)
    print(bic5)
