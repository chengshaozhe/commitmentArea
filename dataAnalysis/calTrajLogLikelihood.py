import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pickle
from scipy.stats import entropy
from scipy.interpolate import interp1d
from dataAnalysis import calculateSE


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


class SoftmaxPolicy:
    def __init__(self, Q_dict, softmaxBeta):
        self.Q_dict = Q_dict
        self.softmaxBeta = softmaxBeta

    def __call__(self, playerGrid, target1, target2):
        actionDict = self.Q_dict[(playerGrid, tuple(sorted((target1, target2))))]
        actionValues = list(actionDict.values())
        softmaxProbabilityList = calculateSoftmaxProbability(actionValues, self.softmaxBeta)
        softMaxActionDict = dict(zip(actionDict.keys(), softmaxProbabilityList))
        return softMaxActionDict


class CalRLLikelihood():
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, trajectory, aimAction, target1, target2):
        trajectory = list(map(tuple, trajectory))
        likelihoodList = []
        for playerGrid, action in zip(trajectory, aimAction):
            likelihood = self.policy(playerGrid, target1, target2).get(action)
            likelihoodList.append(likelihood)
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
        logLikelihood = np.prod(likelihoodList)
        return logLikelihood


def calculateGridDis(grid1, grid2):
    gridDis = np.linalg.norm(np.array(grid1) - np.array(grid2), ord=1)
    return gridDis


class IsCommit:
    def __init__(self,):

    def __call__(self, selfGrid, target1, target2):
        distanceDiff = abs(calculateGridDis(state, grid1) - calculateGridDis(state, grid2))
        pAvoidCommit = 1 / (distanceDiff + 1)

        return isCommited


class AvoidCommitPolicy:
    def __init__(self, goalPolicy):
        self.isCommited = isCommited
        self.goalPolicy = goalPolicy

    def __call__(self, state, target1, target2):
        return action


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


def calBICFull(logLikelihood, numOfObservations, numOfParas=3):
    bic = -2 * logLikelihood + np.log(numOfObservations) * numOfParas
    return bic


def calBIC(logLikelihood):
    bic = -2 * logLikelihood
    return bic

# def calBasePolicy(posteriorList, actionProbList):
#     basePolicyList = [np.multiply(goalProb, actionProb) for goalProb, actionProb in zip(posteriorList, actionProbList)]
#     basePolicy = np.sum(basePolicyList, axis=0)
#     return basePolicy


class BasePolicy:
    def __init__(self, goalPolicy):
        self.goalPolicy = goalPolicy

    def __call__(self, playerGrid, priorList):
        actionProbList = [self.goalPolicy(playerGrid, goal) for goal in [target1, target2]]
        basePolicyList = [np.multiply(prior, actionProb) for prior, actionProb in zip(priorList, actionProbList)]
        basePolicy = np.sum(basePolicyList, axis=0)
        return actionDis


if __name__ == '__main__':
    machinePolicyPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'machinePolicy'))
    softmaxBeta = 2.5
    goal_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGoalGird15_policy.pkl"), "rb"))
    goalPolicy = GoalPolicy(goal_dict, softmaxBeta)

    # Q_dict = pickle.load(open(os.path.join(machinePolicyPath, "noise0.1commitAreaGird15_policy.pkl"), "rb"))
    # softmaxPolicy = SoftmaxPolicy(Q_dict, softmaxBeta)

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
    # calLogLikelihood = CalRLLikelihood(softmaxPolicy)
    calImmediateIntentionLh = CalImmediateIntentionLh(goalPolicy)

    logLikelihoodSum = calImmediateIntentionLh(trajectory, aimAction, target1, target2)
    print(calBIC(np.log(logLikelihoodSum)))


###
    # resultsPath = os.path.join(os.path.join(DIRNAME, '..'), 'results')
    # participant = 'human'
    # dataPath = os.path.join(resultsPath, participant)
    # df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))), sort=False)
    # print(df.columns)
    # df['likelihood'] = df.apply(lambda x: calLogLikelihood(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

    # df['likelihood2'] = df.apply(lambda x: calImmediateIntentionLh(eval(x['trajectory']), eval(x['aimAction']), eval(x['target1']), eval(x['target2'])), axis=1)

    # import random
    # random.seed(147)
    # numOfSamples = 30
    # samples = random.sample(range(len(df['likelihood'])), numOfSamples)
    # likelihoodList = np.array(df['likelihood'])[samples]
    # likelihoodList2 = np.array(df['likelihood2'])[samples]

    # likelihoodAll = np.prod(likelihoodList)
    # likelihoodAll2 = np.prod(likelihoodList2)
    # bic = calBIC(np.log(likelihoodAll))
    # bic2 = calBIC(np.log(likelihoodAll2))

    # print(bic)
    # print(bic2)
