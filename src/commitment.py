import numpy as np


def sigmoidScale(x, commitBeta):
    aNew = (1 / (1 + 1 * np.exp(- 75 * commitBeta * (x - (commitBeta + 1) / 2))) + 1) / 2
    return aNew


def goalCommited(probList, commitBeta):
    a, b = probList
    if a > 0.5:
        aNew = sigmoidScale(a, commitBeta)
        bNew = 1 - aNew
    else:
        bNew = sigmoidScale(b, commitBeta)
        aNew = 1 - bNew
    return [aNew, bNew]


def commitSigmoid(x, commitBeta):  # [1,5,10,20]
    return 1 / (1 + np.exp(- commitBeta * (x - 0.5)))


def goalCommit(intention, commitBeta):
    commitedIntention = [commitSigmoid(x, commitBeta) for x in intention]
    return commitedIntention


if __name__ == '__main__':
    # a = 0.65
    # p = [1 - a, a]
    # commitBeta = 0.8
    # print('old', p)
    # print('new', goalCommited(p, commitBeta))

    import matplotlib.pyplot as plt
    pList = list(np.arange(0, 1.1, 0.1))
    print(pList)
    for commitBeta in np.arange(1, 20, 2):
        pNew = []
        for p in pList:
            pNew.append(goalCommit([p, 1 - p], commitBeta))
        plt.plot(pList, pNew)
        plt.title('commitBeta={}'.format(commitBeta))
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.show()
