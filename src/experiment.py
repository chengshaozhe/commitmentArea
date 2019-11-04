
class Experiment():
    def __init__(self, normalTrial, specialTrial, writer, experimentValues, samplePositionFromCondition, drawImage, resultsPath):
        self.normalTrial = normalTrial
        self.specialTrial = specialTrial
        self.writer = writer
        self.experimentValues = experimentValues
        self.samplePositionFromCondition = samplePositionFromCondition
        self.drawImage = drawImage
        self.resultsPath = resultsPath

    def __call__(self, noiseDesignValues, conditionList):
        for trialIndex, condition in enumerate(conditionList):
            playerGrid, bean1Grid, bean2Grid, direction = self.samplePositionFromCondition(condition)
            if isinstance(noiseDesignValues[trialIndex], int):
                results = self.normalTrial(bean1Grid, bean2Grid, playerGrid, noiseDesignValues[trialIndex])
            else:
                results = self.specialTrial(bean1Grid, bean2Grid, playerGrid, noiseDesignValues[trialIndex])

            # results["noiseNumber"] = noiseDesignValues[trialIndex]
            # results["width"] = shapeDesignValues[trialIndex][0]
            # results["height"] = shapeDesignValues[trialIndex][1]
            # results["direction"] = direction
            # response = self.experimentValues.copy()
            # response.update(results)
            # self.writer(response, trialIndex)


if __name__ == "__main__":
    main()
