import numpy as np
import random
from math import floor
import itertools as it


class UpdateWorld():
    def __init__(self, direction, dimension):
        self.direction = direction
        self.dimension = dimension

    def __call__(self, width, height, distance):
        direction = random.choice(self.direction)
        if direction == 45:
            pacmanPosition = (random.randint(1, self.dimension - distance - width - 2), random.randint(1, self.dimension - distance - height - 2))
            bean1Position = (pacmanPosition[0] + width + distance, pacmanPosition[1] + height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] + height + distance)

        elif direction == 135:
            pacmanPosition = (random.randint(1 + distance + width, self.dimension - 2), random.randint(1, self.dimension - 2 - distance - height))
            bean1Position = (pacmanPosition[0] - width, pacmanPosition[1] + height + distance)
            bean2Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] + height)

        elif direction == 225:
            pacmanPosition = (random.randint(1 + distance + width, self.dimension - 2), random.randint(1 + distance + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] - width - distance, pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] - width, pacmanPosition[1] - height - distance)
        else:
            pacmanPosition = (random.randint(1, self.dimension - distance - width - 2), random.randint(1 + distance + height, self.dimension - 2))
            bean1Position = (pacmanPosition[0] + width + distance, pacmanPosition[1] - height)
            bean2Position = (pacmanPosition[0] + width, pacmanPosition[1] - height - distance)

        return pacmanPosition, bean1Position, bean2Position, direction


def createNoiseDesignValue(condition, blockNumber):
    noiseDesignValuesIndex = random.sample(list(range(len(condition))), blockNumber)
    noiseDesignValues = np.array(condition)[noiseDesignValuesIndex].flatten().tolist()
    noiseDesignValues.append('special')
    return noiseDesignValues


def createShapeDesignValue(width, height, distance):
    shapeDesignValues = [[b, h, d] for b in width for h in height for d in distance]
    random.shuffle(shapeDesignValues)
    shapeDesignValues.append([random.choice(width), random.choice(height), random.choice(distance)])
    return shapeDesignValues


def main():
    pass


if __name__ == "__main__":
    main()
