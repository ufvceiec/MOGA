import numpy as np
import random
from jmetal.core.operator import Selection
from typing import List, TypeVar


#De momento no hace falta.
class EstochasticUniversalSampling(Selection[List[S], List[S]]):

    def __init__(self,
                 max_population_size: int, reference_point: S,
                 dominance_comparator: Comparator = DominanceComparator()):
        super(RankingAndFitnessSelection, self).__init__()
        self.max_population_size = max_population_size
        self.dominance_comparator = dominance_comparator
        self.reference_point = reference_point
        
        def execute(self, front: List[S], weights: List[S]) -> List[S]:
            if front is None:
                raise Exception('The front is null')
            elif len(front) == 0:
                raise Exception('The front is empty')
                
            #LLamo a rouletteWheelSelection
                

class RouletteWheelSelection(Selection[List[S], S]):
    """Performs roulette wheel selection.
    """

    def __init__(self):
        super(RouletteWheelSelection).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        maximum = sum([solution.attributes["fitness"] for solution in front])
        rand = random.uniform(0.0, maximum)
        value = 0.0

        for solution in front:
            value += solution.attributes["fitness"]

            if value > rand:
                return solution

        return None
    
class BinaryTournamentSelection(Selection[List[S], S]):

    def __init__(self, comparator: Comparator = DominanceComparator()):
        super(BinaryTournamentSelection, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception('The front is null')
        elif len(front) == 0:
            raise Exception('The front is empty')

        if len(front) == 1:
            result = front[0]
        else:
            # Sampling without replacement
            i, j = random.sample(range(0, len(front)), 2)
            solution1 = front[i]
            solution2 = front[j]

            flag = self.comparator.compare(solution1, solution2)

            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                result = [solution1, solution2][random.random() < 0.5]

        return result