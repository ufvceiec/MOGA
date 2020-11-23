import copy
import random
from jmetal.core.operator import Crossover
from jmetal.core.solution import Solution, FloatSolution, BinarySolution, PermutationSolution
from typing import List

"""
.. module:: crossover
   :platform: Unix, Windows
   :synopsis: Module implementing crossover operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullCrossover(Crossover[Solution, Solution]):

    def __init__(self):
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        return parents

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Null crossover'




class SBXCrossover(Crossover[FloatSolution, FloatSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBXCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            if  parents[0].number_of_variables > parents[1].number_of_variables:
                parent_aux = parents[1]
                parents[1] = parents[0]
                parents[0] = parent_aux
                offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
            print("longitud off 0", offspring[0].variables)
            print("longitud off 1", offspring[1].variables)


            for i in range(0, parents[0].number_of_variables):
                j = int((5*random.randint(0, (parents[1].number_of_variables/5) - 1)) + (i%5))
                print("N variables de parent1: ", parents[1].number_of_variables)
                print("valor de i", i)
                print("Valor de j", j)
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[j]
                
                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[j]
                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = c2
                            offspring[1].variables[j] = c1
                        else:
                            offspring[0].variables[i] = c1
                            offspring[1].variables[j] = c2
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[j] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[j] = value_x2
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'SBX crossover'



class DifferentialEvolutionCrossover(Crossover[FloatSolution, FloatSolution]):
    """ This operator receives two parameters: the current individual and an array of three parent individuals. The
    best and rand variants depends on the third parent, according whether it represents the current of the "best"
    individual or a random one. The implementation of both variants are the same, due to that the parent selection is
    external to the crossover operator.
    """

    def __init__(self, CR: float, F: float, K: float):
        super(DifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.K = K

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """ Execute the differential evolution crossover ('best/1/bin' variant in jMetal).
        """
        if len(parents) != self.get_number_of_parents():
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)

        number_of_variables = parents[0].number_of_variables
        rand = random.randint(0, number_of_variables - 1)

        for i in range(number_of_variables):
            if random.random() < self.CR or i == rand:
                value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])

                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            else:
                value = child.variables[i]

            child.variables[i] = value

        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'Differential Evolution crossover'
