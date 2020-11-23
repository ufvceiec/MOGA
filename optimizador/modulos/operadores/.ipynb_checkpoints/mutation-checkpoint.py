import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution, Solution, FloatSolution, IntegerSolution, PermutationSolution

"""
.. module:: mutation
   :platform: Unix, Windows
   :synopsis: Module implementing mutation operators.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullMutation(Mutation[Solution]):

    def __init__(self):
        super(NullMutation, self).__init__(probability=0)

    def execute(self, solution: Solution) -> Solution:
        return solution

    def get_name(self):
        return 'Null mutation'






class IntegerPolynomialMutation(Mutation[IntegerSolution]):

    def __init__(self, probability: float, distribution_index: float = 0.20):
        super(IntegerPolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        
        if random.random() < self.probability:
            for i in range(5):
                solution.lower_bound.append(solution.lower_bound[i])
                solution.upper_bound.append(solution.upper_bound[i])
                solution.variables.append(int(random.uniform(solution.lower_bound[i], solution.upper_bound[i])))

            solution.number_of_variables += 5
            
        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    rnd = random.random()
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.distribution_index + 1.0))
                        deltaq = 1.0 - val ** mut_pow

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = int(round(y))
        return solution

    def get_name(self):
        return 'Polynomial mutation (Integer)'


class SimpleRandomMutation(Mutation[FloatSolution]):

    def __init__(self, probability: float):
        super(SimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(solution.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                solution.variables[i] = solution.lower_bound[i] + \
                                        (solution.upper_bound[i] - solution.lower_bound[i]) * random.random()
        return solution

    def get_name(self):
        return 'Simple random mutation'


class UniformMutation(Mutation[FloatSolution]):

    def __init__(self, probability: float, perturbation: float = 0.5):
        super(UniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability:
                tmp = (random.random() - 0.5) * self.perturbation
                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def get_name(self):
        return 'Uniform mutation'


class NonUniformMutation(Mutation[FloatSolution]):

    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 0.5):
        super(NonUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                rand = random.random()

                if rand <= 0.5:
                    tmp = self.__delta(solution.upper_bound[i] - solution.variables[i], self.perturbation)
                else:
                    tmp = self.__delta(solution.lower_bound[i] - solution.variables[i], self.perturbation)

                tmp += solution.variables[i]

                if tmp < solution.lower_bound[i]:
                    tmp = solution.lower_bound[i]
                elif tmp > solution.upper_bound[i]:
                    tmp = solution.upper_bound[i]

                solution.variables[i] = tmp

        return solution

    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float):
        return (y * (1.0 - pow(random.random(),
                               pow((1.0 - 1.0 * self.current_iteration / self.max_iterations), b_mutation_parameter))))

    def get_name(self):
        return 'Uniform mutation'


class PermutationSwapMutation(Mutation[PermutationSolution]):

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        rand = random.random()

        if rand <= self.probability:
            pos_one, pos_two = random.sample(range(solution.number_of_variables - 1), 2)
            solution.variables[pos_one], solution.variables[pos_two] = \
                solution.variables[pos_two], solution.variables[pos_one]

        return solution

    def get_name(self):
        return 'Permutation Swap mutation'


class ScrambleMutation(Mutation[PermutationSolution]):

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability:
                point1 = random.randint(0, len(solution.variables[i]))
                point2 = random.randint(0, len(solution.variables[i]) - 1)

                if point2 >= point1:
                    point2 += 1
                else:
                    point1, point2 = point2, point1

                if point2 - point1 >= 20:
                    point2 = point1 + 20

                values = solution.variables[i][point1:point2]
                solution.variables[i][point1:point2] = random.sample(values, len(values))

        return solution

    def get_name(self):
        return 'Scramble'