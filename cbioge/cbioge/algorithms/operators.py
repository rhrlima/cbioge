import numpy as np
import math


class GeneticOperator:

    def export(self):
        return {'name': self.__str__(), 'config': self.__dict__}


# Replacement

class ReplaceWorst(GeneticOperator):

    def __init__(self, maximize=False):
        self.maximize = maximize

    def execute(self, population, offspring):
        population += offspring

        # TODO REVER
        for i, s in enumerate(population):
            if s.fitness is None:
                print(i, 'solution fitness is none, assigning -1')
                population[i].fitness = -1

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        return population[:len(offspring)]


class ElitistReplacement(GeneticOperator):

    def __init__(self, rate=0.1, maximize=False):
        self.rate = rate
        self.maximize = maximize

    def execute(self, population, offspring):

        # GAMBI
        for s in population:
            if s.fitness is None:
                s.fitness = -1

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        offspring.sort(key=lambda x: x.fitness, reverse=self.maximize)

        elites = int(math.floor(self.rate * len(population)))

        if elites < 1:
            print('[replace] not applied, elites less than 1')
            return population

        population = population[:elites] + offspring[:-elites]

        return population


# Prune

class GEPrune(GeneticOperator):

    ''' The Prune operator truncates a solution in a random point, it repeates
        for each solution of the offspring list

        prun_rate: chance of applying the operator
        offspring: list of solutions
    '''

    def __init__(self, prun_rate):
        self.prun_rate = prun_rate

    def __str__(self):
        return 'Prune'

    def execute(self, offspring):
        if np.random.rand() < self.prun_rate:
            for off in offspring:
                # not apply when solution has one gene
                if len(off.genotype) <= 1:
                    return
                cut = np.random.randint(1, len(off.genotype))
                off.genotype = off.genotype[:cut]


# Duplication

class GEDuplication(GeneticOperator):

    ''' Duplication selects part of a solution and copy-paste it at the end
        of that solution, it repeats for each solution in the offpring list.

        dupl_rate: change of applying the operator
        offspring: list of solutions
    '''

    def __init__(self, dupl_rate):
        self.dupl_rate = dupl_rate

    def __str__(self):
        return 'Duplication'

    def execute(self, offspring):
        if np.random.rand() < self.dupl_rate:
            for off in offspring:
                if len(off.genotype) > 1:
                    cut = np.random.randint(0, len(off.genotype))
                else:
                    # if one gene, setting cut to 1
                    cut = 1
                genes = off.genotype
                off.genotype = np.concatenate((genes, genes[:cut]))


# Custom

class HalfAndHalfOperator(GeneticOperator):

    def __init__(self, op1, op2, rate=0.5):
        self.op1 = op1
        self.op2 = op2
        self.rate = rate

    def execute(self, parents):

        offspring = parents[0].copy()

        if np.random.rand() < self.rate:
            #print('applied crossover')
            offspring = self.op1.execute(parents)
        else:
            #print('applied mutation')
            offspring = self.op2.execute(offspring)

        return offspring


class HalfAndChoiceOperator(GeneticOperator):

    def __init__(self, h_op, o_ops, h_rate=0.5, o_rate=[0.5]):
        self.h_op = h_op
        self.o_ops = o_ops
        self.h_rate = h_rate
        self.o_rate = o_rate

    def execute(self, parents):

        offspring = parents[0].copy()

        if np.random.rand() < self.h_rate:
            print('applied crossover')
            offspring = self.h_op.execute(parents)
        else:
            rand = np.random.rand()
            print(rand)
            for i in range(len(self.o_ops)):
                if rand < np.sum(self.o_rate[:i+1]):
                    print('applying', self.o_ops[i])
                    offspring = self.o_ops[i].execute(offspring)
                    break

        return offspring