import datetime
# import json
# import re
import glob
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# from algorithms.solutions import GESolution
# from grammars import BNFGrammar
# from problems import UNetProblem

# from keras.models import model_from_json

from utils import checkpoint as ckpt



# solutions = []

# for i, line in enumerate(data.split('\n')):
#     if 'started' in line:
#         solutions.append({})
#         m = re.match('<([0-9/ :]+)>', line)
#         if m:
#             datestr = m.group(1)
#             date = datetime.datetime.strptime(datestr[1:-1], '%x %X')
#             solutions[-1]['started'] = date
#     if 'ended' in line:
#         m = re.match('<([0-9/ :]+)>', line)
#         if m:
#             datestr = m.group(1)
#             date = datetime.datetime.strptime(datestr[1:-1], '%x %X')
#             solutions[-1]['ended'] = date
#             solutions[-1]['time'] = date-solutions[-1]['started']
#     if 'genotype' in line:
#         gen = line.split(' ', 1)[1]
#         solutions[-1]['gen'] = gen

# #solutions.sort(key=lambda x: x['time'], reverse=True)
# for s in solutions[:20]:
#     print(s['time'])#, s['gen'])


# def plot_time_x_complex(solutions):
#     parser = BNFGrammar('grammars/unet_mirror.bnf')
#     problem = UNetProblem(parser)
#     problem.read_dataset_from_pickle('datasets/membrane.pickle')

#     times = []
#     params = []
#     for s in solutions:
#         solution = GESolution(eval(s['gen']))
#         solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
#         model = model_from_json(solution.phenotype)
        
#         times.append(s['time'].total_seconds())
#         params.append(model.count_params())
#         #print(s['time'], model.count_params())

#     plt.scatter(range(len(solutions)), times, s=10, edgecolors='none', c='green')
#     plt.xlabel('# Params')
#     plt.ylabel('Time (sec)')
#     plt.show()

# plot_time_x_complex(solutions)


def plot(x, y, xlabel='x', ylabel='y', name='plot.png'):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(name)
    plt.clf()


if __name__ == '__main__':
    
    folder = sys.argv[1]

    files = glob.glob(os.path.join(folder, '*.ckpt'))
    files.sort(key=lambda f: ckpt.natural_key(f))

    print(files)
    
    population = []
    for file in files:

        data = ckpt.load_data(file)
        population += data['population']

    totaltime = datetime.timedelta()

    fitness = []
    params = []
    times = []
    #population.sort(key=lambda s: s['params'])
    for s in population:
        #print(s['fitness'], s['params'], s['time'].total_seconds())
        fitness.append(s['fitness'])
        params.append(s['params'])
        times.append(s['time'].total_seconds())

        totaltime += s['time']

    mean_fitness = []
    mean_params = []
    mean_times = []
    for i in range(0, 1000, 10):
        print(i, i+10, np.mean(fitness[i:i+10]))
        #mean_fitness.append(max(fitness[i:i+10]))
        mean_fitness.append(np.mean(fitness[i:i+10]))
        mean_params.append(np.mean(params[i:i+10]))
        mean_times.append(np.mean(times[i:i+10]))

    print('total time', totaltime)
    #plot(params, times, 'params', 'times')
    plot(range(len(mean_fitness)), mean_fitness, 'index', 'fitness', 'mean_fitness.png')
    plot(range(len(mean_params)), mean_params, 'index', 'params', 'mean_params.png')
    plot(range(len(mean_times)), mean_times, 'index', 'times', 'mean_times.png')
    #plot(range(len(fitness)), params, 'index', 'params', 'params.png')