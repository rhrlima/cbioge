import datetime
import glob
import json
import os
import re
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from keras.models import model_from_json

from algorithms.solutions import GESolution


def ckpt2json(filename):
    population = load_checkpoint(filename)
    for i, s in enumerate(population):
        new_s = GESolution()
        new_s.id = s.id
        new_s.genotype = s.genotype
        new_s.phenotype = s.phenotype
        new_s.fitness = s.fitness
        new_s.evaluated = s.evaluated
        new_s.time = None
        new_s.params = None
        population[i] = new_s.to_json()
        del new_s

    filename = filename.replace('.ckpt', '.json')
    with open(filename, 'w') as f:
        json.dump({'population': population}, f)
    del population


def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data['population']


def load_json_ckpt(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data


def save_json_ckpt(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_all_checkpoints(folder, step, max_):
    solutions = []
    fitnesses = []
    means = []
    bests = []
    for i in range(step, max_+step, step):

        filename = os.path.join(folder, f'data_{i}.json')

        data = load_json_ckpt(filename)
        
        population = data['population']

        temp = [s['fitness'] for s in population]

        solutions += population
        fitnesses += temp
        means.append(np.mean(temp))
        bests.append(max(temp))

    return solutions, fitnesses, means, bests


def get_mean_from_group(group, step, max_):

    fitness = []
    params = []

    for i in range(0, max_, step):
        mean_fit_in_gen = []
        mean_par_in_gen = []

        for j, g in enumerate(group):

            print(i, j)
            mean_fit_in_group = np.mean([s['fitness'] for s in g[i:i+step]])
            mean_par_in_group = np.mean([s['params'] for s in g[i:i+step]])
            
            mean_fit_in_gen.append(mean_fit_in_group)
            mean_par_in_gen.append(mean_par_in_group)

        fitness.append(np.mean(mean_fit_in_gen))
        params.append(np.mean(mean_par_in_gen))

    return {'fitness': fitness, 'params': params}


def plot_evolution(group, labels, save_to=None):

    markers = ['o-', '*-', 'v-', 'x-', '+-']

    indexes = list(range(len(group[0])))

    for i, g in enumerate(group):
        plt.plot(g['fitness'], markers[i], label=labels[i])# line

    plt.legend(loc='lower right')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    if not save_to is None:
        plt.savefig(save_to)
    plt.show()


def plot_mean_fitness_params(group, save_to=None):

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    f, ax = plt.subplots(1, 5, sharey=True)

    for i, d in enumerate(group):
        
        fitness = d['fitness']
        params = d['params']

        ax[i].scatter(fitness, params, edgecolors='none', alpha=.5, color=colors[i])
        #ax[i].set_xticks([.0, .5, 1.])

    #ax[0].set_xlabel('Fitness')
    ax[0].set_ylabel('Parameters')
    f.text(0.5, 0.03, 'Fitness', ha='center')
    #f.text(0.01, 0.5, 'common Y', va='center', rotation='vertical')
    if not save_to is None:
        plt.savefig(save_to)
    plt.show()


def plot_fitness_params(group, labels, save_to=None):

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    f, ax = plt.subplots(1, 5, sharex=True, sharey=True)

    for i, d in enumerate(group):
        
        fitness = d['fitness']
        params = d['params']

        ax[i].scatter(fitness, params, edgecolors='none', alpha=.5, color=colors[i])
        ax[i].set_title(labels[i])
        #ax[i].set_xticks([.0, .5, 1.])

    #ax[0].set_xlabel('Fitness')
    ax[0].set_ylabel('Parameters')
    f.text(0.5, 0.03, 'Fitness', ha='center')
    #f.text(0.01, 0.5, 'common Y', va='center', rotation='vertical')
    if not save_to is None:
        plt.savefig(save_to)
    plt.show()


def ckpt_to_json(max_, step, folder):
    # convert the old pickle checkpoint to json data
    for i in range(step, max_+step, step):
        filename = os.path.join(folder, f'data_{i}')

        if os.path.exists(filename+'.json'):
            print('skipping', filename)
            continue

        if not os.path.exists(filename+'.ckpt'):
            print('DOES NOT EXISTS', filename)
            continue

        print(filename)

        ckpt2json(filename+'.ckpt')

        data = load_json_ckpt(filename+'.json')
        population = data['population']

        for i, s in enumerate(population):
            obj = GESolution(json_data=s)
            model = model_from_json(obj.phenotype)
            obj.params = model.count_params()
            population[i] = obj.to_json()

        save_json_ckpt({'population': population}, filename+'.json')
        del data
        del population


def parse_times(filename):
    with open(filename, 'r') as f:
        lines = f.read()

    times = []

    lines = lines.split('\n')
    for l in lines:
        if l.endswith('.ckpt') and 'data' in l:

            m = re.search('(\\w{3})\\s+(\\d{1,})\\s+(\\d{2}:\\d{2})', l)

            print(m)
            if m:
                year = '2019' if m.group(1) == 'Dec' else '2020'
                date = datetime.datetime.strptime(m.group(0)+' '+year, '%b %d %H:%M %Y')
                print(date)
                times.append(date)

    print('times', len(times))
    for d1, d2 in zip(times[:-1], times[1:]):
        print(d2-d1)


def get_best_from_group(data, group):

    best = None
    for key in group:
        curr_best = max(data[key], key=lambda x: x['fitness'])
        #print(key, curr_best)
        best = max(best, curr_best, key=lambda x: x['fitness']) if best is not None else curr_best
    return best


def load_data_and_group(files, convert=False):
    files.sort()
    data = {}
    group = []

    for file in files:
        print('loading:', file)
        if convert:
            ckpt_to_json(max_, step, file)

    for file in files:
        key = file.split('/')[-1]

        #print(key)
        if key[-1] == '1':
            group.append([key])
        else:
            group[-1].append(key)

        solutions, _, _, _ = load_all_checkpoints(file, step, max_)
        data[key] = solutions
    group.sort(key=lambda x: x[0], reverse=True)
    print(group)
    return data, group


def plot_all(data, group_names, output_names):

    labels = ['Simple', 'Regular', 'Moderate', 'Hard', 'Full']
    
    group_data = []
    group_names.sort(key=lambda x: x[0], reverse=True)
    labels.sort(reverse=True)
    for subgroup in group_names:
        print(subgroup)

        mean_data = []
        for f in subgroup:
            print(f)
            mean_data.append(data[f])

        mean_data = get_mean_from_group(mean_data, step, max_)
        group_data.append(mean_data)
    
    plot_evolution(group_data, labels, output_names[0])
    plot_fitness_params(group_data, labels, output_names[1])


def plot_evol_per_epoch(filenames, labels, save_to=None):

    markers = ['o-', '*-', 'v-', 'x-', '+-']

    all_data = []
    for filename in filenames:
        with open(filename, 'r') as f:
            data = f.read().split('\n')
            epochs = []
            for line in data:
                if 'val_loss' in line:
                    found = re.findall('\\d+\\.\\d+', line)
                    print(found)
                    epochs.append([float(f) for f in found])
            all_data.append(epochs)
    
    print(len(all_data))

    for i, epochs in enumerate(all_data):
        plt.plot([i[3] for i in epochs], markers[i], label=labels[i])
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if not save_to is None:
        plt.savefig(save_to)
    plt.show()


if __name__ == '__main__':

    step = 20
    max_ = 400

    no_train_files = glob.glob(os.path.join('analyze', 'TEXTURE', 'notrain', '*'))
    train_files = glob.glob(os.path.join('analyze', 'TEXTURE', 'train', '*'))

    # no_train_data, no_train_group = load_data_and_group(no_train_files)
    # output_names = ['evolution-notrain.png', 'fit-param-notrain.png']
    # plot_all(no_train_data, no_train_group, output_names)

    train_data, train_group = load_data_and_group(train_files, convert=True)
    output_names = ['evolution-train.png', 'fit-param-train.png']
    plot_all(train_data, train_group, output_names)
   
    # for group in no_train_group:
    #     print(group)
    #     best = get_best_from_group(no_train_data, group)
    #     print(best['fitness'], best['params'], best['genotype'])
    #     model = model_from_json(best['phenotype'])
    #     model.summary()

    # for group in train_group:
    #     print(group)
    #     best = get_best_from_group(train_data, group)
    #     print(best['fitness'], best['params'], best['genotype'])
    #     model = model_from_json(best['phenotype'])
    #     model.summary()

    # labels = ['Simple', 'Regular', 'Moderate', 'Hard', 'Full']
    # filenames = [
    #     'analyze/bestST.out', 
    #     'analyze/bestRT.out',
    #     'analyze/bestMT.out',
    #     'analyze/bestHT.out',
    #     'analyze/bestFT.out']
    # save_to = 'pos-trained-epochs.png'
    # plot_evol_per_epoch(filenames, labels, save_to)

    # filenames = [
    #     'analyze/bestS.out', 
    #     'analyze/bestR.out',
    #     'analyze/bestM.out',
    #     'analyze/bestH.out',
    #     'analyze/bestF.out']
    # save_to = 'pre-trained-epochs.png'
    # plot_evol_per_epoch(filenames, labels, save_to)

    # filenames = [
    #     'analyze/unetS.out', 
    #     'analyze/unetR.out',
    #     'analyze/unetM.out',
    #     'analyze/unetH.out',
    #     'analyze/unetF.out']
    # save_to = 'unet-epochs.png'
    # plot_evol_per_epoch(filenames, labels, save_to)