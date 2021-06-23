import datetime

import glob
import re
import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from cbioge.utils import checkpoint as ckpt


def natural_key(string_):

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def load_data(files):

    parsed_data = {
        'fitness': [],
        'params': [],
        'time': [],

        'mean_fit': [],
        'mean_param': [],
        'mean_time': [],

        'best_fit': [],
    }

    for f in files:
        data = ckpt.load_data(f)
        population = data['population']
        
        print(f, len(population))

        mean_fit = []
        mean_param = []
        mean_time = []

        for s in population:
            mean_fit.append(s['fitness'])
            mean_param.append(s['params'])
            mean_time.append(s['time'].total_seconds())

        parsed_data['fitness'] += mean_fit
        parsed_data['params'] += mean_param
        parsed_data['time'] += mean_time

        parsed_data['mean_fit'].append(np.mean(mean_fit))
        parsed_data['mean_param'].append(np.mean(mean_param))
        parsed_data['mean_time'].append(np.mean(mean_time))

        parsed_data['best_fit'].append(max(mean_fit))

    return parsed_data


def plot(x, y, xlabel='x', ylabel='y', markers='*', label='label', name='plot.png'):
    plt.plot(x, y, markers, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(0, 30, step=5))
    #plt.yscale('log')
    plt.legend(loc='lower right')
    #plt.legend(loc='upper left')
    plt.savefig(name)
    #plt.clf()


def group_mean(data):
    mean_data = {
        'fitness': [],
        'params': [],
        'time': [],

        'mean_fit': [],
        'mean_param': [],
        'mean_time': [],

        'best_fit': [],
    }

    for key in mean_data:
        temp = []
        for d in data:
            temp.append(d[key])
        temp = np.mean(temp, axis=0)
        mean_data[key] = temp

    return mean_data


def load_and_plot(folders, tag, labels, best=False, name='plot'):

    for i, f in enumerate(folders):

        files = glob.glob(os.path.join('results', f, 'data_*.ckpt'))
        files.sort(key=lambda x: natural_key(x))

        data = load_data(files)

        if tag is 'fitness':
            if best:
                y_axis = data['best_fit']
            else:
                y_axis = data['mean_fit']
            plot(range(len(y_axis)), y_axis, 'generation', 'fitness', markers[i], labels[i], name=f'{name}-fit.png')
        elif tag is 'param':
            y_axis = data['mean_param']
            plot(range(len(y_axis)), y_axis, 'generation', 'params', markers[i], labels[i], name=f'{name}-param.png')
        elif tag is 'time':
            y_axis = data['mean_time']
            plot(range(len(y_axis)), y_axis, 'generation', 'time', markers[i], labels[i], name=f'{name}-time.png')
    #plt.show()
    plt.clf()


def load_group_plot(folders, tag, labels, best=False, name='plot'):

    for i, f in enumerate(folders):

        folder_runs = glob.glob(os.path.join('results', f))

        data = []
        for f_run in folder_runs:
            files = glob.glob(os.path.join(f_run, 'data_*.ckpt'))
            files.sort(key=lambda x: natural_key(x))
            data.append(load_data(files))

        data = group_mean(data)

        if tag is 'fitness':
            if best:
                y_axis = data['best_fit']
            else:
                y_axis = data['mean_fit']
            plot(range(len(y_axis)), y_axis, 'generation', 'fitness', markers[i], labels[i], name=f'{name}-fit.png')
        elif tag is 'param':
            y_axis = data['mean_param']
            plot(range(len(y_axis)), y_axis, 'generation', 'params', markers[i], labels[i], name=f'{name}-param.png')
        elif tag is 'time':
            y_axis = data['mean_time']
            plot(range(len(y_axis)), y_axis, 'generation', 'time', markers[i], labels[i], name=f'{name}-time.png')
    #plt.show()
    plt.clf()



if __name__ == '__main__':

    markers = ['o-', '*-', 'v-', 'x-', '+-', '--', ',-']

    #labels = ['Normal', 'NoMut', 'NoCross']

    #load_and_plot(['small/10484'], 'fitness', ['small'], 'dataset1')

    #load_and_plot(['normal', 'nomut', 'nocross'], 'fitness', labels, 'dataset1')
    #load_and_plot(['normal', 'nomut', 'nocross'], 'param', labels, 'dataset1')
    #load_and_plot(['normal', 'nomut', 'nocross'], 'time', labels, 'dataset1')

    #load_group_plot(['rand2?', 'tex2?'], 'fitness', labels, 'dataset2')
    #load_group_plot(['rand3?', 'tex3?'], 'fitness', labels, 'dataset3')
    #load_group_plot(['normal?', 'nomut?', 'nocross?'], 'fitness', labels, 'dataset1')
    #load_group_plot(['halfhalf1?', 'normal?', 'cross0?', 'cross1?', 'mut0?', 'mut1?', 'mut2?'], 'fitness', ['HaH', 'Normal', 'Cross', 'GCross', 'Mut', 'TMut', 'NTMut'], 'dataset1')
    
    #load_group_plot(['cross0?', 'cross1?', 'mut0?', 'mut1?', 'mut2?'], 'fitness', ['Cross', 'GCross', 'Mut', 'TMut', 'NTMut'], True, 'operators-best')
    #load_group_plot(['cross0?', 'cross1?', 'mut0?', 'mut1?', 'mut2?'], 'fitness', ['Cross', 'GCross', 'Mut', 'TMut', 'NTMut'], False, 'operators-mean')
    #load_group_plot(['halfhalf1?', 'halfchoice1?', 'normal?'], 'fitness', ['HaH', 'HaC', 'Normal'], True, 'evol-best')
    #load_group_plot(['halfhalf1?', 'halfchoice1?', 'normal?'], 'fitness', ['HaH', 'HaC', 'Normal'], False, 'evol-mean')
    
    #load_group_plot(['halfhalf1?', 'halfhalf2?', 'halfhalf3?'], 'fitness', ['Texture1', 'Texture2', 'Texture3'], 'datasets')

    #load_and_plot(['membrane1', 'membrane2', 'membrane3', 'membrane4', 'membrane5'], 'fitness', ['M1', 'M2', 'M3', 'M4', 'M5'], True, 'membrane-best')
    #load_and_plot(['membrane1', 'membrane2', 'membrane3', 'membrane4', 'membrane5'], 'fitness', ['M1', 'M2', 'M3', 'M4', 'M5'], False, 'membrane-mean')

    #load_and_plot(['borders1', 'borders2', 'borders3', 'borders4', 'borders5'], 'fitness', ['B1', 'B2', 'B3', 'B4', 'B5'], True, 'borders-best')
    #load_and_plot(['borders1', 'borders2', 'borders3', 'borders4', 'borders5'], 'fitness', ['B1', 'B2', 'B3', 'B4', 'B5'], False, 'borders-mean')

    #load_and_plot(['bsds5001', 'bsds5002', 'bsds5003', 'bsds5004', 'bsds5005'], 'fitness', ['BS1', 'BS2', 'BS3', 'BS4', 'BS5'], True, 'bsds-best')
    #load_and_plot(['bsds5001', 'bsds5002', 'bsds5003', 'bsds5004', 'bsds5005'], 'fitness', ['BS1', 'BS2', 'BS3', 'BS4', 'BS5'], False, 'bsds-mean')


    # load_group_plot(['rand2?', 'tex2?'], 'param', labels, 'dataset2')
    # load_group_plot(['rand3?', 'tex3?'], 'param', labels, 'dataset3')

    # load_group_plot(['rand2?', 'tex2?'], 'time', labels, 'dataset2')
    # load_group_plot(['rand3?', 'tex3?'], 'time', labels, 'dataset3')