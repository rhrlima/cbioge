import sys

import matplotlib.pyplot as plt

import numpy as np


def draw_line(name, x_values, y_values, box=False):

    # List to hold x values.
    x_number_values = x_values#[1, 2, 3, 4, 5]

    # List to hold y values.
    y_number_values = y_values#[1, 4, 9, 16, 25]

    # Plot the number in the list and set the line thickness.
    if not box:
        plt.plot(x_number_values, y_number_values, linewidth=3)
    #plt.scatter(x_number_values, y_number_values, s=20, edgecolors='none', c='green')
    else:
        # Multiple box plots on one Axes
        fig, ax = plt.subplots()
        ax.boxplot(y_values)

    # Set the line chart title and the text font size.
    plt.title(name, fontsize=14)

    # Set x axes label.
    plt.xlabel("Generation", fontsize=10)

    # Set y axes label.
    plt.ylabel("Fitness", fontsize=10)

    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both', labelsize=9)

    # Save figure
    #plt.savefig(f'{name}.png')
    
    # Display the plot in the matplotlib's viewer.
    #plt.show()

if __name__ == '__main__':
    
    file_name = sys.argv[1]

    with open(file_name, 'r') as f:
        data = f.read()

    bests = []
    every = []
    for line in data.split('\n'):
        if 'best so far' in line: #best per gen
            fit = line.split(' ')[-1]
            #print(fit)
            bests.append(float(fit))
        if line.startswith('fitness'): #any
            fit = line.split(' ')[1]
            #print(fit)
            every.append(float(fit))

    means = []
    for i in range(len(bests)):
        print(i)
        means.append(np.mean(every[i*len(bests):(i+1)*len(bests)]))

    print('size', len(bests), len(every))
    print('best', max(bests))

    #plt.subplot(3, 1, 1)
    #draw_line('Bests', range(len(bests)), bests)
    #plt.subplot(3, 1, 2)
    #draw_line('Means', range(len(means)), means)
    #plt.savefig('plot.png')
    #plt.show()

    chunks = [every[i*len(bests):(i+1)*len(bests)] for i in range(len(bests))]
    draw_line('Means', range(len(bests)), chunks, box=True)
    plt.savefig('plot.png')
    #plt.show()
