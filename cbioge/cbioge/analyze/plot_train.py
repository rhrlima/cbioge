import re

import matplotlib.pyplot as plt 


def read_log(file, metric):
    data = open(file, 'r').read().split('\n')

    metrics = ['loss', 'acc', 'jac', 'dic', 'spe', 'sen']

    loss = []
    val_loss = []

    train = []
    val = []

    for line in data:
        if 'loss' in line:
            m = re.findall(': (\\d\\.\\d+)', line)
            
            loss.append(m[0])
            train.append(m[metrics.index(metric)])

            if len(m) > 6:
                val_loss.append(m[6])
                val.append(m[metrics.index(metric)+6])

    return val_loss

if __name__ == '__main__':
    
    loss = read_log('results/best/best_acc1.log', 'acc')

# print(len(loss))
# print(len(train))

# print(len(val_loss))
# print(len(val))

    plt.plot(range(len(loss)), loss)
    plt.yticks([0, 1])
    #plt.plot(val_loss)
    plt.show()