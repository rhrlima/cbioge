import os

runs = 30

s1 = [
    '238,135,125,101,240',
    '100,238,150,245,34,93,171,139',
    '197,23,173',
    '127,45,228',
    '168,185',
]

s2 = [
    '232,1,43,253,132,136,106,192',
    '122,220,98,97,135,106,76',
    '5,1,226,61',
    '77,124,135,57,126,157,12,206,207,77,124,135,57',
    '150,67',
]

for r in range(runs):
    for s in s1:
        os.system(f"./run_intel.sh {s} run_solution.py \
            '../grammars/cnn.bnf ../datasets/mnist.pickle {s} -b 128'")
    for s in s2:
        os.system(f"./run_intel.sh {s} run_solution.py \
            '../grammars/cnn.bnf ../datasets/c10.pickle {s} -b 128'")

os.system('watch qstat -n -1')
