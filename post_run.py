import glob

from cbioge.datasets.dataset import read_dataset_from_pickle
from cbioge.grammars import Grammar
from cbioge.problems.classification import CNNProblem

from cbioge.analyze import plots
from cbioge.utils import checkpoint as ckpt

def test_validity():
    problem = CNNProblem(
        Grammar('data/grammars/res_cnn.json'), 
        read_dataset_from_pickle('data/datasets/cifar10.pickle'))

    num = 1000
    invalid = 0
    for _ in range(num):
        genotype = problem.parser.dsge_create_solution()
        mapping  = problem.parser.dsge_recursive_parse(genotype)
        try:
            model = problem.sequential_build(mapping)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        except Exception as e:
            invalid += 1

        print(f'\r{_+1}/{num} {invalid}/{num} {invalid/num*100:.3}%', end='')
    print(f'\r{_+1}/{num} {invalid}/{num} {invalid/num*100:.3}%')

if __name__ == '__main__':
    
    files = glob.glob('small/10484/data_*.ckpt')
    files.sort(key=lambda f: ckpt.natural_key(f))
    print(files)

    
