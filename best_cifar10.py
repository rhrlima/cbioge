import os

from keras.models import model_from_json

from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem
from cbioge.algorithms.solution import GESolution
from cbioge.datasets.dataset import read_dataset_from_pickle
from cbioge.problems.dnn import ModelRunner
from cbioge.datasets.dataset import read_dataset_from_pickle

from cbioge.utils import checkpoint as ckpt
from cbioge.utils.experiments import check_os


if __name__ == '__main__':

    import logging
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    check_os()

    solutions = [
        [[0], [1, 1, 0, 0, 0, 3], [1, 0, 1, 3], [], [], [0, 0, 0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 3, 2, 0, 2, 5], [1, 1, 0, 0, 1, 0], [4, 2, 2, 3], [0, 0, 1, 1, 0, 0], [2, 3, 0, 3, 2, 3], [4, 1], [0, 0]], 
        [[0], [1, 2], [0, 2], [0], [1], [0, 0], [0, 0], [0], [0], [0], [5, 3, 4, 5], [0, 0, 0], [3, 3], [0, 0, 1], [2, 3, 2], [3, 1], [0]], 
        [[0], [0, 1, 0, 1, 2], [1, 2], [0], [0], [0, 0, 0, 0], [0, 0], [0], [0, 0], [0, 0], [1, 1, 5, 2, 4, 2], [1, 1, 1, 0, 1, 1], [3, 0, 4, 2], [1, 0, 0, 0, 0, 1], [3, 0, 0, 1, 3, 1], [5, 5], [0]], 
        [[0], [0, 1, 0, 2], [2], [1], [1], [0, 0, 0], [0], [0], [0, 0], [0, 0], [5, 0, 3, 2], [1, 0, 1, 0, 0], [2, 4, 2], [0, 0, 1, 0, 0], [3, 1, 3, 3, 3], [0], [0]], 
        [[0], [1, 0, 1, 2], [1, 0, 0, 3], [0], [], [0, 0, 0], [0, 0, 0], [0], [0, 0], [0, 1], [2, 5, 0, 5, 1, 1], [1, 0, 1, 0, 1], [3, 0, 4], [1, 0, 0, 0, 0], [2, 2, 2, 1, 2], [0, 5, 1], [0]], 
    ]

    problem = CNNProblem(
        Grammar('data/grammars/cnn.json'), 
        read_dataset_from_pickle('data/datasets/cifar10.pickle'), 
        batch_size=32, 
        epochs=500)

    for genotype in solutions:
        
        print(genotype)

        solution = GESolution(genotype)

        solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
        
        #problem.evaluate(solution)

        model = model_from_json(solution.phenotype)
        model.compile(
            loss=problem.loss, 
            optimizer=problem.opt,
            metrics=['accuracy']
        )

        model.summary()

        # solution_path = os.path.join(ckpt.ckpt_folder, f'solution_{solution.id}')
        # runner = ModelRunner(model, path=solution_path, verbose=True)
        # runner.train_model(problem.x_train, problem.y_train, 
        #     problem.batch_size, 
        #     problem.epochs, 
        #     validation_data=(problem.x_valid, problem.y_valid), 
        #     timelimit=problem.timelimit, 
        #     save_weights=True)

        # runner.test_model(problem.x_test, problem.y_test, 
        #     problem.batch_size)
        
        # print(solution.fitness)
