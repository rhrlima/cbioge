import cbioge
from cbioge.experiments.parser import evolution_args
from cbioge.datasets import Dataset
from cbioge.utils import checkpoint as ckpt
from cbioge.algorithms.solution import Solution
import logging
import keras
import pickle as pck

DEFAULTS = {
    'checkpoint': 'exp_lstm',
    'dataset': 'assets/datasets/acoes_1.pickle',
    'grammar': 'assets/grammars/lstm_regressor.json',
    'train-size': 5000,
    'valid-split': 0.1,

    'epochs': 400,
    'batch': 128,

    'verbose':True
}

def save_solution(solution: Solution) -> None:
    json_solution = solution.to_json()
    filename = ckpt.SOLUTION_NAME.format(solution.id)
    ckpt.save_data(json_solution, filename)

logger = logging.getLogger('cbioge')
verbose = DEFAULTS['verbose']
def evaluation_best_solutions(args):
    dataset = cbioge.datasets.Dataset.from_pickle(args.dataset)
    data_ckpts = ckpt.get_files_with_name(ckpt.DATA_NAME.format('*'))
    if len(data_ckpts) == 0:
        if verbose:
            args.logger.debug('No checkpoint found.')
        return None
    last_ckpt = max(data_ckpts, key=ckpt.natural_key)
    data = ckpt.load_data(last_ckpt)
    evals = data['evals']
    population = [
        Solution.from_json(s) for s in data['population']
    ]
    if verbose:
        logger.debug('Latest checkpoint file found: %s', last_ckpt)
        logger.debug('Current evals: %d', evals)
        logger.debug('Population size: %d', len(population))

    print(args)

    problem = cbioge.problems.LSTMRegressorProblem(
        cbioge.grammars.Grammar(args.grammar),
        cbioge.datasets.Dataset.from_pickle(args.dataset),
        batch_size=args.batch,
        epochs=args.epochs,
        opt=keras.optimizers.Adam(learning_rate=1e-4),
        train_args= {
            'validation_split': args.valid_split,
            'workers': 4,
            'use_multiprocessing': True,
        })

    logger.info('Problema carregado')
    models = []

    for solution in population:
        model = problem.map_genotype_to_phenotype(solution)
        problem.evaluate(solution)
        models.append(model)
        save_solution(solution)
    
    pck.dump(models, open(f"{args.checkpoint}/models.bin", "wb"))
    
    logger.info(f'Models size: {len(models)}')







if __name__ == '__main__':
    evaluation_best_solutions(evolution_args(DEFAULTS))