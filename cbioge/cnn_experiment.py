from cbioge.experiments.parser import evolution_args

DEFAULTS = {
    'checkpoint': 'exp_cnn',

    'grammar': 'cbioge/assets/tmp_ignore/res_cnn_v2.json',
    'dataset': 'cbioge/assets/datasets/cifar10.pickle',

    'train-size': 5000,
    'valid-split': 0.1,

    'epochs': 10,
    'batch': 128,

    'pop': 10,
    'evals': 20,

    'selection': 'tournament',
    't-size': 5,

    'crossover': 'onepoint',
    'cross-rate': 1.0,

    'mutation': 'nonterm',
    'mut-rate': 1.0,

    'replace': 'elitsm',
    'elites': 0.25,

    'custom-op': 'halfhalf',
    'op-rate': 0.6,
}


def run_cnn_experiment(args):

    import keras
    import cbioge

    problem = cbioge.problems.CNNProblem(
        cbioge.grammars.Grammar(args.grammar),
        cbioge.datasets.Dataset.from_pickle(args.dataset, train_size=args.train_size),
        batch_size=args.batch,
        epochs=args.epochs,
        opt=keras.optimizers.Adam(lr=1e-4),
        train_args= {
            'validation_split': args.valid_split,
            'workers': 4,
            'use_multiprocessing': True,
        })

    cbioge.algorithms.GrammaticalEvolution(problem,
        pop_size=args.pop,
        max_evals=args.evals,
        selection=cbioge.algorithms.TournamentSelection(
            t_size=args.t_size, maximize=True),
        crossover=cbioge.algorithms.HalfAndHalfOperator(
            op1=cbioge.algorithms.OnePointCrossover(rate=1.0),
            op2=cbioge.algorithms.NonterminalMutation(
                parser=problem.parser, rate=1.0, end_index=7),
            rate=args.op_rate),
        replacement=cbioge.algorithms.ElitistReplacement(
            rate=args.elites, maximize=True),
        verbose=args.verbose
    ).execute(checkpoint=True)


if __name__ == '__main__':
    run_cnn_experiment(evolution_args(DEFAULTS))
