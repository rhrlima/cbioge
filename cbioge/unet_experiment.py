from cbioge.experiments import evolution_args

DEFAULTS = {
    'checkpoint': 'exp_unet',

    'grammar': 'cbioge/assets/tmp_ignore/unet_restricted.json',
    'dataset': 'cbioge/assets/datasets/bsds500norm-2022.pickle',

    'epochs': 50,
    'batch': 10,

    'pop': 10,
    'evals': 100,
}


def run_experiment(args):

    import keras
    import cbioge
    from sklearn.utils import class_weight

    metric = cbioge.problems.dnns.image_metrics.WeightedMetric(
        w_dic=0.6,
        w_jac=0.1,
        w_sen=0.2,
        w_spe=0.1,
    )
    accuracy = metric.get_metric()

    dataset = cbioge.datasets.Dataset.from_pickle(args.dataset,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size)

    _, y_train = dataset.get_data('train')
    y_shape = (y_train.shape[0] * y_train.shape[1] * y_train.shape[2])

    problem = cbioge.problems.UNetProblem(
        cbioge.grammars.Grammar(args.grammar),
        dataset,
        batch_size=args.batch,
        epochs=args.epochs,
        opt=keras.optimizers.Adam(lr=1e-4),
        metrics=[accuracy],
        train_args= {
            'workers': 4,
            'use_multiprocessing': True,
            'class_weight': class_weight.compute_class_weight(
                'balanced',
                classes=[0, 1],
                y=y_train.reshape(y_shape)
            ),
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
            rate=args.cross_rate),
        replacement=cbioge.algorithms.ElitistReplacement(
            rate=args.elites, maximize=True),
        verbose=args.verbose
    ).execute(checkpoint=True)


if __name__ == '__main__':
    run_experiment(evolution_args(DEFAULTS))
