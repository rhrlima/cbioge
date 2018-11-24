qsub rjob -N c10-1 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar10.pickle c10_ckpt_1'
qsub rjob -N c100-1 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar100.pickle c100_ckpt_1'

qsub rjob -N c10-2 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar10.pickle c10_ckpt_2'
qsub rjob -N c100-2 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar100.pickle c100_ckpt_2'

qsub rjob -N c10-3 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar10.pickle c10_ckpt_3'
qsub rjob -N c100-3 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar100.pickle c100_ckpt_3'

qsub rjob -N c10-4 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar10.pickle c10_ckpt_4'
qsub rjob -N c100-4 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar100.pickle c100_ckpt_4'

qsub rjob -N c10-5 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar10.pickle c10_ckpt_5'
qsub rjob -N c100-5 -F 'run_experiment.py ../grammars/cnn.bnf ../datasets/cifar100.pickle c100_ckpt_5'