num_runs=5

for i in `seq $num_runs`; do
	python3 run_experiment.py ../grammars/cnn.bnf ../datasets/cifar100.pickle c100_ckpt_$i/ 2>&1 | tee cifar100_$i.out
done