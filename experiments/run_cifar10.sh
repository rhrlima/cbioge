num_runs=5

for i in `seq $num_runs`; do
	python3 run_experiment.py ../grammars/cnn.bnf ../datasets/cifar10.pickle c10_ckpt_$i/ 2>&1 | tee cifar10_$i.out
done