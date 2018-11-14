num_runs=5

for i in `seq $num_runs`; do
	echo qsub rjob -N mnist_$i -F "run_experiment.py ../grammars/cnn.bnf ../datasets/mnist.pickle mnist_ckpt"
	echo qsub rjob -N notmnist_$i -F "run_experiment.py ../grammars/cnn.bnf ../datasets/notmnist.pickle notmnist_ckpt"
	echo qsub rjob -N fashion-mnist_$i -F "run_experiment.py ../grammars/cnn.bnf ../datasets/fashion-mnist.pickle fashion_mnist_ckpt"
	echo qsub rjob -N cifar-10_$i -F "run_experiment.py ../grammars/cnn.bnf ../datasets/cifar-10.pickle cifar10_ckpt"
	echo qsub rjob -N cifar-100_$i -F "run_experiment.py ../grammars/cnn.bnf ../datasets/cifar-100.pickle cifar100_ckpt"
done