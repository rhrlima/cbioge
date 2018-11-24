start=$1
for file in *.ckpt; do
	echo $file
	test=$(echo $file | cut -c $start-30)
	echo $test
	mv $file $test
done

