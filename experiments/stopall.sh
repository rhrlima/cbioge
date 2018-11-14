start=$1
end=$2

for p in `seq $start $end`; do
	echo qdel $p
done