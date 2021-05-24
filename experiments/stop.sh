start=$1
end=$2
for id in `seq $start $end`; do
  echo qdel $id
  qdel $id
done
