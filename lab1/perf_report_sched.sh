perf record -o $1_o.data -e $2 "${@:3}"
perf report -i $1_o.data --stdio --percent-limit 0 >> $1
rm $1_o.data
