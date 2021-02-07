Lab1
====

Makefile will create 3 different executables that will each run different algorithms: basicmatmul cacheaware and cacheoblivious. These can be run using ./basicmatmul N M P, for example.

Iter_runs will create multiple iterations of each algorithm across multiple iterations, and specifying a few performance counters. Currently there are 3 sets: basic, l1perf, llcperf. Basic will measure simple cycles/instructions, L1/LLC will measure access/miss rates on L1 and LLC respectively. Note that this is split up into 3 different runs as we don't want to overload the perf counters.


To put it all together, running the setup looks like:

    $ make
    $ ./iter_runs.sh > launcher_jobs.sh
    $ sbatch sbatch_script.sh

Here, you will see a results folder which contains the dump of each of the permutations. There will also be a parsed.csv that contains average/std dev across all the runs.

Note that sbatch_script depends on the results folders created by iter_runs~ and will clear the results output before running to get a fresh start. If you want collective, then should remove that line.

