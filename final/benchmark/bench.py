import os
import re
from pathlib import Path
import subprocess
import statistics



PROJ = "../eigenfaces"
FACES = "../media/att_faces/"
# TRAIN_TEST_SPLITS = [50, 60, 70, 80, 90]
TRAIN_TEST_SPLITS = [50, 60]
# NUM_COMPONENTS = [5, 10, 25, 50, 75, 100]
NUM_COMPONENTS = [5, 10]
ITERATIONS = 1
METHODS = ["jacobi", "qr"]

ODIR = Path("tmp/")
TRAIN_OUT = ODIR / "train_out.txt"
TEST_OUT = ODIR / "test_out.txt"

RESULTS_OUT = "results.txt"

patt_check = re.compile("Time (.+): (\d+) us")


class ViviDict(dict):
    """Implementation of perl's autovivification feature."""

    def __init__(self, parent=None, ptr=None, *args, **kwargs):
        self.__vivify = True
        self.__parent = parent
        self.__ptr = ptr
        super(ViviDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if(self.__vivify):
            value = self[key] = type(self)(parent=self, ptr=key)  # retain local pointer to value
            return value  # faster to return than dict lookup
        else:
            raise KeyError(key)


def run_cmd(cmd):
    lines = []

    print(f"Executing cmd {cmd}")
    process = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE)

    while(True):
        line = process.stdout.readline()
        if not line: break
        lines.append(line.decode("utf-8"))
        print(lines[-1][:-1])

    return_code = process.poll()

    if(return_code != 0 and return_code is not None):
        print(f"ERROR! Subprocess exited with non-zero return code: {return_code}")
        print("Exiting benchmark early... time to check params")
        exit()

    return lines


def parse_training(out, result, split):
    for line in out:
        match = patt_check.search(line)
        if(match):
            grouping = match.group(1)
            time = int(match.group(2)) / 10 ** 3  # convert to milliseconds
            
            result[split].setdefault(grouping, []).append(time)


def parse_test(out, result, split, num_components):
    for line in out:
        match = patt_check.search(line)
        if(match):
            grouping = match.group(1)
            time = int(match.group(2)) / 10 ** 3  # convert to milliseconds
            
            result[split][num_components].setdefault(grouping, []).append(time)


def output_training(results, results_file):
    if(not(results)): return

    for split, split_dict in results.items():
        for grouping, vals in split_dict.items():
            print(f"{grouping:<16} Split: {split:>2} Mean: {statistics.mean(vals):>10.03f} StdDev: {statistics.stdev(vals) if len(vals) > 1 else 0.0:.06f}")
            results_file.write(f"{grouping:<16} Split: {split:>2} Mean: {statistics.mean(vals):>10.03f} StdDev: {statistics.stdev(vals) if len(vals) > 1 else 0.0:.06f}\n")


def output_test(results, results_file):
    if(not(results)): return

    for split, split_dict in results.items():
        for k, k_dict in split_dict.items():
            for grouping, vals in k_dict.items():
                print(f"{grouping:<16} Split: {split:>2} K: {k:>3} Mean: {statistics.mean(vals):>10.03f} StdDev: {statistics.stdev(vals) if len(vals) > 1 else 0.0:.06f}")
                results_file.write(f"{grouping:<16} Split: {split:>2} K: {k:>3} Mean: {statistics.mean(vals):>10.03f} StdDev: {statistics.stdev(vals) if len(vals) > 1 else 0.0:.06f}\n")
            


def main():
    results_file = open(RESULTS_OUT, "w")
    for method in METHODS:
        training_results = ViviDict()
        test_results = ViviDict()
        for iteration in range(ITERATIONS):
            for split in TRAIN_TEST_SPLITS:
                ODIR.mkdir(exist_ok=True)
                cmd = f"{PROJ} -s {FACES} -d {TRAIN_OUT} -m {split} -t train -a {method}"

                output = run_cmd(cmd)
                parse_training(output, training_results, split)

                for k in NUM_COMPONENTS:
                    cmd = f"{PROJ} -s {FACES} -i {TRAIN_OUT} -d {TEST_OUT} -m {split} -t test -k {k}"

                    output = run_cmd(cmd)
                    parse_test(output, test_results, split, k)

        print(f"<<<Results for {method}>>>")
        results_file.write(f"<<<Results for {method}>>>\n")
        output_training(training_results, results_file)
        output_test(test_results, results_file)
    results_file.close()


if(__name__ == '__main__'):
    main()

