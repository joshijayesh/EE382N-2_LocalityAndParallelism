import argparse
import glob
import re
import csv
import numpy as np

PERF_PATT = re.compile(r'(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)\s+Joules\s+([\w/-]+)')
SAMPLES_PATT = re.compile(r'of event \'(.+):u\'')
EVENTS_PATT = re.compile(r'\(approx.\): (.+)')
MATMUL_PATT = re.compile(r'((?:|0|[1-9]\d?|100)(?:\.\d{1,2})?)%.* matmul')

TARGET_FREQ = 2100  # In MHZ


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_results(src):
    results = {}
    sample_name = ""
    events_count = 0
    matmul_pct = 0
    for output in glob.glob("{}/**.out".format(src)):
        with open(output, "r") as results_file:
            for line in results_file:
                match = PERF_PATT.search(line.split("#")[0])
                if(match):
                    num = match.groups()[0]
                    label = match.groups()[-1]
                    print(label)
                    if(represents_int(label)): continue
                    num = float(num.replace(',', ''))

                    results.setdefault(label, []).append(num)
                '''
                match = SAMPLES_PATT.search(line)

                if(match):
                    if(sample_name):
                        results.setdefault(sample_name, {}).setdefault("cnt", []).append(events_count)
                        results.setdefault(sample_name, {}).setdefault("matmul", []).append(matmul_pct)

                    sample_name = match.group(1)
                    events_count = 0
                    matmul_pct = 0.0
                    continue
                
                match = EVENTS_PATT.search(line)

                if(match):
                    events_count = int(match.group(1))
                    continue
                
                match = MATMUL_PATT.search(line)
                if(match):
                    matmul_pct = int((float(match.group(1)) / 100) * events_count)
                '''

    return results


def _calc_gflops(cycle, nmp):
    n = int(nmp.split("-")[0])
    num_ops = 2 * (n ** 3)   # Need to reconsider this for oblivious?

    freq_to_s = 1.0 / (TARGET_FREQ * (10 ** 6))
    execution_seconds = cycle * freq_to_s

    return (num_ops / execution_seconds) / (10 ** 9)


def main():
    base_path = "results/"
    results = {}
    for target in glob.glob("{}/**".format(base_path)):
        path_name = target.split("_")
        NMP = path_name[-1]
        alg_name = "_".join(path_name[:-1])

        results.setdefault(alg_name, {})[NMP] = parse_results(target)

    rows = [["Alg", "NMP", "Label", "Average -- Complete", "STD Dev -- Complete", "Average -- matmul", "STD Dev -- matmul"]]

    gflops_rows = [["Alg", "NMP", "GFLOPS Prog", "GFLOPS MatMul"]]
    for alg_name, alg_name_dict in results.items():
        for nmp, nmp_dict in alg_name_dict.items():
            for label, values in nmp_dict.items():
                # values['matmul'] = list(filter(lambda k: k != 0, values['matmul']))
                rows.append([alg_name, nmp, label, np.average(values), np.std(values)]) #, np.average(values)]) # np.std(values["matmul"])])

                if(label == 'cycles'):
                    gflops_prog = _calc_gflops(np.average(values["cnt"]), nmp)
                    gflops_matmul = _calc_gflops(np.average(values["matmul"]), nmp)

                    gflops_rows.append([alg_name, nmp, gflops_prog, gflops_matmul])

    with open(base_path + "parsed.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)

    # with open(base_path + "gflops.csv", "wb") as csv_file:
    #     csv_writer = csv.writer(csv_file)
    #    csv_writer.writerows(gflops_rows)

    print("Saving parsed file to {}".format(base_path + "parsed.csv"))
    # print("Saving glops calc file to {}".format(base_path + "gflops.csv"))
                    

if(__name__ == '__main__'):
    main()

