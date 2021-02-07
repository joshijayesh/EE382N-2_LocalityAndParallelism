import argparse
import glob
import re
import csv
import numpy as np

PERF_PATT = re.compile(r'(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)\s+([\w-]+)')


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_results(src):
    results = {}
    for output in glob.glob("{}/**.out".format(src)):
        with open(output, "r") as results_file:
            for line in results_file:
                match = PERF_PATT.search(line.split("#")[0])
                if(match):
                    num = match.groups()[0]
                    label = match.groups()[-1]
                    if(represents_int(label)): continue
                    num = float(num.replace(',', ''))

                    results.setdefault(label, []).append(num)

    return results


def main():
    base_path = "results/"
    results = {}
    for target in glob.glob("{}/**".format(base_path)):
        path_name = target.split("_")
        NMP = path_name[-1]
        alg_name = "_".join(path_name[:-1])

        results.setdefault(alg_name, {})[NMP] = parse_results(target)

    rows = [["Alg", "NMP", "Label", "Average", "STD Dev"]]
    for alg_name, alg_name_dict in results.items():
        for nmp, nmp_dict in alg_name_dict.items():
            for label, values in nmp_dict.items():
                rows.append([alg_name, nmp, label, np.average(values), np.std(values)])

    with open(base_path + "parsed.csv", "wb") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(rows)

    print("Saving parsed file to {}".format(base_path + "parsed.csv"))
                    

if(__name__ == '__main__'):
    main()

