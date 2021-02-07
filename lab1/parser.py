import argparse
import glob
import re

PERF_PATT = re.compile(r'(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)\s+([\w-]+)')


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_results(src):
    for output in glob.glob("{}/**.out".format(src)):
        with open(output, "r") as results_file:
            for line in results_file:
                match = PERF_PATT.search(line.split("#")[0])
                if(match):
                    num = match.groups()[0]
                    label = match.groups()[-1]
                    if(represents_int(label)): continue
                    num = float(num.replace(',', ''))


def main():
    base_path = "results/"
    for target in glob.glob("{}/**".format(base_path)):
        path_name = target.split("_")
        NMP = path_name[-1]
        alg_name = "_".join(path_name[:-1])

        parse_results(target)
                    

if(__name__ == '__main__'):
    main()

