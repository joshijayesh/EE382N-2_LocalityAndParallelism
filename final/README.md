Face recognition on GPU using PCA
=================================

Setup
-----

This project assumes a 8-bit grayscale pgm images. Expected user will download the target database, such as
yale or att_faces, and pass the path to the database.

Also, the project uses cxxopts for argparse which is added as a git submodule. If pulling the project for the first time,
can use this command: `git pull --recurse-submodules`, but if you've already pulled it and haven't initialized the
submodules, this should work: `git submodule update --init --recursive`

Building
--------

Just run `make`.


Runtime
-------

Currently supports these command line arguments:

## TRAIN

* -s {BASE DIRECTORY OF DATABASE}: program will recursively parse through the directory for all .pgm files
* -d {OUTPUT FILE}: file to save results onto, to be used by test
* -k {NUM COMPONENTS}: number of components to save off to file (default = max)
* -t train: Selects the current run as training phase
* -m {TRAIN TEST SPLIT}: Percentage for train-test split. I.e. -m 60 => 60% training
* -a {ALGORITHM}: Target algorithm for Eigenvectors (default jacobi). Supported: jacobi, qr
* -h/--help: Show help message

## TEST

* -s {BASE DIRECTORY OF DATABASE}: program will recursively parse through the directory for all .pgm files
* -d {OUTPUT FILE}: file to save results onto, to be used by test
* -I {INPUT FILE}: output file generated from training phase
* -k {NUM COMPONENTS}: number of components to use for testing (default = max, but set to training -k if -k specified during training)
* -t test: Selects the current run as test phase
* -m {TRAIN TEST SPLIT}: Percentage for train-test split. I.e. -m 60 => 60% training // Should be the same as what you ran for training!
* -h/--help: Show help message

