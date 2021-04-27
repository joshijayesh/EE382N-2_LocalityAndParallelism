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

* -s {BASE DIRECTORY OF DATABASE}: program will recursively parse through the directory for all .pgm files
* -n {MAX NUM TO SEARCH}: used to put an upper limit on number of images to read in (for test purposes)
* -h/--help: Show help message

