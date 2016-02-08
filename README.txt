This submission requires the following environment: Python 2.7, numpy 1.9.3, ete2-2.3.10 (pip install ete2 will do the installation).

To use custom .nex file as an input, change the string path on line 6: Input = open('simple_tests/sample.nex').read().splitlines()
The alpha parameter can be changed on the next line.

The output of the program - largest log likelihood and the corresponding tree are printed to stdout and output.txt file.