#..............................................................................
Name:     Chris Williams
SUNet:    ccwillia
Language: Python
#..............................................................................

This file contains information for running three different programs:
tanimoto.py, pvalue.py, and networkgen.py

All functions for these programs are contained within an utilities
module chemoUtils.py. It contains utility functions useful for computing 
Tanimoto similarity scores for drug compounds, bootstrap p-values for inferring 
protein similarity based on the similarity of the drugs which bind the proteins, 
as well as several read/write helper functions.

The main Tanimoto score calculations are setup such that all pair-wise drug
Tanimoto scores are computed a single time, then referenced again later
several times. This makes it slower for tanimoto.py and pvalue.py because 
of the initial overhead, but faster for bootstrap calculations and network 
generation.

chemoUtils.py itself is organized by the following headers:
    - Readers
    - Writers
    - Tanimoto score functions
    - Bootstrap p-value functions
    - misc

Usage of tanimoto.py, pvalue.py, and networkgen.py follows:

#..............................................................................
#   tanimoto.py

usage: tanimoto.py [-h] [-l {10,20,30}] drugs targets outfile

This program reads in a table of drugs with associated chemical feature
vectors as well as a table mapping drugs to protein targets, and computes
Tanimoto similarity scores between all drug pairs. It writes these scores to
file and also writes a flag indicating if the drugs have a common protein
target (1) or not (0).

positional arguments:
  drugs                 The .csv (path/)file containing drug IDs, common
                        names, and space-delimited drug feature numbers.
  targets               The .csv (path/)file mapping drug IDs to their target
                        protein IDs and protein common names.
  outfile               The entire (path/)filename for output. Data will be
                        written to this exact filename.

optional arguments:
  -h, --help            show this help message and exit
  -l {10,20,30}, --logger_level {10,20,30}
                        Specify the detail of print/logging messages. 10, 20,
                        30 correspond to debug, info, and warning.

#..............................................................................
#   pvalue.py

usage: pvalue.py [-h] [-n N] [-l {10,20,30}] drugs targets proteinA proteinB

This program generates a bootstrap p-value for the comparison of two proteins.
The p-value is based on the similarity of the sets of compounds known to bind
the two proteins, versus sets of randomly chosen ligands.

positional arguments:
  drugs                 The .csv (path/)file containing drug IDs, common
                        names, and space-delimited drug feature numbers.
  targets               The .csv (path/)file mapping drug IDs to their target
                        protein IDs and protein common names.
  proteinA              Protein ID string for the first protein of interest
  proteinB              Protein ID string for the second protein of interest

optional arguments:
  -h, --help            show this help message and exit
  -n N                  Specifiy the number of iterations for calculating the
                        bootstrap p-value. Default: 100
  -l {10,20,30}, --logger_level {10,20,30}
                        Specify the detail of print/logging messages. 10, 20,
                        30 correspond to debug, info, and warning.
#..............................................................................
#   networkgen.py

usage: networkgen.py [-h] [-n N] [-l {10,20,30}] drugs targets nodes

This program generates a protein network based on the similarity of the ligand
sets which bind the proteins. For each protein pair, a Tanimoto summary score
is generated describing the similarity of the ligand sets which bind the two
proteins, and the significance of this is evaluated using a bootstrap p-value
for randomly chosen ligand sets. If the p-values satisfy a cutoff value, the
proteins are connected by a node.

positional arguments:
  drugs                 The .csv (path/)file containing drug IDs, common
                        names, and space-delimited drug feature numbers.
  targets               The .csv (path/)file mapping drug IDs to their target
                        protein IDs and protein common names.
  nodes                 The csv (path/)file containing swisprot ID, swisprot
                        name, and indications. All proteins in the file will
                        be considered for inclusion in network

optional arguments:
  -h, --help            show this help message and exit
  -n N                  Specifiy the number of iterations for calculating the
                        bootstrap p-value. Default: 100
  -l {10,20,30}, --logger_level {10,20,30}
                        Specify the detail of print/logging messages. 10, 20,
                        30 correspond to debug, info, and warning.
