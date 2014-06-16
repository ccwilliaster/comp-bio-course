usage: md_ccwillia.py [-h] [--iF IF] [--kB KB] [--kN KN] [--nbCutoff NBCUTOFF]
                      [--m M] [--dt DT] [--n N] [--out OUT] [--plot]
                      [--track TRACK [TRACK ...]] [-t]
                      [-l {debug,info,warning}]
#..............................................................................
# Program

This is an implementation of a simplified molecular dynamics simulation. It
uses a simplified force field function that includes only two terms: 
    1. bonding interaction and 2. non-bonding interaction. 
Furthermore, it treats all atoms the same (same mass, no charges considered, 
etc.) and makes the assumption that non-bonding interactions are constant for 
the entirety of the simulation. 

It approximates equations of motion using the the Velocity Verletintegration 
algorithm. Briefly, this updates positions, forces, accelerations, and 
velocities for a given delta time step by estimating a half-time-step
velocity for each step. It has default parameters for the # of time steps in
the simulation, the size of each step, bonding and non-bonding spring
constants, but any of these can be specified. It takes as input a .rvc file
(see --iF for details) and writes to a similarly formatted .rvc file and a
.erg file containing kinetic, bonding and non-bonding potential energy, and
total energy, every 10 steps of the simulation. If the simulation becomes
unstable, a warning is thrown and the simulation will terminate. Additionally,
options are available for generating a plot of all energy terms over all steps
of the simulation, as well as tracking the euclidean distance between
specified atom pairs every 10 steps (see --plot and --track). Internally, it
uses Atom objects which have methods to write information about themselves,
and a bond indicator matrix mapping different bonding relationships between
atoms, which allows for an array-based implementation rather than numerous for
loops

optional arguments:
  -h, --help            show this help message and exit
  --iF IF               (dir/)Name of the input .rvc file which contains a
                        single header line, followed by one space-delimited
                        line per atom containing: atom ID, rx, ry, rz, vx, vy,
                        vz, and up to 4 other atom IDs corresponding to bonded
                        atoms.
  --kB KB               The spring constant for BONDED atoms in the force
                        field function. Default: 40000.0
  --kN KN               The spring constant for NON-BONDED atoms in the force
                        field function. Default: 400.0
  --nbCutoff NBCUTOFF   Distance within which atoms are considered as having
                        non-bonded interactions (if not covalently bonded).
                        Default: 0.5
  --m M                 Atom mass applied as a constant to all atoms. Default:
                        12.0
  --dt DT               Length of time step. Default: 0.0010
  --n N                 Number of time steps to iterate. Default: 1000
  --out OUT             Prefix of the output filename, e.g., <out>_out.rvc. By
                        default, the suffix-stripped input file name is used.
  --plot                If this option is specified, a plot of energy over the
                        course of the simulation will be made.
  --track TRACK [TRACK ...]
                        Specify a list of interaction pairs to track over the
                        course of the simulation. The euclidean distance
                        between each pair will be written to file every 10
                        steps. The expected format is ID1-ID2 to track the
                        distance betweeatoms with IDs 1 and 2. Output format
                        will be: step distance .
  -t, --tests_only      If this option is specified, all input is ignored and
                        only unit tests are run.
  -l {debug,info,warning}, --logger_level {debug,info,warning}
                        Specify the detail of print/logging messages.

#..............................................................................
# .euc files / plot generation

.euc files were generated with an R script that pulled information from three 
different distance-tracking files, one for each simulation of interest. As 
described in the arguments below, there is an option to track specified atom
pairs throughout the simulation, and this flag was used to generate the 
individual files used by the R script to make the .euc files. R was also used
to generate all plots (ggplot2 library specifically).
