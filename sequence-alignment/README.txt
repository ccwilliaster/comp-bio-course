Christopher Williams
SUNet ID: ccwillia

This project was not discussed with anyone

usage: align.py [-h] [-l {debug,info,warning}] [-p] [-t]
                input_file output_file

This program implements a dynamic programming algorithm with affine gap
penalty as described in Durbin et al 2004. It is capable of performing either
a local alignment or an ends-free global alignment for two sequences. It takes
one input file and generates one output file. The input file must contain the
follwing: 

Line 1:   A sequence of letters indicating sequence X 
Line 2:   A sequence of letters indicating sequence Y 
Line 3:   An indication of whether local (1) or global (0) alignment is sought. 
Line 4:   A set of 4 space-delimited gap penalties for introducing gaps in 
          either X or Y, the negative of which will be used in calcs: 
            open_gap_x extend_gap_x open_gap_y extend_gap_y
            e.g., 0 0.7 0 0.4 
Line 5:   The number of possible symbols for seqeuence X 
Line 6:   A string of all possible symbols for sequence X 
Line 7:   The number of possible symbols for seqeuence Y 
Line 8:   A string of all possible symbols for sequence Y 
Line 9-?: Lines showing the score between an element in X and one in Y. There 
          are Nx*Ny lines (one for each possible x/y combination). Each line
          contains the entry for a cell in the match matrix with space-delimited 
          items containing 1. the row number, 2. the col number, 3. the letter 
          from alphabet X, 4. the letter from alphabet, 5. A match score between 
          the letters 
            e.g., 1 1 A T 0

The output file lists the score of the optimal alignment(s) that follow. 
Each alignment has one line for X (with necessary gaps), followed by a line 
for Y (with necessary gaps). Gaps are represented by "_". If there was more 
than one optimal alignment, all of them are listed (in no particular order) with 
each set of alignments separated by a blank line.

positional arguments:
  input_file            The input (directory/)file which contains the seqs to
                        align and the match matrix scores (see above for
                        format information.
  output_file           The ouput (directory/) file to which the optimal
                        alignment score and alignment(s) are written.

optional arguments:
  -h, --help            show this help message and exit
  -l {debug,info,warning}, --logger_level {debug,info,warning}
                        Specify the detail of print/logging messages.
  -p, --print_alignments
                        If this option is specified, alignment score and
                        sequences are printed to the screen.
  -t, --tests_only      If this option is specified, input/output files are
                        ignored and only unit tests are run.
