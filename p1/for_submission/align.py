#!/usr/bin/env python
info="""This program implements a dynamic programming algorithm with affine
        gap penalty as described in Durbin et al 2004. It is capable of 
        performing either a local alignment or an ends-free global alignment
        for two sequences.

        It takes one input file and generates one output file. The input file
        must contain the follwing:
        Line 1:  A sequence of letters indicating sequence X
        Line 2:  A sequence of letters indicating sequence Y
        Line 3:   An indication of whether local (1) or global (0) alignment is 
                  sought.
        Line 4:   A set of 4 space-delimited gap penalties for introducing gaps
                  in either X or Y, the negative of which will be used in calcs: 
                  open_gap_x extend_gap_x open_gap_y extend_gap_y
                    e.g., 0 0.7 0 0.4
        Line 5:   The number of possible symbols for seqeuence X
        Line 6:   A string of all possible symbols for sequence X
        Line 7:   The number of possible symbols for seqeuence Y
        Line 8:   A string of all possible symbols for sequence Y
        Line 9-?: Lines showing the score between an element in X and one in Y.
                  There are Nx*Ny lines (one for each possible x/y combination)
                  Each line contains the entry for a cell in the match matrix
                  with space-delimited items containing 1. the row number,
                  2. the col number, 3. the letter from alphabet X, 4. the 
                  letter from alphabet, 5. A match score between the letters
                    e.g., 1 1 A T 0

        The output file lists the score of the optimal alignment(s) that follow
        Each alignment has one line for X (with necessary gaps) followed by a
        line for Y (with necessary gaps). Gaps are represented by "_". If 
        there was more than one optimal alignment, all of them are listed (in
        no particular order) with each set of alignments separated by a  blank
        line.
     """

__author__ = "ccwilliams"
__date__   = "2013-09-30"

import re
import numpy as np
import argparse
import logging

#...............................................................................
#   Define input arguments with the argparse module
prsr = argparse.ArgumentParser(description=info)

prsr.add_argument("input_file",
                  help = "The input (directory/)file which contains the seqs"
                         " to align and the match matrix scores (see above for"
                         " format information.")
prsr.add_argument("output_file",
                  help = "The ouput (directory/) file to which the optimal alig"
                         "nment score and alignment(s) are written.")
prsr.add_argument("-l", "--logger_level",
                  default="warning", choices=["debug", "info", "warning"],
                  help  = "Specify the detail of print/logging messages.")
prsr.add_argument("-p", "--print_alignments",
                  action="store_true",
                  help  = "If this option is specified, alignment score and"
                          " sequences are printed to the screen.")
prsr.add_argument("-t", "--tests_only",
                  action="store_true",
                  help  = "If this option is specified, input/output files are"
                          " ignored and only unit tests are run.")
#...............................................................................
#   Global variables 
EPSILON     = 0.00001 # difference cutoff for float comparison
REPAT_MATCH = "(\w+)\s+(\w+)\s+([-]*\d+[.]*\d*)" # regex pattern for finding
                                                 # x, y, match_score in input
#...............................................................................
#   Helper functions
def getLogger(logger_level):
    """Returns a console-only (i.e. no writing) logger with level set to 
       logger_level
       @param logger_level string, "debug"/"info"/"warning"
    """
    levels = {"debug":logging.DEBUG, 
              "info":logging.INFO, 
              "warning":logging.WARN}
    logger    = logging.getLogger("align.py")
    console   = logging.StreamHandler()
    formatter = logging.Formatter('%(name)-12s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(levels[logger_level])
    return logger

def parse_input(input_file):
    """This function parses the input file and returns a tuple of 10 items
       (see @return)

       @param input_file    string specifying the location of the input file
                            **nb: this function does NOT check file format.
       @return tuple< str<seq_x>, str<seq_y>, bool<local>,
                      int<dx>, int<ex>, int<dy>, int<ey>, 
                      str<letters_x>, str<letters_y>, dct<match_matrix> >
    """
    logger.info("Parsing input file from: %s" % input_file)

    # Parse and clean up input file lines
    f     = open(input_file, "r")
    lines = [ line.strip() for line in f.readlines() if len(line) != 1 ]
    f.close()

    # Unpack all values except the match matrix 

    seq_x, seq_y         = lines[0], lines[1]
    local                = True if int( lines[2] ) else False
    dx, ex, dy, ey       = [ float(penalty) for penalty in lines[3].split(" ") ]
    n_seqx, n_seqy       = [ int(N) for N in [lines[4], lines[6]] ]
    letters_x, letters_y = lines[5], lines[7]

    # Construct the match matrix
    dct_match_matrix = load_match_matrix(letters_x, letters_y, lines[8:])

    logger.info("Peforming %s alignment for sequences (x,y) of length (%i, %i)"\
                % ("local" if local else "ends-free global", len(seq_x), len(seq_y)))
    return (seq_x, seq_y, local, dx, ex, dy, ey, letters_x, letters_y, dct_match_matrix)

def load_match_matrix(letters_x, letters_y, sim_list):
    """This function returns a match matrix based on data from the input 
       file in the form of a dictionary:
            dict< str<letters_x> : dict< str<letters_y> : float<match> > >

       @param letters_x     A str containing all possible letters in seq_x
       @param letters_y     A str containing all possible letters in seq_y
       @param sim_list      A list of input_file items for the match 
                            matrix, one for each cell in the form:
                                str<row col xi yi match>
       @return dict (described above)
    """
    # Check the number of values
    assert len(sim_list) == ( len(letters_x) * len(letters_y) )

    # Initialize all keys in the match matrix dictionary
    sim_dct = { x : { y : np.nan for y in letters_y } for x in letters_x } 
    
    # Fill the dictionary
    for cell_data in sim_list:
        x, y, match_str = re.findall(REPAT_MATCH, cell_data)[0]
        sim_dct[x][y] = float(match_str)

    return sim_dct

def init_score_matrices(seq_x, seq_y, dx, ex, dy, ey, local):
    """This function initializes the scores matrices M / Ix / Iy. Specifically,
       it initializes three numpy arrays of dtype = (np.float64, "S3"), i.e. a 
       length 2 tuple to hold a score (float64) and pointer string (max len 3).
       
       Each array has len(seq_x)+1 rows and  len(seq_y)+1 columns.
       Only the first row and the left-most columns are initialized with values
       Other cells are left empty (garbage values). 

       @param seq_x     str representing sequence x (seq will become cols)
       @param seq_y     str representing sequence y (seq will become rows)
       @param dx        float score for opening a gap in x
       @param ex        float score for extending a gap in x
       @param dy        float score for opening a gap in y
       @param ey        float score for extending a gap in y 
       @param local     bool, whether the alignment is local (else global)
       @return tuple< 3x np.ndarray >
    """
    # Define the data type for each cell in the score matrices, and calc shape
    dtype      = np.dtype([('score', np.float64), ('pointer', 'S3')])
    nrow, ncol = len(seq_y) + 1, len(seq_x) + 1

    # Initialize the matrices as empty arrays. 
    # We will iterate over all cells so the garbage values will not be an issue
    M, Ix, Iy = [ np.empty(shape=(nrow, ncol), dtype=dtype) for i in range(3) ]

    # Initialize the M matrix, 
    # 1st row/col are -inf as they are not allowed, except the origin which = 0
    M[0,0]  = (0, "NA")
    M[0,1:] = [ (-np.inf, "NA") for i in range(ncol-1) ] 
    M[1:,0] = [ (-np.inf, "NA") for i in range(nrow-1) ]

    # Initialize the Ix matrix. The first row=0 (no penalty for ends free/local), 
    # and the first col (including 0) is -inf as this is not allowed. Setting
    # the origin as -inf prevents redundant right substrings
    Ix[0,:] = [ (0, "NA")       for i in range(ncol) ]
    Ix[:,0] = [ (-np.inf, "NA") for i in range(nrow) ] #range(nrow-1) ]

    # Initialize the Iy matrix. The first col = 0 (no penalty for ends free),
    # and the first row (again including 0) is -inf as this is not allowed
    Iy[:,0] = [ (0, "NA")       for i in range(nrow) ]
    Iy[0,:] = [ (-np.inf, "NA") for i in range(ncol) ] 

    logger.info("Score matrices initialized with %i rows, %i cols" % (nrow, ncol))
    return (M, Ix, Iy)

def make_score_matrices(seq_x, seq_y, dx, ex, dy, ey, local, match_matrix):
    """This function creates the dynamic programming tables for the alignment,
       based on gap penalties dx/ex/dy/ey, the match matrix scores, and whether 
       the alignment local or global ends-free. 
       Three matrices (M, Ix, Iy) are constructed in which each cell contains
       a length 2 tuple containing 
        0. the score for the alignment so far, and
        1. a string pointing to the matrix from which the current cell/alignment
           was derived (e.g. "M", "X", "Y", "MY", "YXM", "NA", etc.). 
       The matrices, along with the best score (given local or global alignment) 
       and a list of tuples of the indices of that score are returned (in case 
       multiple cells score equivalently). 

       @param seq_x, _y strings representing the full x&y sequences
       @param dx, dy    floats representing the gap opening penalties for x&y
       @param ex, ey    floats representing the gap extension penalties for x&y
       @param local     bool, if the alignment is local (else global)
       @param match_matrix dict<str<x_letter>:dict<str<y_letter>:float<score>>>
       @return M, Ix, Iy, 
               list<tuple<int<start_row>,int<start_col>>>, float<max_score>
    """
    # Initialize the matrices (the first row and column)
    M, Ix, Iy  = init_score_matrices(seq_x, seq_y, dx, ex, dy, ey, local)
    nrow, ncol = M.shape # all same size

    # Now fill the matrices from left to right, top to bottom, skipping the
    # first row and cols. At each index we will populate all matrices 
    # We will keep track of the maximum value as we go.
    start_indices, curr_max = [], -np.inf
    
    for row in range(nrow)[1:]:
        for col in range(ncol)[1:]:
            # Get match score. 
            x_letter, y_letter = seq_x[col - 1], seq_y[row - 1] # -1 for 1st row/col
            match_score = match_matrix[x_letter][y_letter]      # first key = x

            # Get the best scores for each matrix 
            M[row, col]  = get_M_max(M, Ix, Iy, match_score, local, col, row)
            Ix[row, col] = get_Ix_max(M, Ix, dy, ey, local, col, row)
            Iy[row, col] = get_Iy_max(M, Iy, dx, ex, local, col, row)

            # Since the desired alignment is local or ends-free global, we only
            # consider a traceback from matrix M. If the desired alignment is 
            # global, we also restrict ourselves to the outer margins of M
            check_max = True if (local or (row == nrow-1 or col == ncol-1)) \
                        else False 

            if check_max:
                curr_max, start_indices = update_max(M, row, col, 
                                                     curr_max, start_indices)

    indices_str = "/".join([str(indices) for indices in start_indices])
    logger.debug("Max score in M @ (row, col): %s" % indices_str)

    return (M, Ix, Iy, start_indices, curr_max)

def update_max(M, row, col, curr_max, start_indices):
    """This function updates the maximum value and starting indices during
       score matrix constructions. If the current position in M is better
       @param M             The M matrix (only matrix considered for traceback)
       @param row           Current row
       @param col           Current col
       @param curr_max      The current maximum score, to which score at the
                            current indices will be compared
       @param start_indices The indices of the current maximum value
       @return updated_curr_max, updated_start_indices
    """
    curr_score, curr_index = M[row, col][0], (row, col)
    # Anything is better than -inf (this may update to -inf as well)
    if curr_max == -np.inf: return curr_score, [curr_index]
    # If equal
    elif abs(curr_max - curr_score) < EPSILON:
        start_indices.append( curr_index )
        return curr_max, start_indices
    # If new better than old
    elif curr_score > curr_max:
        return curr_score, [curr_index]
    # If old better than new
    else:
        return curr_max, start_indices

def get_M_max(M, Ix, Iy, match_score, local, curr_col, curr_row):
    """This is the recurrence relation function for the M matrix. It returns a 
       tuple to be used in the traceback for the best alignment so far:
       tuple< float< best_score >, str<pointer_key(s)> 
    """
    score_dict  = { "M" :  M[curr_row - 1, curr_col - 1][0] + match_score,
                    "X" : Ix[curr_row - 1, curr_col - 1][0] + match_score,
                    "Y" : Iy[curr_row - 1, curr_col - 1][0] + match_score }

    if local: score_dict["NA"] = 0    
    return compare_scores(score_dict)

def get_Ix_max(M, Ix, dy, ey, local, curr_col, curr_row):
    """This is the recurrence relation function for the Ix matrix. It returns a
       tuple to be used in the traceback for the best alignment so far:
       tuple< float< best_score >, str<pointer_key(s)>
    """
    score_dict = { "M" :  M[curr_row, curr_col - 1][0] - dy,
                   "X" : Ix[curr_row, curr_col - 1][0] - ey }

    if local: score_dict["NA"] = 0
    return compare_scores(score_dict)

def get_Iy_max(M, Iy, dx, ex, local, curr_col, curr_row):
    """This is the recurrence relation function for the Iy matrix. It returns a
       tuple to be used in the traceback for the best alignment so far:
       tuple< float< best_score >, str<pointer_key(s)>
    """
    score_dict = { "M" :  M[curr_row - 1, curr_col][0] - dx,
                   "Y" : Iy[curr_row - 1, curr_col][0] - ex }    

    if local: score_dict["NA"] = 0
    return compare_scores(score_dict)

def compare_scores(score_dict):
    """Given a dict of key-score pairs, this function returns the tuple to be 
       used in the traceback: tuple< score, str<pointer_key(s)> >
       Score is the maximum score in the score_dict, and the key string is the
       key with that score. 
       
       In the case that multiple keys have the same score, key names are 
       concatenated into a single string (no spaces). 
       In the case that all scores are -np.inf, tuple< -np.inf, "NA" >
       is returned

       @param score_dict dict< str<matrix_key> : float<score> >
       @return tuple< float<max score>, str<pointer key name(s)> >
    """
    maxes = [] # list for storing the key(s) with current highest score
    for key, score in score_dict.iteritems():
        if abs(score) == np.inf: continue # store only finite scores
        elif maxes:
            if abs(score_dict[ maxes[0] ] - score) < EPSILON: # equal scores
                if key == "NA": maxes = [key] # store only one 0 value
                else:           maxes.append(key)
            elif score_dict[ maxes[0] ] < score: # change the max, else continue
                maxes = [key]
        else:
            maxes.append(key) # populate with first valid key
        
    return (score_dict[maxes[0]], "".join(maxes)) if maxes else (-np.inf, "NA")

def rec_traceback(x_seq, y_seq, curr_row, curr_col, curr_matrix, matrix_dict, 
                  local):
    """A recursive implementation of the sequence alignment traceback. During
       each call the current indices, matrix, and pointers are used to determine 
       the next move. The x and y letters represented by the current state are
       added to the alignments returned by recursive calls to the pointer cells
       and the concatenated alignments are returned.

       @param x_seq       string representing the entire x sequence
       @param y_seq       string representing the entire y sequence
       @param curr_row    int for the current row position in the matrix
       @param curr_col    int for the current col position in the matrix
       @param curr_matrix String key representing the current matrix (M/X/Y)
       @param matrix_dict dict< "M": M, "X": Ix, "Y": Iy >
       @param local       bool, if the alignment is local
       @return list< tuple< str<x alignment>, str<y alignment> >, ... >
    """
    # Get information about the current cell and the downstream cell(s)
    curr_pointers = matrix_dict[curr_matrix][curr_row, curr_col][1]  
    next_row, next_col = \
        get_next_matrix_indices(curr_matrix, curr_row, curr_col)
    x_curr, y_curr = \
        get_curr_letters(x_seq, y_seq, curr_matrix, curr_row, curr_col)

    # Iterate through the pointers, collecting downstream (left) seqs as we go
    list_of_seq_tups = []
    for pointer in curr_pointers:
        next_pointer = matrix_dict[pointer][next_row, next_col][1]

        # Base case: Stop if the cell pointed TO points to NA
        if (next_pointer == "NA"):
            # Limit terminal tracebacks to M if possible for loc alignments
            if (local) and (pointer != "M") and ("M" in curr_pointers): continue
            return [(x_curr, y_curr)]

        # Otherwise collect sequences and add the current letters below
        left_alignments = rec_traceback(x_seq, y_seq, 
                                        next_row, next_col, 
                                        pointer, matrix_dict, local)

        list_of_seq_tups.extend(left_alignments)

    # For tracing recursion stats during debugging    
    logger.debug("In matrix %s (%i,%i)" % (curr_matrix, curr_row, curr_col))
    logger.debug("Pointers: %s, Curr X,Y: %s,%s" % (curr_pointers, x_curr, y_curr))
    logger.debug("Alignments so far:\n\t%s" % \
                 " / ".join([str(al) for al in list_of_seq_tups]))

    return \
        [(x_left + x_curr, y_left + y_curr) for x_left, y_left in list_of_seq_tups]       

def get_next_matrix_indices(curr_matrix_type, curr_row, curr_col):
    """Given the current matrix type and indices, this function returns the 
       row and col of the next cell you will move to.

       @param curr_matrix_type string M/X/or Y, representing the current matrix
       @param curr_row         int representing the index of the current row
       @param curr_col         int representing the index of the current col
       @return tuple<next_row, next_col>
    """
    next_row = curr_row if curr_matrix_type == "X" else curr_row - 1
    next_col = curr_col if curr_matrix_type == "Y" else curr_col - 1
    return (next_row, next_col)

def get_curr_letters(seq_x, seq_y, curr_matrix_type, curr_row, curr_col):
    """Given the current matrix type and the current indices, this function 
       returns the x and y alignment letters that correspond to the current
       cell .

       @param seq_x, _y        Strings representing the entire x&y sequences 
       @param curr_matrix_type String M/X/or Y, representing the current matrix
       @param curr_row         int representing the index of the current row
       @param curr_col         int representing the index of the current col
       @return tuple< str<right_x>, str<right_y>
    """
    curr_x = "_" if curr_matrix_type == "Y" else seq_x[curr_col - 1] # -1 for
    curr_y = "_" if curr_matrix_type == "X" else seq_y[curr_row - 1] # extra row/col
    return (curr_x, curr_y)

def write_output(output_file, alignment_list, max_score, verbose):
    """This function writes sequence alignments to file and optionally prints
       them to screen.

       @param output_file    The (dir/)file to which alignments are written
       @param alignment_list list< tuple< str<x_align1>, str<y_align1>,
                                          str<x_align2>, str<y_align2> >, ... >
       @param max_score      float< score of alignment(s) >
       @param verbose        Boolean, if alignments + score are printed
    """
    # Format output
    output      = "%.1f\n\n" % max_score
    output     += "\n\n".join("%s\n%s" % (x, y) for x, y in alignment_list)

    # Write to file and optionally print
    out_fhandle = open(output_file, "w")
    out_fhandle.write(output)
    out_fhandle.close()
    to_print = "Maximum score and corresponding (%i) alignments:\n%s" \
                % (len(alignment_list), output)
    if verbose: logger.warning(to_print) # print regardless of logger level
    else:       logger.info(to_print)
    return

#...............................................................................
#   Main

def main():
    # Parse the input file to obtain sequences, gap penalties, and match scores
    seq_x, seq_y, \
    bool_local, \
    gap_x, extend_gap_x, gap_y, extend_gap_y, \
    letters_x, lettrs_y, \
    match_matrix = parse_input(args.input_file)

    # Initialize the score matrices (M / Ix / Iy) and get best alignment(s)
    M, Ix, Iy, start_indices, score = make_score_matrices(seq_x, seq_y, 
                                                          gap_x, extend_gap_x,
                                                          gap_y, extend_gap_y,
                                                          bool_local, 
                                                          match_matrix)
    logger.debug("Match matrix:\n%s" % match_matrix)
    logger.debug("Matrix M: \n%s"    % M)
    logger.debug("Matrix Ix:\n%s"    % Ix)
    logger.debug("Matrix Iy:\n%s"    % Iy)

    # Add alignments from each (equally scoring) start index to a list
    matrix_dict    = {"M" : M, "X" : Ix, "Y": Iy }
    alignment_list = []
    for start_row, start_col in start_indices:
         curr_alignments = rec_traceback(seq_x, seq_y, start_row, start_col,
                                         "M", matrix_dict, bool_local)
         alignment_list.extend(curr_alignments)
    
    # Write the outputs to file and optionally print to screen
    write_output(args.output_file, alignment_list, score, 
                 verbose=args.print_alignments)
    return

#..............................................................................
#   Unit tests
def test_all():
    test_compare_scores()
    test_get_next_matrix_indices()
    test_get_curr_letters()

def test_compare_scores():
    """Tests various inputs to the compare_scores() function
    """
    logger.warning("testing: compare_scores() ...")

    test_1   = {"A" : -np.inf, "B" : -np.inf}
    answer_1 = compare_scores(test_1)
    assert answer_1 == (-np.inf, "NA")

    test_2   = {"A" : -2.5, "B" : -2.6}
    answer_2 = compare_scores(test_2)
    assert (answer_2[1] == "A") and (abs(answer_2[0] + 2.5) < EPSILON)

    test_3   = {"A" : -2.5, "B" : -2.6, "NA" : 0}
    answer_3 = compare_scores(test_3)
    assert answer_3 == (0, "NA")

    test_4   = {"A" : -2.03, "B" : -2.03, "C" : -np.inf, "D" : -np.inf}
    answer_4 = compare_scores(test_4)
    assert (answer_4[1] in ("AB", "BA")) and (abs(answer_4[0] + 2.03) < EPSILON)

    test_5   = {"A" : 0, "B" : -2.6, "NA" : 0}
    answer_5 = compare_scores(test_5)
    assert answer_5 == (0, "NA")

    logger.warning("all tests passed for compare_scores()")
    return

def test_get_next_matrix_indices():
    logger.warning("testing get_next_matrix_indices() ...")

    assert get_next_matrix_indices("M", 1, 1) == (0, 0)
    assert get_next_matrix_indices("X", 1, 1) == (1, 0)
    assert get_next_matrix_indices("Y", 1, 1) == (0, 1)

    logger.warning("all tests passed for get_next_matrix_indices()")

def test_get_curr_letters():
    logger.warning("testing get_curr_letters() ...")
    
    test1 = get_curr_letters("abcd", "ABCD", "M", 3, 3)
    assert test1 == ("c", "C")

    test2 = get_curr_letters("abcd", "ABCD", "X", 3, 3)
    assert test2 == ("c", "_")

    test3 = get_curr_letters("abcd", "ABCD", "Y", 3, 3)
    assert test3 == ("_", "C")

    logger.warning("all tests passed for get_curr_letters()")

#...............................................................................
#   Flow determinant
if __name__ == "__main__":
    args = prsr.parse_args()
    logger = getLogger(args.logger_level)
    if args.tests_only: test_all()
    else: main()

