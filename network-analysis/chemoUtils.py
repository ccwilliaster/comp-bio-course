#!/usr/bin/env python
"""This file contains utility and functions useful for chemoinformatics.
   Specifically, it contains functions useful for computing Tanimoto similarity
   scores for drug compounds, bootstrap p-values for inferring protein 
   similarity based on the similarity of the drugs which bind the proteins, as
   well as several read/write helper functions. All functions are used within
   the tanimoto.py, pvalue.py, and networkgen.py programs.

   The main Tanimoto score calculations are setup such that all pair-wise drug
   Tanimoto scores are computed a single time, then referenced again later
   several times. This makes it slower for tanimoto.py and pvalue.py because 
   of the initial overhead, but faster for bootstrap calculations and network 
   generation.

   The file itself is organized by the following headers:
        - Readers
        - Writers
        - Tanimoto score functions
        - Bootstrap p-value functions
        - misc
"""

__author__ = "ccwilliams"
__date__   = "2013-11-20"

import sys
import csv
import logging
import numpy as np
import datetime as dt

np.random.seed() # set for testing

#...............................................................................
#   Readers

def get_drugs_to_featsets(drugfile, as_lists=False):
    """This function reads a .csv file containing  drug ID, generic name, and 
       space-delimited features characterizing the molecule, and returns a dict
       keyed on drug ID with sets of features as values.

       @param drugfile  str, (path/)filename containing drug information
       @param as_lists  Boolean, whether drugs and feature sets should be
                        returned as sorted lists/arrays, instead of a dict
       @return dict< drug ID : set< int features > >
    """
    start   = dt.datetime.now()
    f_lines = list( csv.reader( open(drugfile, "r") ) )
    assert f_lines[0][0] == "db_id" # enforce expected format

    getfeats = lambda feat_str: set([int(feat) for feat in feat_str.split(" ")])
    drugs    = { ID : getfeats(feat_str) for ID, name, feat_str in f_lines[1:] }
        
    stop =  dt.datetime.now()
    logger.info("%i drugs parsed from file: %s" % (len(drugs), drugfile))
    logger.debug("Drugs parsed in %ius" % ((stop-start).microseconds))
    
    if as_lists:
        sorted_drugs, sorted_features = \
            dict_to_sorted_lists(drugs, vals_as_array=True)
        return sorted_drugs, sorted_features
    return drugs

def dict_from_csv(infile, key_idx, val_idx, skip=0):
    """This function maps an arbitrary column of a csv file to another column,
       skipping the specified number of lines first.
       @param infile    (directory/)file to be parsed
       @param key_idx   0-based index for csv column to be mapped as dict keys
       @param val_idx   0-based index for csv column to be mapped as dict values
       @param skip      number of lines to skip in file, for headers, etc.
       @return dict< csv key_idx : csv val_idx >
    """
    f_lines = list( csv.reader( open(infile, "r") ) )
    return { line[ key_idx ] : line[ val_idx ] for line in f_lines[skip:] }

def map_drugs_and_targets(targetfile, key_on="drugs"):
    """This function reads a .csv file containing drug ID, protein ID, and 
       protein name, and returns a dict keyed on either drug ID or target ID
       with sets of the other as values

       @param targetfile str, path(/)filename containing drug target information
       @param key_on     str, "drugs" or "targets", specifying dict keys
       @return dict< drug ID : set< str<target protein ID> > 
                OR
               dict< protein ID : set< str<drug ID> >
    """
    start   = dt.datetime.now()
    dct     = {}
    f_lines = list( csv.reader( open(targetfile, "r") ) )
    assert f_lines[0][0] == "db_id" # enforce expected format
    assert key_on in ("drugs", "targets")
    if key_on == "drugs":
        key, val = 0, 1 
    else:
        key, val = 1, 0

    for line in f_lines[1:]:
        curr_key, curr_val = line[key], line[val]

        if dct.has_key(curr_key):
            dct[curr_key].add(curr_val)
        else:
            dct[curr_key] = set( [curr_val] )

    stop =  dt.datetime.now()
    logger.info("Mapped %i %s to %i total %s from file: %s" % (len(dct),
                 key_on, len(f_lines)-1, 
                 "drugs" if key else "targets", targetfile))
    logger.debug("Drugs targets parsed in %ius" % ((stop-start).microseconds)) 
    return dct

def dict_to_sorted_lists(dct, keys_as_array=False, vals_as_array=False):
    """This function takes a dictionary and returns two lists (or numpy arrays)
       consisting of dict keys and values, each sorted by keys.
       @param dct           dict
       @param keys_as_array Bool, if keys should be returned as a numpy array
       @param vals_as_array Bool, if keys shoudl be returned as a numpy array
       @return list/array<keys>, list/array<values> 
    """
    keys = np.array( sorted(dct.keys()) ) if keys_as_array \
           else sorted(dct.keys())
    vals = np.array( [dct[key] for key in keys] ) if vals_as_array \
           else [dct[key] for key in keys]
    return keys, vals 

#...............................................................................
#   Writers

def write_all_tanimotos(outfile, drug_ID_array, tanimoto_array, drug_to_targets):
    """This function writes pair-wise Tanimoto scores for all drugs represented
       in the input arrays, and also indicates whether the drugs share any 
       protein targets. The function writes output in the order of the drug 
       arrays, so the desired order should be represented within the array.

       @param outfile
       @param drug_ID_array     It is assumed that the order of this array
                                matches the order of scores for tanimoto_array
       @param tanimoto_array    nxn numpy array containing Tanimoto scores,
                                where n = the length of drug_ID_array/# drugs
                                This will be parsed left to right across columns
                                and top to bottom for rows > current col #
       @param drug_to_targets   dict< drug ID : set(target IDs)
    """
    logger.info("Writing all pair-wise Tanimoto scores to file %s" % outfile)
    start   = dt.datetime.now() 
    fh_out  = open(outfile, "w")

    col_idx = 0
    for col_drug in drug_ID_array:
        for row_idx in range(col_idx + 1, len(drug_ID_array)): # idxs > curr col

            # Obtain Tanimoto scores and common_target flag for the two
            # drugs at the current row and column indices
            row_drug       = drug_ID_array[row_idx]
            tanimoto_score = tanimoto_array[row_idx, col_idx]
            try:
                common_target  = share_targets(drug_to_targets[col_drug],
                                               drug_to_targets[row_drug])
            except KeyError:
                logger.warning("Warning: error mapping drug %s or %s to " \
                               "targets, skipping." % (col_drug, row_drug))
                common_target = 0
            outline = "%s,%s,%.6f,%i\n" % \
                      (col_drug, row_drug, tanimoto_score, common_target)
            # Write info to file
            fh_out.write(outline) 

        col_idx += 1

    stop    = dt.datetime.now()
    return

def write_SIF(outfile, edges):
    """This function writes a network .sif file to the specified filename in
       the format: (no header) "swissprot_ID1 edge swisprot_ID2\n"

       @param outfile   The (directory/)file for the output SIF file
       @param edges     list< list< swisprot_ID1, swisprot_ID2> > defining
                        node pairs for edges. One line is written for each pair
    """
    fh_out = open(outfile, "w")
    # Write one line per node pair / edge
    for node_pair in edges:
        line = "%s edge %s\n" % (node_pair[0], node_pair[1])
        fh_out.write(line)
    logger.info("Data for %i edges written to: %s" % (len(edges), outfile))
    return

def write_node_attrs(name_out, indication_out, node_IDs, 
                     ID_to_name, ID_to_indication):
    """This function writes two node attribute files, one for mapping protein
       ID to protein name and one for mapping protein ID to its indication(s).

       @param name_out       name of file mapping protein ID to protein name
       @param indication_out name of file mapping protein ID to indication(s)
       @param node_IDs       list of node IDs for which attr data will be written
       @param ID_to_name     dict< protein ID : str< protein name > >
       @param ID_to_indication dict< protein ID : str< indication(s) > >
    """
    fh_name       = open(name_out, "w")
    fh_indication = open(indication_out, "w")

    # Write headers for the name and indication files
    fh_name.write("name\n")
    fh_indication.write("indication\n")

    # Now write node information to each file
    for node in node_IDs:
        fh_name.write("%s = %s\n" %       (node, ID_to_name[node]) )
        fh_indication.write("%s = %s\n" % (node, ID_to_indication[node]) )

    logger.info("node attribute data for %i nodes written to: %s, %s" % \
                (len(node_IDs), name_out, indication_out))
    return

def write_network_data(sif_out, name_out, indication_out, filt_nodes, 
                       filt_node_pairs, ID_to_name, ID_to_indication):
    """A wrapper function for performing all writing operations for networkgen
       See write_SIF() and write_node_attrs() for more info.
    """ 
    write_SIF(sif_out, filt_node_pairs)
    write_node_attrs(name_out, indication_out, filt_nodes, 
                     ID_to_name, ID_to_indication)
    return

#...............................................................................
#   Functions for determining Tanimoto scores

def tanimoto(features_1, features_2):
    """This function takes two sets of drug features and returns a Tanimoto
       score describing the feature similarity:
            Tscore = | f1 intersection f2 | / | f1 union f2 |

       @param features_1    set<int>, features of drug 1
       @param features_2    set<int>, features of drug 2
       @return float< Tanimoto score >
    """
    f = lambda s1, s2: len( s1.intersection(s2) ) / float(len( s1.union(s2) ))
    return f(features_1, features_2)

# The following is a vectorized version of tanimoto, useful for computing
# several Tanimoto values at once. 
# np.frompyfunc returns an array with dtype = object, 
# but np.vectorize returns float64. frompyfunc seems ~300000 us faster
#vect_tanimoto = np.vectorize(tanimoto, otypes=[np.float64])
vect_tanimoto = np.frompyfunc(tanimoto, 2, 1)

def make_tanimoto_array(ordered_feats):
    """This function creates and returns a nxn np.array consisting of all
       pair-wise Tanimoto scores between drugs in the drug_to_feat dictionary,
       where n is the number of drugs. Note that pair-wise scores are symmetric,
       and so is the resulting array.

       @param ordered_feats np.array< set< int< feature > > >. The order of the
                            array will be preserved along rows and cols of the
                            returned array
       @return nxn np.array of Tanimoto scores
    """
    start   = dt.datetime.now()
    # Compute all pair-wise Tanimoto scores, removing symmetry if specified
    tan_array = vect_tanimoto(ordered_feats, ordered_feats[:,None]) 

    stop =  dt.datetime.now()
    logger.debug("Pair-wise Tanimoto values computed for %i drugs" % \
                len(ordered_feats) )
    logger.debug("Tanimoto array made in %ius" % ((stop-start).microseconds))
    return tan_array

def get_tanimoto_summary(tan_array, idxs_A, idxs_B, cutoff=0.5):
    """This function returns the sum of all tanimoto scores above a specified
       cutoff in sub-array of the Tanimoto array specified by the indices in 
       idxs_A and idxs_B.
       @param tan_array nxn symmetric np.array of pairwise tanimoto scores
       @param idxs_A    Sequence of indices for cols of interest in tan_array
       @param idxs_B    Sequence of indices for rows of interest in tan_array
       @param cutoff    Cutoff for Tanimoto scores from the specified indices
                        to be included in Tsummary. Values <= this value are
                        filtered
       @return float< summary score >
    """
    assert cutoff > 0 # 0's in tan_array will be interpreted incorrectly
    
    # Create sub-array of tan_array, then sum scores higher than cutoff score
    subarray       = tan_array[ np.ix_(idxs_A, idxs_B) ]
    cutoff_mask    = subarray > cutoff
    summary_score  = np.sum( subarray[cutoff_mask] )
    
    return summary_score if summary_score else 0

#...............................................................................
#   For calculatinig p-values

def get_drug_indices(drugs, *query_IDs):
    """This function returns lists of indices for one or more lists of drug IDs
       representing the indices of those IDs within the drugs array. If any of
       the query IDs are not in drugs, this will raise a ValueError
       
       @param drugs         sequence of drug IDs
       @param *query_IDs    one or more sequences of drug IDs whose indices in 
                            drugs will be returned
    """
    return [ [ drugs.index(ID) for ID in ID_list ] for ID_list in query_IDs ]

def get_random_indices(tuple_n_idxs, possible_idxs, indpt=True):
    """This function returns a list of lists of indices sampled from 
       possible_idxs. The number of indices in each list is specified by a tuple
       of integers in tuple_n_idxs 

       @param tuple_n_idxs  tuple< n_idx1, n_idx2, ... > where n_idxi represents
                            the length of the list of idxs for list i
       @param possible_idxs sequence of numbers from which indices can be be
                            sampled without replacement.
       @return  tuple< n lists of random idxs >
    """
    assert type(tuple_n_idxs) in (type(()), type([]))
    # Choose random indices, and group into lists of specified sizes 
    try:
        rand_idxs = \
            [ list( np.random.choice(possible_idxs, n_idxs, replace=False) ) \
              for n_idxs in tuple_n_idxs ]
    except:
        logger.warning("get_random_indices(): Not enough values to sample from")
        raise

    return rand_idxs
     
def indices_from_targets(targets, drug_array, targets_to_drugs):
    """This is a wrapper function for returning drug ID indices for drugs 
       associated with one or more targets.

       @param targets          sequence of targets for which drug indices
                               will be returned
       @param drug_array       sequence of drug IDs for which indices will
                               will be returned
       @param targets_to_drugs dict< protein ID : set< drug ID > 
       @return 
    """
    # 
    list_of_drug_lists = [ list( targets_to_drugs[ID] ) for ID in targets]
    list_of_idx_lists  = get_drug_indices(drug_array, *list_of_drug_lists)
    return list_of_idx_lists

def get_p_bootstrap(tan_array, set_size_1, set_size_2, Tsummary, n, tan_cutoff):
    """This function determines a bootstrap p-value representing the liklihood
       of obtaining a Tanimoto summary score of Tsummary or better for two 
       proteins with drug sets of specified size. The number of iterations can
       be specified as can the Tanimoto score cutoff for inclusion in Tsummary
       values.

       @param tan_array     nxn array of all pair-wise Tanimoto scores
       @param set_size_1    Size of drug set of protein 1 from Tsummary calc
       @param set_size_2    Size of drug set of protein 2 from Tsummary calc
       @param Tsummary      Tsummary for which p-value is determined
       @param n             The number of bootstrap iterations
       @param tan_cutoff    Tanimoto scores below this will not be considered
                            when computing Tsummary
       @return float< p-bootstrap >
    """
    logger.debug("Computing %i iteration bootstrap p-value for array sizes " 
                 "%i, %i, and Tsummary=%.3f" % \
                 (n, set_size_1, set_size_2, Tsummary))

    start = dt.datetime.now()
    n_as_good = 0 # number of times Trandom >= Tsummary

    for i in range(n):
        # Generate random drug indices and compute Tanimoto summary values
        all_idxs = range( tan_array.shape[0] )
        idxs_1, idxs_2 = get_random_indices((set_size_1, set_size_2), all_idxs)
        Trandom = \
            get_tanimoto_summary(tan_array, idxs_1, idxs_2, tan_cutoff)      
        
        # If these are randomly as good as our test value, increment counter
        n_as_good += 1 if Trandom >= Tsummary else 0

    # p-value represents the probability of generating a value as good or 
    # better than the observed value by chance
    p_boot = n_as_good / float( n )

    stop =  dt.datetime.now()
    logger.debug("p-bootstrap done in %ius" % ((stop-start).microseconds))
    return p_boot

def populate_node_pvalues(targets, drug_array, targets_to_drugs, tan_array,
                          n, tan_cutoff):
    """This function initializes and populates a nxn target array where n is
       the number of targets in targets. Each cell in the array represents the
       bootstrap p-value for the Tanimoto scores for the drugs which target
       the targets in the current row and column. The array is symmetric and 
       thus only the lower half of the array is populated excluding the diagonal 
            (i.e., for col j, only rows i are populated where i > j)
    
       @param targets          sequence of protein targets
       @param drug_array       sorted array of drugs, indices match tan_array
       @param targets_to_drugs dict< protein target : set( drugs )
       @param tan_array        nxn symmetric array containing 
       @param n                number of iterations in bootstrap p-value 
                               computation
       @param tan_cutoff       Cutoff used in computing a Tsummary, Tanimoto
                               scores <= this value are filtered.
       @return nxn tril (lower-triangle-populated) numpy array of pvalues
    """
    start = dt.datetime.now()
    logger.debug("Computing %i pairwise p-values for protein similarity scores"\
                 % len(targets) )
    
    target_array   = np.zeros( (len(targets), len(targets)) ) 

    # Populate only the lower diagonal of the array
    for col_idx in range(len(targets)): # all column idxs
        col_target = targets[col_idx]
        
        for row_idx in range((col_idx + 1), len(targets)):
            # For each cell, compute Tsummary and then bootstrap p-value
            row_target = targets[row_idx]
            idxs_A, idxs_B = indices_from_targets([row_target, col_target], 
                                                  drug_array, targets_to_drugs)
            Tsummary = get_tanimoto_summary(tan_array, 
                                            idxs_A, idxs_B, tan_cutoff)
            p_boot   = get_p_bootstrap(tan_array, len(idxs_A), len(idxs_B),
                                       Tsummary, n, tan_cutoff)
            target_array[row_idx, col_idx] = p_boot

    stop  = dt.datetime.now()
    logger.debug("populate p-vals done in %ius" % ((stop-start).microseconds))

    return target_array

def get_sorted_filtered_nodes(targets, pval_array, pval_cutoff): 
    """This function takes an array of bootstrap p-values and returns a sorted
       list of nodes which meet a p-value cutoff and a sorted list of sorted 
       lists of the paired nodes (row/col) that met the cutoff in the array.

       @targets     Sequence of protein targets. It is assumed that this is sorted
                    and that its indices match those in the pval_array
       @pval_array  2D np.array with pvalues. Indices are assumed to correspond
                    to the same indices in the targets sequence, and only the 
                    lower triangle of the array is queried, excluding the diagonal
       @pval_cutoff p-values > this value are filtered
    """
    assert pval_cutoff > 0 # else array zeros are mis-interpretted 
    node_set          = set()
    sorted_node_pairs = []
    total_pvals, sig_pvals = 0, 0

    # Iterate over lower triangle of array and note significant nodes & edges
    for col_idx in range( len(targets) ):
        for row_idx in range((col_idx + 1), len(targets)):
            total_pvals += 1
            # float comparison okay bc boot strap is exact if n is multiple of 10
            if pval_array[row_idx, col_idx] <= pval_cutoff: 

                # This pvalue is significant, note nodes/edge
                col_target = targets[col_idx]
                row_target = targets[row_idx]
                # This preserves row/column sorting because targets is sorted
                sorted_node_pairs.append( [col_target, row_target ]  ) 
                node_set.add(col_target)
                node_set.add(row_target)
                sig_pvals += 1
    
    logger.info("%i of %i bootstrap p-values satisfied significance <= %.6f" % \
                (sig_pvals, total_pvals, pval_cutoff) )
    return sorted( list(node_set) ), sorted_node_pairs

#...............................................................................
#   Misc functions
def share_targets(targets1, targets2):
    """Given two sets of target proteins, this function returns a 1 if there are
       any shared targets and 0 if not.
       @param targets1  set< str<protein IDs> > for drug1
       @param targets2  set< str<protein IDs> > for drug2
       @return int< 0 or 1 >
    """
    common_target = len( targets1.intersection(targets2) ) > 0
    return 1 if common_target else 0

# A vectorized implementation
vect_share_targets = np.vectorize(share_targets, otypes=[np.float64])

#...............................................................................
#   For logging
def get_logger(logger_level, name):
    """Returns a console-only (i.e. no writing) logger with level set to
       logger_level
       @param logger_level string, "debug"/"info"/"warning"
    """
    levels = {"debug"   : logging.DEBUG,
              "info"    : logging.INFO,
              "warning" : logging.WARN}
    logger    = logging.getLogger(name)
    console   = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(levels[logger_level])
    return logger

# The level of this logger can be overwritten when imported into other programs
logger = get_logger("debug", "chemoUtils")
