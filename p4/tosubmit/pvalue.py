#!/usr/bin/env python
info = """This program generates a bootstrap p-value for the comparison of two
          proteins. The p-value is based on the similarity of the sets of 
          compounds known to bind the two proteins, versus sets of randomly 
          chosen ligands. 
       """

__date__   = "2013-11-23"
__author__ = "ccwilliams"

import argparse
import datetime as dt
from chemoUtils import get_drugs_to_featsets, map_drugs_and_targets, \
                       make_tanimoto_array, get_drug_indices, \
                       get_tanimoto_summary, get_p_bootstrap, logger
#...............................................................................
#   Global variables

DEFAULT_N = 100 # default value for number of iterations in the bootstrap calc
TANCUTOFF = 0.5 # The cutoff used in determining if a Tanimoto score contributes
                # to the Tanimoto summary score for two proteins

#...............................................................................
#   Define arguments

prsr = argparse.ArgumentParser(description=info)
prsr.add_argument("drugs", type=str,
                  help="The .csv (path/)file containing drug IDs, common names,"
                       " and space-delimited drug feature numbers.")
prsr.add_argument("targets", type=str,
                  help="The .csv (path/)file mapping drug IDs to their target "
                       "protein IDs and protein common names.")
prsr.add_argument("proteinA", type=str,
                  help="Protein ID string for the first protein of interest")
prsr.add_argument("proteinB", type=str,
                  help="Protein ID string for the second protein of interest")
prsr.add_argument("-n", type=int, default=DEFAULT_N,
                  help="Specifiy the number of iterations for calculating the "
                       "bootstrap p-value. Default: %i" % DEFAULT_N)
prsr.add_argument("-l", "--logger_level", type=int,
                  default=30, choices=[10,20,30],
                  help="Specify the detail of print/logging messages. 10, 20,"                       
                       " 30 correspond to debug, info, and warning.")

#...............................................................................
#   Main

def main():
    start = dt.datetime.now()
    # First parse drugs and target files 
    sorted_drugs, \
    sorted_feats      = get_drugs_to_featsets(args.drugs, as_lists=True)
    targets_to_drugs  = map_drugs_and_targets(args.targets, key_on="targets")
   
    # Compute all pair-wise Tanimoto scores in preparation for boot-strapping
    tanimoto_array = make_tanimoto_array(sorted_feats)

    # Fetch the Tanimoto summary score for the Proteins A & B
    try:
        drugsA         = list( targets_to_drugs[args.proteinA] )
        drugsB         = list( targets_to_drugs[args.proteinB] )
    except:
        logger.warning("One or more protein names were invalid, exiting")
        return

    idxs_A, idxs_B = get_drug_indices(sorted_drugs, drugsA, drugsB)
    logger.debug("Drugs for \n\tprotein %s: %s\n\tprotein %s: %s" % \
                 (args.proteinA, drugsA, args.proteinB, drugsB))
    
    Tsummary = get_tanimoto_summary(tanimoto_array, idxs_A, idxs_B, TANCUTOFF)
    p_boot   = get_p_bootstrap(tanimoto_array, len(idxs_A), len(idxs_B), 
                               Tsummary, args.n, TANCUTOFF)

    # Print p-value to standard out
    logger.info("%i iteration bootstrap p-value for Tanimoto summary score of "
                "%.6f between proteins %s & %s:" % \
                (args.n, Tsummary, args.proteinA, args.proteinB) )
    logger.warning(p_boot)

    stop   = dt.datetime.now()
    logger.info("Program done in %.1f s." % ((stop-start).total_seconds()) )
    return

#...............................................................................
#   Program flow
if __name__ == "__main__":
    args   = prsr.parse_args()
    logger.setLevel(args.logger_level)
    main()
