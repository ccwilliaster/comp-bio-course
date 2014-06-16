#!/usr/bin/env python
info = """This program generates a protein network based on the similarity of 
          the ligand sets which bind the proteins. For each protein pair, a
          Tanimoto summary score is generated describing the similarity of the
          ligand sets which bind the two proteins, and the significance of this
          is evaluated using a bootstrap p-value for randomly chosen ligand 
          sets. If the p-values satisfy a cutoff value, the proteins are 
          connected by a node.
       """

__author__ = "ccwilliams" 
__date__   = "2013-11-20"

import argparse
import datetime as dt
from chemoUtils import map_drugs_and_targets, get_drugs_to_featsets, \
                       dict_from_csv, make_tanimoto_array, logger, \
                       populate_node_pvalues, get_sorted_filtered_nodes, \
                       write_network_data

#...............................................................................
#   Global variables

PVAL_CUTOFF = 0.05
DEFAULT_N   = 100 # default value for number of iterations in the bootstrap calc
TANCUTOFF   = 0.5 # The cutoff used in determining if a Tanimoto score 
                  # contributes to the Tanimoto summary score for two proteins

FILE_SIF        = "network.sif"         # output .sif file for network edges
FILE_NODE_ATTR  = "name.nodeAttr"       # output .nodeAttr filefor ID - name
FILE_INDICATION = "indication.nodeAttr" # output .nodeAttr file for ID - indication

#...............................................................................
#   Parameters

prsr = argparse.ArgumentParser(description=info)
prsr.add_argument("drugs", type=str,
                  help="The .csv (path/)file containing drug IDs, common names,"
                       " and space-delimited drug feature numbers.")
prsr.add_argument("targets", type=str,
                  help="The .csv (path/)file mapping drug IDs to their target "
                       "protein IDs and protein common names.")
prsr.add_argument("nodes", type=str,
                  help="The csv (path/)file containing swisprot ID, swisprot "
                       "name, and indications. All proteins in the file will "
                       "be considered for inclusion in network")
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
    
    # Read in data for drug-features, drug-target mappings, and network nodes
    targets_to_drugs  = map_drugs_and_targets(args.targets, key_on="targets")
    
    sorted_drugs, sorted_feats = \
        get_drugs_to_featsets(args.drugs, as_lists=True)

    targets_to_indications = \
        dict_from_csv(args.nodes, key_idx=0, val_idx=2, skip=1)
    
    target_ID_to_name = \
        dict_from_csv(args.nodes, key_idx=0, val_idx=1, skip=1)
    
    sorted_targets = sorted( target_ID_to_name.keys() )

    # Calculate all pair-wise drug Tanimoto scores once
    tanimoto_array = make_tanimoto_array(sorted_feats)

    # Now compute the bootstrap p-values for all potential protein target nodes 
    pvalue_array = populate_node_pvalues(sorted_targets, sorted_drugs, 
                                         targets_to_drugs, tanimoto_array, 
                                         args.n, TANCUTOFF)
    
    # Filter nodes by p-value cutoff
    filt_nodes, filt_edge_pairs = \
        get_sorted_filtered_nodes(sorted_targets, pvalue_array, PVAL_CUTOFF)

    # Write data for filtered nodes
    write_network_data(FILE_SIF, FILE_NODE_ATTR, FILE_INDICATION, filt_nodes,
                       filt_edge_pairs, target_ID_to_name, 
                       targets_to_indications)

    stop  = dt.datetime.now()
    logger.info("Program done in %.1f s" % ((stop-start).total_seconds()) )
    return


#...............................................................................
#   
if __name__ == "__main__":
    args   = prsr.parse_args()
    logger.setLevel(args.logger_level)
    main()
