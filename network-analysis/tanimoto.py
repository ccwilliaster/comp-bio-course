#!/usr/bin/env python
info = """This program reads in a table of drugs with associated chemical 
          feature vectors as well as a table mapping drugs to protein targets,
          and computes Tanimoto similarity scores between all drug pairs. It
          writes these scores to file and also writes a flag indicating if the
          drugs have a common protein target (1) or not (0).
       """

__date__   = "2013-11-23"
__author__ = "ccwilliams"

import logging
import argparse
import datetime as dt
from chemoUtils import get_drugs_to_featsets, map_drugs_and_targets, \
                       dict_to_sorted_lists, make_tanimoto_array, \
                       write_all_tanimotos, logger

#...............................................................................
#   Define arguments

prsr = argparse.ArgumentParser(description=info)
prsr.add_argument("drugs", type=str,
                  help="The .csv (path/)file containing drug IDs, common names,"
                       " and space-delimited drug feature numbers.")
prsr.add_argument("targets", type=str,
                  help="The .csv (path/)file mapping drug IDs to their target "
                       "protein IDs and protein common names.")
prsr.add_argument("outfile", type=str,
                  help="The entire (path/)filename for output. Data will be "
                       "written to this exact filename.")
prsr.add_argument("-l", "--logger_level", type=int,
                  default=30, choices=[10,20,30],
                  help="Specify the detail of print/logging messages. 10, 20,"
                       " 30 correspond to debug, info, and warning.")

#...............................................................................
#   Main
def main():
    start = dt.datetime.now()
    # First, parse input files to map drugs to their features and targets
    sorted_drugs, \
    sorted_feats      = get_drugs_to_featsets(args.drugs, as_lists=True)
    drugs_to_targets  = map_drugs_and_targets(args.targets, key_on="drugs")

    # Compute pair-wise Tanimoto similarity scores between all drugs
    tanimoto_array = make_tanimoto_array(sorted_feats)

    # Now write the data to file
    write_all_tanimotos(args.outfile, sorted_drugs, 
                        tanimoto_array, drugs_to_targets)

    stop = dt.datetime.now()
    logger.info("Program done in %.1f s." % ((stop-start).total_seconds()) )
    return

#...............................................................................
#   Program flow
if __name__ == "__main__":
    args   = prsr.parse_args()
    logger.setLevel(args.logger_level)
    main()
