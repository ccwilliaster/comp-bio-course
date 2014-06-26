#!/usr/bin/env python
info="""This is an implementation of the K-nearest-neighbors supervised ML alg
        orithm. Given positive and negative example/training features, the cla
        ss of an uncharacterized example is predicted based on the classes of 
        its k nearest neighbors, where 'nearest' is the smallest Euclidean 
        distance in feature space. In this implementation, the accuracy of 
        classifications is based on n-fold cross validation with the following
        pseudo code:
            1. Divide training set into n equally sized groups
               (+/- class ratios reflect those in the entire training set)
            2. Using 1 group as a test set and n-1 groups as the training set,
               make classifications
            3. Repeat 2 for each group in n
            4. Use TP/FP/TN/FN outcomes for all test items to determine 
               sensitivity/specificity/accuracy.

        The positive/negative input files should contain tab-delimited expressi
        on values, where each row is a gene and each column is a patient. 
        Accuracy assesement metrics for the classification are written to 
        stdout and to an output file, for example:
            k: 10
            p: 0.75
            accuracy: 0.68
            sensitivity: 0.85
            specificity: 0.90

        See arguments for more specifics on input/output.
     """

__author__ = "ccwilliams"
__date__   = "2013-10-20"

import sys
import argparse
import Queue
import logging
import random
import numpy as np

#...............................................................................
#   Global vars
OUTFILE     = "knn.out" # name of the output file
NFOLDS      = 4         # number of folds for cross-validation
POS_CLASS   = 1         # value for global positive classification
NEG_CLASS   = 0         # value for global negative classification
random.seed()

#...............................................................................
#   Define arguments
prsr = argparse.ArgumentParser(description=info)

prsr.add_argument("file_train_positive",
                  type=str,
                  help="The input (directory/)file which contains training "
                       "data for the items to be classified as positive. In"
                       "put file should contain tab-delimited values where "
                       "each row represents a ")
prsr.add_argument("file_train_negative",
                  type=str,
                  help="The input (directory/)file which contains training "
                       "data for the items to be classified as negative.")
prsr.add_argument("k",
                  type=int,
                  help="The number of 'neighbors' used to classify a given "
                       "item. Must be >= number items / NFOLDS.")
prsr.add_argument("p",
                  type=float,
                  help="The minimum fraction of 'neighbors' necessary to "
                       "classify an item as positive.")
prsr.add_argument("-cv", "--folds",
                  type=int, default=NFOLDS, 
                  help="Specify the number of folds used in n-fold cross-valid"
                       "ation for assessing classifier accuracy. Default: %i" 
                       % NFOLDS)
prsr.add_argument("-o", "--outfile",
                  type=str,
                  default=OUTFILE,
                  help="Specify an output (directory/)filename. Output contains"
                       " an accuracy assessment with sensitivity/specificity/ac"
                       "curacy metrics, as well as the k & p used.Default: %s"
                       % OUTFILE)
prsr.add_argument("-l", "--logger_level",
                  default="warning", choices=["debug", "info", "warning"],
                  help="Specify the detail of print/logging messages.")
prsr.add_argument("-t", "--tests_only",
                  action="store_true",
                  help="If this option is specified, all input is ignored "
                       "and only unit tests are run.")

#...............................................................................
#   Classes 

class Item(object):
    """A simple container class representing Items for classification. 
       Attributes include a classification class and a feature array. 
       Methods for determining the Euclidean distance to another 
       Item, for finding a specified number of Items that have the 
       smallest Euclidean distance to the self item, and for predicting
       the class of self are supported.
    """
    def __init__(self, features, classification, column_id):
        self.features       = np.array(features)
        self.classification = classification
        self.column_id      = column_id
    
    def get_euclid_distance(self, other):
        """This function returns the euclidean distance between self
           and other, which must also be an Item.
           
           @param other_item
           @return float< Euclidean distance >
        """
        assert isinstance(other, Item)
        return np.linalg.norm(self.features - other.features) 

    def get_KNN(self, k, all_neighbors):
        """Given a list of Items/neighbors, this method returns a sub-list of k
           Items/neighbors which have the smallest Euclidean distance to self.
           In the case that more than k items have equally good distances, k 
           Items are randomly chosen.

           @param k         int for the number of neighbors to find
           @param all_neighbors list< Items >
           @return list< Items >
        """
        # Cannot find more nearest neighbors than neighbors
        assert k <= len(all_neighbors)

        # We will construct a pritority queue based on 1/distance. If the queue
        # becomes larger than k, we dequeue the Item with largest distance
        KNN = Queue.PriorityQueue()
        for neighbor in all_neighbors:
            distance   = self.get_euclid_distance(neighbor)
            q_distance = 1/distance if distance != 0 else np.inf
            KNN.put( (q_distance, neighbor) ) # priority based on 0-idx value
            
            if KNN.qsize() > k:
                KNN.get() # = pop()

        return [ tup[1] for tup in KNN.queue ]

    def predict_class(self, k, p, all_neighbors):
        """This function predicts the class to which self belongs, based
           on k, p, and all_neighbors

           @param k             Number of neighbors to consider
           @param p             float, the fraction of neighbors that must
                                be classified as POS_CLASS for self to be
                                classified as POS_CLASS, else NEG_CLASS.
           @param all_neighbors list< Item >, a list of all neighbors
        """
        num_positive = 0
        KNN = self.get_KNN(k, all_neighbors)
        for nearest_neighbor in KNN:
            neighbor_class = nearest_neighbor.classification
            num_positive += 1 if neighbor_class == POS_CLASS else 0
        
        return POS_CLASS if num_positive >= (p*k) else NEG_CLASS
 
#...............................................................................
#   Helper functions 

def get_logger(logger_level):
    """Returns a console-only (i.e. no writing) logger with level set to 
       logger_level
       @param logger_level string, "debug"/"info"/"warning"
    """
    levels = {"debug"   : logging.DEBUG,
              "info"    : logging.INFO,
              "warning" : logging.WARN}
    logger    = logging.getLogger("knn.py")
    console   = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(levels[logger_level])
    return logger

def parse_input_file(filename, classification):
    """This function parses an input file in which rows are feature values and 
       columns represent an item to be classified.
        e.g. a column might be a person or a condition, 
             and rows might be gene expression values
       It returns a list of Items with the specified classification and the 
       feature vectors from the input file.
             
       @param filename  A string representing the (directory/)input file to
                        be parsed
       @return  list< all input Items >
    """
    # First we will gather a list of lists representing rows, then transpose 
    # this for a list of columns so each item is paired with its features
    rows = []
    filehandle = open(filename, 'r')
    for line in filehandle.readlines():
        rows.append( [ float(val) for val in line.strip('\n').split('\t') ] )
    filehandle.close()

    cols = zip(*rows)
    logger.info("Input file %s (class=%s) had %i rows, %i cols" % \
            (filename, str(classification), len(rows), len(cols)) )
    return make_Items(cols, classification)
      
def make_Items(list_of_features, classification):
    """This function returns a list of Item objects, one for each of the feature
       lists in list_of_features
       
       @param list_of_features  A list of lists of features. Each list of 
                                features will become an Item
       @return list< Items >
    """
    logger.debug("Constructing %i Items of class '%s'" % \
                 (len(list_of_features), str(classification)))
    return [ Item(features, 
                  classification, 
                  list_of_features.index(features)+1) \
             for features in list_of_features ]

def get_n_groups(n, list_of_Items):
    """This function randomizes and then divides a list of Items into n groups,
       and returns a list of n lists of items

       @param n             int specifying the number of groups of Items desired
       @param list_of_Items list< Item >
       @return list< n x list< Item > >
    """
    total_items = len(list_of_Items)
    group_size  = int( np.ceil( total_items / float(n) ) )
    chunks      = range(0, total_items, group_size)
    random.shuffle(list_of_Items) # note: this shuffles the list ITSELF
    
    assert group_size >= 1
    groups      = [ list_of_Items[i:i+group_size] for i in chunks ]
    group_sizes = ", ".join([str(len(subgroup)) for subgroup in groups])

    logger.debug("List of %i Items divided into %i groups" % (total_items, n))
    logger.debug("Group sizes = %s" % group_sizes)
    return groups


def classify_Items(k, p, test_Items, training_Items):
    """This function classifies all Items in test_Items, based on the training
       set of Items in training_Items. Classifications are based on p and k,
       and a dictionary of the number of TP/TN/FP/FN outcomes is returned.

       @param k              Number of neighbors to base classification on
       @param p              Fraction of neighbors that must be 
       @param test_Items
       @param training_Items
       @return dict< outcome : count >
    """
    # For debugging
    logger.debug("k=%i, p=%.2f, #testItems=%i, #trainItems=%i" % \
                 (k, p, len(test_Items), len(training_Items)))

    outcomes = { type : [] for type in ["TP", "TN", "FP", "FN" ]}
    for test_Item in test_Items:
        predicted_class = test_Item.predict_class(k, p, training_Items)
        actual_class    = test_Item.classification
        
        logger.debug("Actual: %s, Predicted: %s" % (str(actual_class), 
                                                    str(predicted_class)))
        if actual_class == predicted_class:
            if actual_class == POS_CLASS: outcomes["TP"].append(test_Item)
            else:                         outcomes["TN"].append(test_Item)
        else:
            if predicted_class == POS_CLASS: outcomes["FP"].append(test_Item)
            else:                            outcomes["FN"].append(test_Item)

    summary = ["%s:%i" % (stat,len(items)) for stat, items in outcomes.items()]
    summary = ", ".join(summary)
    logger.info("Current fold outcomes: %s" % summary)
    logger.info("\tTest item IDs: %s" % \
                ",".join([str(item.column_id) for item in test_Items]))
    return outcomes

def update_outcomes(all_outcomes, curr_outcomes):
    """This function updates the outcome dictionary, which counts the number of
       TP/FP/TN/FN: dict< str(stat) : int(count) > 
       If all_outcomes is empty, it returns curr_outcomes.

       @param all_outcomes  dict containing counts for all previous folds                     
       @param curr_outcomes dict containing counts for the current fold
       @return dict of same format with summed values
    """
    if len(all_outcomes) == 0: 
        return curr_outcomes

    return { stat : all_outcomes[stat] + count \
             for stat, count in curr_outcomes.items() }

def get_sensitivity(TP, FN):
    """Given the number of true positives (TP) and false negatives (FN), this 
       function returns the sensitivity or true positive rate of a model.
    """
    return float(TP) / (TP + FN) # float to prevent rounding 

def get_specificity(TN, FP):
    """Given the number of false positives (FP) and true negatives (TN), this
       function returns the specificity of a model.
    """
    return float(TN) / (TN + FP) # float to prevent rounding

def get_accuracy(TP, TN, N):
    """Given the number of accurate predictions (TP and TN) and the total 
       number of predictions, this function returns the accuracy of a model.
    """
    return (TP+TN) / float(N) # float to prevent rounding

def return_results(outfile, k, p, sensitivity, specificity, accuracy):
    """This function writes statistics for how well KNN performed after all 
       CV folds. It writes to stdout and to the specified input file.
       @param k, p  script input k and p args
       @param sensitivity, specificity, accuracy 
                    float metrics for performance
    """
    output = "k: %i\np: %.2f\naccuracy: %.2f\nsensitivity: %.2f\nspecificity:" \
             " %.2f" % (k, p, accuracy, sensitivity, specificity)

    logger.info("Writing output to file: %s" % outfile)
    filehandle = open(outfile, "w")
    filehandle.write(output)
    filehandle.close()
    logger.warning(output)
    return

def more_info(outcomes):
    """This function logs more information about the fate of each object 
       classified in the classification 
       
       @param outcomes the outcome dictionary from KNN
    """
    # Output the number of outcomes for each TP/TN/FP/FN stat
    stats = ["%s:%i" % (stat,len(items)) for stat, items in outcomes.items()]
    stats = ", ".join(stats)
    logger.info("Final outcomes: %s" % stats)

    # Output more detailed information for how each item was classified
    output = ""
    for outcome_type, item_list in outcomes.items():
        output += "\n%s:\n\t" % outcome_type
        output += ",".join([str(item.column_id) for item in item_list])

    logger.info("Final reults:\n%s" % output)
    return

#...............................................................................
#   Main
def main():
    # Load positive and negative training sets from input files, 
    # these are lists of Item objects with feature vectors and a class
    all_positive = parse_input_file(args.file_train_positive, POS_CLASS) 
    all_negative = parse_input_file(args.file_train_negative, NEG_CLASS)

    # Divide into NFOLDS groups for NFOLDS cross-validation
    groups_positive = get_n_groups(args.folds, all_positive)
    groups_negative = get_n_groups(args.folds, all_negative)

    # Perform NFOLDS cross-validation, i.e., predict the class of 1 group based
    # on KNN 'voting' by NFOLDS-1 other groups. Repeat for each group.
    all_outcomes = {}
    for index in range(args.folds):
        test_Items     = groups_positive[index] + groups_negative[index] 
        training_sets  = [ groups_positive[i] + groups_negative[i] \
                           for i in range(args.folds) if i != index ]
        training_Items = [item for sublist in training_sets for item in sublist]

        # Get a dictionary of TP/TN/FP/FN frequencies for this fold
        curr_outcomes = \
                classify_Items(args.k, args.p, test_Items, training_Items)
        # Add to total
        all_outcomes = update_outcomes(all_outcomes, curr_outcomes)

    # Get performance stats
    n = sum( [len(outcome_list) for outcome_list in all_outcomes.values()])
    sensitivity = get_sensitivity( len(all_outcomes["TP"]), 
                                   len(all_outcomes["FN"]) )
    specificity = get_specificity( len(all_outcomes["TN"]), 
                                   len(all_outcomes["FP"]) )
    accuracy    = get_accuracy( len(all_outcomes["TP"]), 
                                len(all_outcomes["TN"]), 
                                n)

    # Return results to std out and file
    return_results(args.outfile, 
                   args.k, args.p, sensitivity, specificity, accuracy)

    more_info(all_outcomes)

#...............................................................................
#   Unit tests
def test_all():
    logger.warning("Running all unit tests ...")
    test_get_euclid_distance()
    test_get_KNN()
    test_get_n_groups()
    test_predict_class() 
    test_classify_Items()
    logger.warning("All tests passed, exiting.")

def make_test_Items():
    P0 = Item([1,1,1], POS_CLASS,0)
    P1 = Item([1.1,1.2,1.1], POS_CLASS,1)
    P2 = Item([2,2,2], POS_CLASS,2)
    P3 = Item([3,3,3], POS_CLASS,3)
    N4 = Item([20,20,20], NEG_CLASS,4)
    return P0, P1, P2, P3, N4

def test_get_n_groups():
    logger.warning("Testing get_n_groups() ...")

    l1 = [1,1,1,1,1]
    l2 = [1,1,1,1,1,1,1,1]
    l3 = [1,1,1,1,1,1,1,1,1,1]

    assert get_n_groups(5, l1) == [[1] for i in range(len(l1))]
    assert get_n_groups(4, l2) == [[1,1] for i in range(len(l2)/2)]
    assert get_n_groups(3, l2) == [[1,1,1],[1,1,1],[1,1]]
    assert get_n_groups(2, l3) == [[1,1,1,1,1] for i in range(len(l3)/5) ]

    logger.warning("Tests passed.")

def test_get_euclid_distance():
    logger.warning("Testing Item.get_euclid_distance() ...")

    P0, P1, P2, P3, N4 = make_test_Items()
    d0, d1, d2, d3, d4 = P0.get_euclid_distance(P0),\
                         P0.get_euclid_distance(P1),\
                         P0.get_euclid_distance(P2),\
                         P0.get_euclid_distance(P3),\
                         P0.get_euclid_distance(N4)
    assert d0 == 0
    assert abs(d2 - np.sqrt(3))      < 0.00001
    assert abs(d3 - (2*np.sqrt(3)))  < 0.00001
    assert abs(d4 - (np.sqrt(1083))) < 0.00001
    assert d0 < d1 < d2 < d3 < d4

    logger.warning("Tests passed.")

def test_get_KNN():
    logger.warning("Testing Item.get_KNN() ...")
    
    P0, P1, P2, P3, N4 = make_test_Items()
    KNN1 = P0.get_KNN(1, [P1,P2,P3,N4])
    KNN2 = P0.get_KNN(2, [P1,P2,P3,N4])
    KNN3 = P0.get_KNN(3, [P1,P2,P3,N4])
    KNN4 = P0.get_KNN(4, [P1,P2,P3,N4])

    assert (len(KNN1) == 1) and (set(KNN1) == set([P1]))
    assert (len(KNN2) == 2) and (set(KNN2) == set([P1,P2]))
    assert (len(KNN3) == 3) and (set(KNN3) == set([P1,P2,P3]))
    assert (len(KNN4) == 4) and (set(KNN4) == set([P1,P2,P3,N4]))

    logger.warning("Tests passed.")

def test_predict_class():
    logger.warning("Testing Item.predict_class() ...")

    P0, P1, P2, P3, N4 = make_test_Items()
    
    p1 = P0.predict_class(1, 1, [P1,P2,P3,N4])
    p2 = P0.predict_class(2, 1, [P1,P2,P3,N4])
    p3 = P0.predict_class(3, 1, [P1,P2,P3,N4])
    p4 = P0.predict_class(4, 1, [P1,P2,P3,N4])
    p5 = P0.predict_class(4, 0.8, [P1,P2,P3,N4])
    p6 = P0.predict_class(4, 0.75, [P1,P2,P3,N4])
    p7 = P0.predict_class(4, 0.70, [P1,P2,P3,N4])
    p8 = P0.predict_class(1, 1, [N4])
    p9 = P0.predict_class(1, 0, [N4])

    assert p1 == POS_CLASS
    assert p2 == POS_CLASS
    assert p3 == POS_CLASS
    assert p4 == NEG_CLASS
    assert p5 == NEG_CLASS
    assert p6 == POS_CLASS
    assert p7 == POS_CLASS
    assert p8 == NEG_CLASS
    assert p9 == POS_CLASS

    logger.warning("Tests passed.")

def test_classify_Items():
    logger.warning("Testing classify_Items() ...")

    P0, P1, P2, P3, N4 = make_test_Items()
    d1 = classify_Items(1, 1, [P0], [P1,P2,P3,N4]) # TP
    d2 = classify_Items(4, 1, [P0], [P1,P2,P3,N4]) # FN
    d3 = classify_Items(4, 0, [P0], [P1,P2,P3,N4]) # TP
    d4 = classify_Items(4, 1, [N4], [P0,P1,P2,P3]) # FP
    d5 = classify_Items(4, 0, [N4], [P0,P1,P2,P3]) # FP
    d6 = classify_Items(1, 1, [N4], [N4])          # TN
    d7 = classify_Items(1, 0, [N4], [N4])          # FP
    d8 = classify_Items(1, 0, [N4], [P0,P1,P2,P3]) # FP
    d9 = classify_Items(4, 1, [N4], [P0,P1,P2,P3]) # FP

    assert len(d1["TP"]) == len(d3["TP"]) == 1
    assert len(d2["FN"]) == 1
    assert len(d4["FP"]) == len(d5["FP"]) == len(d7["FP"]) == len(d8["FP"]) == len(d9["FP"]) ==  1
    assert len(d6["TN"]) == 1

    logger.warning("Tests passed.")

#...............................................................................
#   Flow determinant
if __name__ == "__main__":
    args   = prsr.parse_args()
    logger = get_logger(args.logger_level)
    if args.tests_only: test_all()
    else: main()


