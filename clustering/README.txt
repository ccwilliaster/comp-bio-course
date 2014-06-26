#...............................................................................
Name:  Chris Williams
SUNet: ccwillia
Discussed with: Piazza only

#...............................................................................
# K-NEAREST-NEIGHBORS

usage: knn.py [-h] [-cv FOLDS] [-o OUTFILE] [-l {debug,info,warning}] [-t]
              file_train_positive file_train_negative k p

This is an implementation of the K-nearest-neighbors supervised ML alg orithm.
Given positive and negative example/training features, the cla ss of an
uncharacterized example is predicted based on the classes of its k nearest
neighbors, where 'nearest' is the smallest Euclidean distance in feature
space. In this implementation, the accuracy of classifications is based on
n-fold cross validation with the following pseudocode: 

Divide POS and NEG training sets into NFOLD equally-sized groups
Pair each POS group with one NEG group
for each paired set:
    set paired set as TEST set
    set remaining NFOLD-1 paired sets as TRAIN set
    for each point in TEST set:
        find k nearest TRAIN set neighbors based on Euclidean distance to point
        if k*p or more neighbors are of class POS:
            Classify TEST point as POS
            if TEST point is of class POS:
                increment truePositive
            else:
                increment falsePositive
        else:
            Classify TEST point as NEG
            if TEST point is of class NEG:
                increment trueNegative
            else:
                increment falseNegative
Calculate accuracy 
Calculate sensitivity
Calculate specificity
  
The positive/negative input files should contain tab-delimited expressi on 
values, where each row is a gene and each column is a patient. Accuracy 
assesement metrics for the classification are written to stdout and to an output 
file, for example: 
    k: 10 
    p: 0.75 
    accuracy: 0.68 
    sensitivity: 0.85 
    specificity: 0.90

See arguments for more specifics on input/output:

positional arguments:
  file_train_positive   The input (directory/)file which contains training
                        data for the items to be classified as positive. Input
                        file should contain tab-delimited values where each
                        row represents a
  file_train_negative   The input (directory/)file which contains training
                        data for the items to be classified as negative.
  k                     The number of 'neighbors' used to classify a given
                        item. Must be >= number items / NFOLDS.
  p                     The minimum fraction of 'neighbors' necessary to
                        classify an item as positive.

optional arguments:
  -h, --help            show this help message and exit
  -cv FOLDS, --folds FOLDS
                        Specify the number of folds used in n-fold cross-
                        validation for assessing classifier accuracy. Default:
                        4
  -o OUTFILE, --outfile OUTFILE
                        Specify an output (directory/)filename. Output
                        contains an accuracy assessment with
                        sensitivity/specificity/accuracy metrics, as well as
                        the k & p used.Default: knn.out
  -l {debug,info,warning}, --logger_level {debug,info,warning}
                        Specify the detail of print/logging messages.
  -t, --tests_only      If this option is specified, all input is ignored and
                        only unit tests are run.

#...............................................................................
# K-MEANS

usage: kmeans.py [-h] [-o OUTFILE] [-l {debug,info,warning}] [-t]
                 k expression_data max_iterations [centroids]

This is an implementation of the k-means clustering unsupervised ML algorithm.
It attempts to cluster n data points into k clusters such that the within-
cluster sum of squares Euclidean distance from the cluster mean is minimized.
The general procedure it uses is as follows: 

Read in feature vectors for n data points
if positions for clusters are provided:
    Initialize k clusters at specified positions in feature space
else:
    Initialize k clusters at positions of k random data points
while num_iterations < max_iterations:
    for each data point:
        Assign point to cluster with minimum Euclidean distance to point
    for each cluster center:
        Update position to mean position of points assigned to cluster
    if no points changed cluster assignemnt:
        break
    increment num_iterations
Output cluster assignment of each point to file

This program reads an input file where each row contains tab-
delimited features (expression data) for a single point (gene) to be
clustered. It outputs a file containing the line number of the point from the
input file, and the number of the cluster to which it was assigned. See
arguments for more specifics and options on input/output:

positional arguments:
  k                     The number of centroids or clusters for the kmeans alg
  expression_data       The tab-delimited input (directory/)file which
                        containsexpression data to cluster. Each row
                        represents a gene and each column represents some
                        conditions.
  max_iterations        The maximum number of iterations allowed during the
                        algorithm. Convergence may occur before this number,
                        in which case the number of iterations < max_iter
  centroids             An optional file specifying the initial centroids. If
                        this is not specified, initial positions are generated
                        from the positions of k random genes.

optional arguments:
  -h, --help            show this help message and exit
  -goi, --genes_of_interest [GENES_OF_INTEREST [GENES_OF_INTEREST ...]]
                        Specify genes of interest (by 1-index based row
                        number) for which cluster information is desired.
                        Summary info for the cluster results for these genes
                        are written to kmeans_query.out unless -goif is
                        specified. Use the '@' prefix to fetch these IDs from
                        a file
  -goif, --genes_of_interest_file GENES_OF_INTEREST_FILE
                        Specify the output filename for the genes of interest
                        cluster summary information. Default: kmeans_query.out
  -o OUTFILE, --outfile OUTFILE
                        Specify an output (directory/)filename. Output is in
                        tab-delimited format: 'gene row number' 'cluter
                        number' Default: kmeans.out
  -l {debug,info,warning}, --logger_level {debug,info,warning}
                        Specify the detail of print/logging messages.
  -t, --tests_only      If this option is specified, all input is ignored and
                        only unit tests are run.
