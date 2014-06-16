#...............................................................................
Name:  Chris Williams
SUNet: ccwillia
Discussed with: No one

#...............................................................................
# K-NEAREST-NEIGHBORS

usage: knn.py [-h] [-cv FOLDS] [-o OUTFILE] [-l {debug,info,warning}] [-t]
              file_train_positive file_train_negative k p

This is an implementation of the K-nearest-neighbors supervised ML alg orithm.
Given positive and negative example/training features, the cla ss of an
uncharacterized example is predicted based on the classes of its k nearest
neighbors, where 'nearest' is the smallest Euclidean distance in feature
space. In this implementation, the accuracy of classifications is based on
n-fold cross validation with the following pseudo code: 
1. Divide training set into n equally sized groups (+/- class ratios reflect 
   those in the entire training set) 
2. Using 1 group as a test set and n-1 groups as the training set, make 
   classifications 
3. Repeat 2 for each group in n 
4. Use TP/FP/TN/FN outcomes for all test items to determine 
   sensitivity/specificity/accuracy. 
   
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
1. Initialize k cluster centers. (randomly or from specified positions). 
2. Assign each data point to the cluster it is nearest to. 
3. Update the position of all clusters to the mean of the positions of all 
   points assigned to that cluster 
4. Repeat 2&3 until convergence (no points change clusters) or after a specified 
   number of iterations. 
   
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
