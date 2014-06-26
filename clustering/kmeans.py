#!/usr/bin/env python
info="""This is an implementation of the k-means clustering unsupervised ML 
        algorithm. It attempts to cluster n data points into k clusters such
        that the within-cluster sum of squares Euclidean distance from the 
        cluster mean is minimized. The general procedure it uses is as follows:

            1. Initialize k cluster centers. (randomly or from specified posit
               ions).
            2. Assign each data point to the cluster it is nearest to.
            3. Update the position of all clusters to the mean of the positions
               of all points assigned to that cluster
            4. Repeat 2&3 until convergence (no points change clusters) or 
               after a specified number of iterations

        This program reads an input file where each row contains tab-delimited
        features (expression data) for a single point (gene) to be clustered.
        It outputs a file containing the line number of the point from the input 
        file, and the number of the cluster to which it was assigned.
        See arguments for more specifics and options on input/output: 
     """
__author__ = "ccwiliams"
__date__   = "2013-10-26"

import sys
import random
import argparse
import logging
import numpy as np

#...............................................................................
# Global vars
OUTFILE     = "kmeans.out"       # name of the output file
GOI_OUTFILE = "kmeans_query.out" # name of output file for genes of interest
EPSILON     = 0.00001            # for floating point comparison
random.seed()

#...............................................................................
# Define arguments
prsr = argparse.ArgumentParser(description=info,
                               fromfile_prefix_chars="@")
prsr.add_argument("k", 
                  type=int,
                  help="The number of centroids or clusters for the kmeans alg")
prsr.add_argument("expression_data", 
                  type=str,
                  help="The tab-delimited input (directory/)file which contains" 
                       "expression data to cluster. Each row represents a gene "
                       "and each column represents some conditions.")
prsr.add_argument("max_iterations", 
                  type=int,
                  help="The maximum number of iterations allowed during the alg"
                       "orithm. Convergence may occur before this number, in wh"
                       "ich case the number of iterations < max_iter")
prsr.add_argument("centroids", 
                  type=str, nargs="?", default=None,
                  help="An optional file specifying the initial centroids. If "
                       "this is not specified, initial positions are generated "
                       "from the positions of k random genes.")
prsr.add_argument("-goi", "--genes_of_interest", nargs="*", type=int,
                  help="Specify genes of interest (by 1-index based row number)"
                       " for which cluster information is desired. Summary info"
                       " for the cluster results for these genes are written to"
                       " %s unless -goif is specified. Use the '@' prefix to "
                       "fetch these IDs from a file" % GOI_OUTFILE)
prsr.add_argument("-goif", "--genes_of_interest_file", type=str,
                  default=GOI_OUTFILE,
                  help="Specify the output filename for the genes of interest "
                       "cluster summary information. Default: %s" \
                       % GOI_OUTFILE)
prsr.add_argument("-o", "--outfile", 
                  type=str, default=OUTFILE,
                  help="Specify an output (directory/)filename. Output is in ta"
                       "b-delimited format: 'gene row number'\t'cluter number' "
                       "Default: %s" % OUTFILE)
prsr.add_argument("-l", "--logger_level", 
                  default="warning", choices=["debug", "info", "warning"],
                  help="Specify the detail of print/logging messages.")
prsr.add_argument("-t", "--tests_only", 
                  action="store_true",
                  help="If this option is specified, all input is ignored and o"
                       "nly unit tests are run.")

#...............................................................................
# Classes
class Centroid(object):
    """Class representing a k-means centroid/cluster center. Tracks Centroid
       number, Point assignments and provides methods for determining Euclidean 
       distances, adding/removing Points, and updating Centroid position.
    """
    def __init__(self, Centroid_number, position):
        self.number   = Centroid_number
        self.position = np.array(position)
        self.points   = [] # empty at initialization

    def __eq__(self, other):
        if isinstance(other, Centroid): return self.number == other.number
        return False

    def __ne__(self, other):
        if isinstance(other, Centroid): return self.number != other.number
        return True

    def add_Point(self, a_Point, distance):
        """This function adds a Point to the self points list. It also updates
           the Point's Centroid attribute to point to self, and updates the
           Point's dist_to_Centroid attribute to distance.

           @param a_Point     A Point object, to assign to this Centroid
           @param distance    The Euclidean distance between self and the Point
        """
        a_Point.dist_to_Centroid = distance
        self.points.append(a_Point)
        a_Point.Centroid = self
        logger.debug("Point %i Centroid updated to %i, distance=%.2f" % \
                     (a_Point.number, self.number, distance))
        return

    def remove_Point(self, a_Point):
        """This function removes a Point from the self points list. This will
           raise a Value error if the Point is not in the attribute list.

           @param a_Point   A Point object, to remove from this Centroid
        """
        self.points.remove(a_Point)
        if a_Point.Centroid == self: a_Point.Centroid = None
        return

    def get_euclid_distance(self, other):
        """This function returns the euclidean distance between self and other, 
           which is expected to be a Point object.

           @param other     a Point object for which a distance to this Centroid
                            is computed
           @return float< Euclidean distance >
        """
        return np.linalg.norm(self.position - other.position)

    def update_position(self):
        """This function updates the Centroid's position to the average of all
           Points in self's point attribute list
        """
        # If Centroid is empty, do not force assignment / remove from iterations
        if len(self.points) == 0: return

        # Compute average position of Points assigned to self
        start_position = str(self.position) # for debugging
        sum_positions = self.points[0].position 
        for point in self.points[1:]:
            sum_positions += point.position

        self.position = sum_positions / float( len(self.points) )
        logger.debug("Centroid %i position updated to %s from %s" % \
                     (self.number, start_position, str(self.position )))
        return

class Point(object):
    """Class representing a data Point in k-means clustering. Attributes 
       include Point number, Centroid assignment, and Point position. Supports 
       methods for updating Centroid assignment to the closest Centroid.
    """
    def __init__(self, number, position):
        self.number           = number
        self.Centroid         = None
        self.dist_to_Centroid = np.inf 
        self.position         = np.array(position)
    
    def update_Centroid_assignment(self, Centroids):
        """This function updates self's Centroid attribute to the Centroid in 
           Centroids which has the smallest Euclidean distance to self. In order 
           to determine if convergence has occured, this function returns 1 if
           the Centroid attribute was changed, and 0 if not.
           
           @param Centroids     list< all Centroid objects >
           @return int, 1 if self.Centroid changed and 0 if not
        """
        start_distance = self.dist_to_Centroid # for debugging
        start_Centroid = self.Centroid

        # Assign to the best Centroid based on distances
        for Centroid in Centroids:
            distance   = Centroid.get_euclid_distance(self)
            difference = (self.dist_to_Centroid - distance)
            
            # Keep current Centroid assignment if worse or equivalent to curr
            logger.debug("Point %i vs Centroid %i, distance=%.2f, diff=%.2f" %\
                         (self.number, Centroid.number, distance, difference))
            if (abs(difference) < EPSILON) or (difference < 0): 
                continue

            # Otherwise update to this Centroid
            if self.Centroid != None:
                self.Centroid.remove_Point(self) # not first assignment
            Centroid.add_Point(self, distance)

        logger.debug("Point %i start distance: %.2f, updated distance: %.2f"\
                     % (self.number, start_distance, self.dist_to_Centroid) )
        
        if start_Centroid == self.Centroid: return 0
        return 1
        
#...............................................................................
# Helper functions
def get_logger(logger_level):
    """Returns a console-only (i.e. no writing) logger with level set to 
       logger_level
       @param logger_level string, "debug"/"info"/"warning"
    """    
    levels = {"debug"   : logging.DEBUG,
              "info"    : logging.INFO,
              "warning" : logging.WARN}
    logger    = logging.getLogger("kmeans.py")
    console   = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(levels[logger_level])
    return logger

def positions_from_file(filename, delim='\t'):
    """Helper function to parse an input file and return a list of lists of
       positions, one for each line in the file.

       @param filename  str< the filename to parse >
       @param delim     The deliminator used to separate values on an input line
       @return list< list< float > >
    """
    lines   = []
    fhandle = open(filename,'r')
    for line in fhandle.readlines():
        position = [ float(val) for val in line.strip('\n').split(delim) ]
        lines.append(position)
    return lines

def init_Points(expression_file):
    """This function initializes Point objects from the data in expression_file
       Each Point is assigned a unique identifier equal to the its line number
       in expression_file

       @param expression_file File corresponding to args.expression_data
       @return list< Point >
    """
    lines  = positions_from_file(expression_file)
    points = [ Point(i+1, lines[i]) for i in range(len(lines)) ]

    logger.info("%i Points created from input file %s" % (len(points),
                                                          expression_file))
    return points

def init_Centroids(k, Centroid_file, all_Points):
    """This function initializes k Centroid objects. If Centroid_file
       is a valid file, Centroids are created from the positions defined,
       else positions are taken from k random Point objects from all_Points

       @param k             int, the number of Centroids to initialize
       @param Centroid_file str< filename > for initial Centroid positions 
                            or None
       @param all_Points    list< Point >
       @return list< Centroid >
    """
    # Fetch initial Centroid positions from file, if it's valid
    if Centroid_file != None:
        logger.info("Initializing k=%i Centroids from file %s" % \
                    (k, Centroid_file))
        lines     = positions_from_file(Centroid_file)
        Centroids = [ Centroid(i+1, lines[i]) for i in range(k) ] 

    # Otherwise set Centroid positions from randomly sampled Point positions
    else:
        logger.info("Initializing k=%i Centroids from randomly sampled Point "\
                    "positions" % k)
        random_pts = random.sample(all_Points, k)
        Centroids  = [ Centroid(i+1, random_pts[i].position) for i in range(k) ]

    return Centroids

def perform_kmeans(Centroids, points, max_iterations):
    """This function performs k-means clustering. Starting with inital 
       Centroids, each Point object is assigned to the closest Centroid, each
       Centroid's position is then updated to the mean of the Points assigned
       to it. This is repeated until convergence or max_iterations are carried
       out.
       
       @param Centroids      list< Centroid >
       @param points         list< Point >
       @param max_iterations int< max number of iterations allowed >
    """
    logger.info("Running k-means with %i Centroids, %i data Points" % \
                (len(Centroids), len(points)))
    num_iterations = 0 
    while True:
        Centroid_changes = 0
        num_iterations += 1
        # Update Point Centroid assignemnts based on distance to Centroid
        for point in points:
            Centroid_changes += point.update_Centroid_assignment(Centroids)
        # Update Centroid positions based on Points assigned to it 
        for Centroid in Centroids:
            Centroid.update_position()
        
        # k-means is done if we've converged or reached the max # of iterations
        if (num_iterations == max_iterations) or (Centroid_changes == 0): break

    logger.info("k-means terminated after %i iterations (max=%i)" \
                % (num_iterations, max_iterations) )
    return

def write_output(outfile, pts):
    """This function writes the kmeans results to file in the format:
            point \t cluster assignment
        where point is the line number of the corresponding data point from the
        input file

        @param outfile  The (directory/)name of the output file
        @param pts      list< Points >
    """
    logger.info("Writing output to file: %s" % outfile)
    output = \
        "\n".join(["%s\t%s" % (str(pt.number), str(pt.Centroid.number)) for pt in pts])
    fhandle = open(outfile,'w')
    fhandle.write(output)
    fhandle.close()
    logger.debug("kmeans results (gene, cluster):")
    logger.debug(output)
    return

def write_goi_summary(outfile, pts, Centroids, goi):
    """This function writes summary cluster information for the specified genes
       of interest to outfile.
       @param outfile   name of the (directory/)file to be written
       @param pts       list< Points >
       @param Centroids list< Centroids>
       @param goi       list< ints > where each int corresponds to a Point 
                        (row) number of interest
    """
    total_goi   = len(goi)
    logger.info("Writing summary information for %i points to file %s" %\
                (total_goi, outfile))
    # First group goi by cluster number for more compact summary information
    goi_dct     = {}
    cluster_dct = { number+1 : 0 for number in range(len(Centroids)) }
    for point in pts:
        cluster = point.Centroid.number
        cluster_dct[cluster] += 1
        if point.number in goi:
            if not goi_dct.has_key(cluster):
                goi_dct[cluster] = [point.number]
            else:
                goi_dct[cluster].append(point.number)

    fhandle = open(outfile,'w')
    for cluster, goi_list in goi_dct.items():
        num_goi       = len(goi_list)
        fract_goi     = float(num_goi)/total_goi
        size_cluster  = cluster_dct[cluster]
        fract_cluster = float(num_goi)/size_cluster

        fhandle.write("Cluster %i contained the following goi:\n" % cluster)
        fhandle.write("\t%s\n" % ",".join([str(goi) for goi in goi_list]))
        fhandle.write("\t(%i/%i)=%.5f of goi were in this cluster\n" % \
                      (num_goi, total_goi, fract_goi))
        fhandle.write("\t(%i/%i)=%.5f of this cluster were goi\n" % \
                      (num_goi, size_cluster, fract_cluster))
    fhandle.close()
    return

#...............................................................................
#   Main
def main():
    # First, initialize all Points and Centroids
    points    = init_Points(args.expression_data)
    Centroids = init_Centroids(args.k, args.centroids, points)

    # Now carry out k-means until convergence or max iterations is reached
    perform_kmeans(Centroids, points, args.max_iterations)

    # Write the assigned cluster assignemnts to file
    write_output(args.outfile, points)

    # If genes of interest specified, write summary information
    if isinstance(args.genes_of_interest, list):
        write_goi_summary(args.genes_of_interest_file, points, Centroids, 
                          args.genes_of_interest)
   
#...............................................................................
# Unit tests
def test_all():
    logger.warning("Running all unit tests ...")
    test_init_random_Centroids()
    test_Centroid_assignment()
    test_kmeans()
    logger.warning("All tests, passed, exiting.")

def make_random_Centroid_pts():
    C1 = Centroid(1, [1,1,1])
    C2 = Centroid(2, [100,100,100])
    C3 = Centroid(3, [1000,1000,1000])

    P1 = Point(1, [1.1,1.1,1.1])
    P2 = Point(2, [100.1,100.1,100.1])
    P3 = Point(3, [1000.1,1000.1,1000.1])

    return [C1, C2, C3], [P1,P2,P3]

def test_init_random_Centroids():
    logger.warning("Testing init_Centroids() (from points) ... ")
    P1 = Point(1, [1,1,1])
    Cs = init_Centroids(1, None, [P1])
    
    assert Cs[0].number   == 1
    for i in range(len(Cs[0].position)):
        assert Cs[0].position[i] == P1.position[i]

    logger.warning("Tests passed.")

def test_Centroid_assignment():
    logger.warning("Testing Centroid assignement methods ...")
    Centroids, points = make_random_Centroid_pts()

    # Initially, all points should change Centroid assignment
    deltas = 0
    for point in points:
        deltas += point.update_Centroid_assignment(Centroids)
    assert deltas == 3 # 3 updates    

    for point in points:
        assert point.number == point.Centroid.number # set up to be matched
        assert point in Centroids[points.index(point)].points

    # Second time nothing should change
    for point in points:
        deltas += point.update_Centroid_assignment(Centroids)
        # Also test remove_Point method
        point.Centroid.remove_Point(point)

    assert deltas == 3 # 0 additional updates
    # Check that Points were removed
    for point in points:
        assert point.Centroid == None

    logger.warning("Tests passed.")

def test_kmeans():
    logger.warning("Testing perform_kmeans() ...")
    Centroids, points = make_random_Centroid_pts() 
    perform_kmeans(Centroids, points, 1)

    # Test Point assignment and Centroid position updating
    for i in range(len(points)):
        assert points[i].number == points[i].Centroid.number

        for j in range(len(points[i].position)):
            assert points[i].position[j] == points[i].Centroid.position[j]
        
    logger.warning("Tests passed.")

#...............................................................................
# Flow determinant
if __name__ == "__main__":
    args   = prsr.parse_args()
    logger = get_logger(args.logger_level)
    if args.tests_only: test_all()
    else: main()
