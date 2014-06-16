#!/usr/bin/env python
info="""This is an implementation of a simplified molecular dynamics simulation.
        It uses a simplified force field function that includes only two terms:
        1. bonding interaction and 2. non-bonding interaction. Furthermore, it
        treats all atoms the same (same mass, no charges considered, etc.) and
        makes the assumption that non-bonding interactions are constant for 
        the entirety of the simulation. It approximates equations of motion 
        using the the Velocity Verlet integration algorithm. Briefly, this 
        updates positions, forces, accelerations, and velocities for a given
        delta time step by estimating a half-time-step velocity for each step.
        It has default parameters for the # of time steps in the simulation, 
        the size of each step, bonding and non-bonding spring 
        constants, but any of these can be specified. It takes 
        as input a .rvc file (see --iF for details) and writes to a similarly
        formatted .rvc file and a .erg file containing kinetic, bonding and 
        non-bonding potential energy, and total energy, every 10 steps of
        the simulation. If the simulation becomes unstable, a warning is thrown
        and the simulation will terminate. Additionally, options are available
        for generating a plot of all energy terms over all steps of the 
        simulation, as well as tracking the euclidean distance between specified
        atom pairs every 10 steps (see --plot and --track). Internally, it uses
        Atom objects which have methods to write information about themselves,
        and a bond indicator matrix mapping different bonding relationships 
        between atoms, which allows for an array-based implementation rather
        than numerous for loops
     """

__author__ = "ccwillia"
__date__   = "2013-11-05"

import re
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging
import datetime as dt
import numpy as np
np.seterr(over="raise",     # Raise FloatingPointError
          invalid="ignore") # Ignore ZeroDivisionError

#...............................................................................
#   GLOBAL parameters
DEFAULT_KB        = 40000.0
DEFAULT_KN        = 400.0
DEFAULT_NBCUTOFF  = 0.5
DEFAULT_MASS      = 12.0
DEFAULT_DT        = 0.001
DEFAULT_N         = 1000
OUTPUT_UNITS      = "kJ"

EPSILON           = 0.00001
WRITE_FREQ        = 10  # Number of time steps separating output file writes
REGEX_INPUT       = "\s*([0-9.-]+)"    # regex for parsing input .rvc file
REGEX_PDB         = "^#[\s]*([\w-]+):" # regex for fetching pdb name from .rvc
REGEX_TEMPERATURE = "([0-9.-]+)$"      # regex for fetching temp from .rvc
REGEX_BASENAME    = "/*([\w-]+)."      # regex for fetching base name of iF

IND_BOND          = 1 # indicator variable for a bonding interaction
IND_NONBOND       = 2 # indicator variable for a non-bonding interaction

#...............................................................................
#   Arguments
prsr = argparse.ArgumentParser(description=info)
prsr.add_argument("--iF", type=str,
                  help="(dir/)Name of the input .rvc file which contains a sing"
                       "le header line, followed by one space-delimited line pe"
                       "r atom containing: atom ID, rx, ry, rz, vx, vy, vz, and"
                       " up to 4 other atom IDs corresponding to bonded atoms.")
prsr.add_argument("--kB", type=float, default=DEFAULT_KB,
                  help="The spring constant for BONDED atoms in the force field"
                       " function. Default: %.1f" % DEFAULT_KB)
prsr.add_argument("--kN", type=float, default=DEFAULT_KN, 
                  help="The spring constant for NON-BONDED atoms in the force f"
                       "ield function. Default: %.1f" % DEFAULT_KN)
prsr.add_argument("--nbCutoff", type=float, default=DEFAULT_NBCUTOFF,
                  help="Distance within which atoms are considered as having no"
                       "n-bonded interactions (if not covalently bonded). Defau"
                       "lt: %.1f" % DEFAULT_NBCUTOFF)
prsr.add_argument("--m", type=float, default=DEFAULT_MASS,
                  help="Atom mass applied as a constant to all atoms. Default: "
                       "%.1f" % DEFAULT_MASS)
prsr.add_argument("--dt", type=float, default=DEFAULT_DT,
                  help="Length of time step. Default: %.4f" % DEFAULT_DT)
prsr.add_argument("--n", type=int, default=DEFAULT_N,
                  help="Number of time steps to iterate. Default: %i" % \
                        DEFAULT_N)
prsr.add_argument("--out", type=str, 
                  help="Prefix of the output filename, e.g., <out>_out.rvc. By "
                       "default, the suffix-stripped input file name is used.")
prsr.add_argument("--plot", action="store_true",
                  help="If this option is specified, a plot of energy over the "
                       "course of the simulation will be made.")
prsr.add_argument("--track", type=str, nargs="+",
                  help="Specify a list of interaction pairs to track over the c"
                       "ourse of the simulation. The euclidean distance between"
                       " each pair will be written to file every %i steps. The "
                       "expected format is ID1-ID2 to track the distance betwee"
                       "atoms with IDs 1 and 2. Output format will be: step\tdi"
                       "stance\n." % WRITE_FREQ)
prsr.add_argument("-t", "--tests_only",
                  action="store_true",
                  help="If this option is specified, all input is ignored and o"
                       "nly unit tests are run.")
prsr.add_argument("-l", "--logger_level",
                  default="warning", choices=["debug", "info", "warning"],
                  help="Specify the detail of print/logging messages.") 

#...............................................................................
#   Classes
class Atom(object):
    """A class representing an atom. 
    """
    def __init__(self, info, positions, velocities, accels, forces, 
                 bond_indicator, mass):
        self.ID        = int( info[0] ) 
        self.mass      = mass
        self.coord     = positions[self.ID - 1, :]
        self.velocity  = velocities[self.ID - 1, :]
        self.accel     = accels[self.ID - 1, :]
        self.force     = forces[self.ID - 1, :]
        self.bonded_atoms = sorted([int(ID) for ID in info[7:]])
        self.populate_indicator_matrix(bond_indicator)

    def __eq__(self, other):
        """Tests Atom equality
        """
        if isinstance(other, Atom): return self.ID == other.ID
        return NotImplemented

    def __str__(self):
        """This method returns a string describing the Atom. It happens to be
           in the format fit for output to a .rvc file:
                atom-ID\trx\try\trz\tvx\tvy\tvz\tbonded-atom-IDs\n
           All values except IDs have four decimal points
        """
        positions  = "\t".join(["%.4f" % pos for pos in self.coord]) 
        velocities = "\t".join(["%.4f" % vel for vel in self.velocity])
        atoms      = "\t".join([str(atom) for atom in self.bonded_atoms])
        line = "%i\t%s\t%s\t%s\n" % (self.ID, positions, velocities, atoms)
        return line

    def get_distance(self, other, euclidean=True):
        """Returns the distance between self and other, with respect to self
           This is only used for tracking atom distances
           (nb: this does not matter if euclidean distance is calculated):
                e.g., other.x - self.x

           @param atom1     Atom object1, distance is relative to this atom
           @param atom2     Atom object2
           @param euclidean bool, whether distance returned should be euclidean
           @return float< euclidean ditance > or
                   np.array(float<x2-x1>, float<y2-y1>, float<z2-z1>) 
        """
        if euclidean: 
            return np.linalg.norm(other.coord - self.coord)
        return other.coord - self.coord

    def populate_indicator_matrix(self, indicator_matrix):
        """This function fills an indicator matrix with bonded interactions
           for this Atom.
           @param indicator_matrix nxn array where rows represent query atom IDs
                                   and columsn indicate atom IDs with respect
                                   to the query Atom
        """
        # Populate cells with the indicator representing bonds, for Atoms to 
        # which this Atom is bonded. Correct for 0-idx of arrays!
        for bonded_atom in self.bonded_atoms:
            indicator_matrix[self.ID - 1, bonded_atom - 1] = IND_BOND
        return

#...............................................................................
#   Helper Functions

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

def init_atoms(inputfile, mass, spring_constant):
    """This function initializes all atoms for the simulation, based on info
       read from inputfile. It returns a dictionary of Atom objects, keyed on
       Atom ID. 
       @param inputfile       input .rvc file to simulation data
       @param mass            float, mass of atoms
       @param spring_constant float, representing the stiffness of bonded 
                              atom bonds
       @return atom_dict, position_array, velocity_array, accel_array, 
               force_array, bond_matrix, temperature, pdb
    """
    start = dt.datetime.now()
    logger.info("Initializing atoms from file '%s'" % inputfile)
    # Determine number of atoms to be made, initialize arrays for values
    lines                 = open(inputfile, 'r').readlines()
    num_atoms             = len(lines) - 1 # -1 for header
    atoms                 = {}
    positions, velocities = [ np.empty( (num_atoms, 3) ) for i in range(2) ]
    accels, forces        = [ np.zeros( (num_atoms, 3) ) for i in range(2) ]
    bond_matrix           = np.zeros( (num_atoms, num_atoms) )

    # Initialize all atoms, bonding interactions noted here
    pdb         = re.findall(REGEX_PDB, lines[0])[0]
    temperature = float(re.findall(REGEX_TEMPERATURE, lines[0])[0])
    for line in lines[1:]:
        info                    = re.findall(REGEX_INPUT, line)
        atom_ID                 = int( info[0] ) # convert to 0-idx for arrays
        positions[atom_ID-1, :] = info[1:4]
        velocities[atom_ID-1,:] = info[4:7]
        atoms[atom_ID] = \
            Atom(info, positions, velocities, accels, forces, bond_matrix, mass)

    num_bonds = np.nansum(bond_matrix) / 2 # each is represented 2x 
    logger.info("%i atoms initialized" % len(atoms))
    logger.info("%i bonded interactions assigned" % num_bonds)
    stop = dt.datetime.now()
    logger.debug("Init atoms completed in %ius" % ((stop-start).microseconds))
    return atoms, positions, velocities, accels, \
           forces, bond_matrix, temperature, pdb

def get_component_distances(pos_matrix, bond_matrix, bond_indicator, eq=True):
    """Given an array of positions and a bond indicator array, this function 
       returns three nxn distance arays, where rows are the query genes,
       i.e., cell values represent:  column gene x - row gene x
                                     column gene y - row gene y
                                 or  column gene z - row gene z

       @param pos_matrix        nx3 array where each row is a gene and columns
                                are x,y,z positions
       @param bond_matrix       nxn array of bond type indicators
       @param bond_indicator    distances for bonds with indicator values 
                                == to this number will be included in the 
                                output arrays
       @param eq                bool, if indicator value shoudl be equal to
                                the indicator specified. If False, will be
                                >=
       @return dict< 'x':dx, 'y':dy, 'z':dz>  dict of 3 nxn distance arrays
    """
    start = dt.datetime.now()
    # Get boolean array for masking bonds we don't care about
    if eq:
        bond_mask = bond_matrix == bond_indicator
    else:
        bond_mask = bond_matrix >= bond_indicator

    # Get nxn arrays for each distance component
    dx        = (bond_mask * pos_matrix[:,0]) - \
                (bond_mask * pos_matrix[:,0]).transpose()
    dy        = (bond_mask * pos_matrix[:,1]) - \
                (bond_mask * pos_matrix[:,1]).transpose()
    dz        = (bond_mask * pos_matrix[:,2]) - \
                (bond_mask * pos_matrix[:,2]).transpose()

    stop = dt.datetime.now()
    logger.debug("Component dist for indicator %i completed in %ius" % \
                 (bond_indicator, (stop-start).microseconds))
    return {'x' : dx, 'y' : dy, 'z': dz }

def get_euclidean_distance(dict_dist):
    """Given x, y, and z component distance matrices, this function returns an
       nxn array of euclidean distances.

       @param dict_dist     dict< 'x':dx, 'y':dy, 'z':dz>  
                            dict of 3 nxn distance arrays
       @return nxn array of euclidean distances
    """
    start = dt.datetime.now()
    # Calculate the euclidean distance by x,y,z components 
    d_euclid = np.sqrt( dict_dist['x']**2 + \
                        dict_dist['y']**2 + \
                        dict_dist['z']**2 )
    stop = dt.datetime.now()
    logger.debug("Euclid. dist completed in %ius" % ((stop-start).microseconds))
    return d_euclid

def get_distances(array_positions, array_bonds, return_comps=False):
    """A wrapper function for computing reference lengths.
    """
    # First get component distances for each bond type
    bond_comp_dist  = \
        get_component_distances(array_positions, array_bonds, IND_BOND)
    nbond_comp_dist = \
        get_component_distances(array_positions, array_bonds, IND_NONBOND)

    # Use these to get component differences
    bond_d0     = get_euclidean_distance(bond_comp_dist)
    nbond_d0    = get_euclidean_distance(nbond_comp_dist)
    dict_euclid = { "bonded" : bond_d0, "nonbonded" : nbond_d0 }

    if return_comps:
        return bond_comp_dist, nbond_comp_dist, dict_euclid
    return dict_euclid

def make_nonbond_pairs(array_positions, bond_array, nbCutoff):
    """This function populates the indicator bond_array matrix with
       indicator variables representing non-bonded interactions. Non-bond
       interactions cannot occur between an Atom and itself, with Atoms
       that are already bonded, or between Atoms separated by a specified
       distance

       @param array_positions nx3 array of x,y,z atom positions
       @param bond_array      nxn array of indicator variables representing
                              bond interactions
       @param nbCutoff        Cutoff used for determining non-bond interactions
                              Atoms with euclidean distances > nbCutoff are not
                              allowed to be non-bonded
    """
    start = dt.datetime.now()
    # First, determine distance between all atoms for subjecting to cutoff
    dict_dist_comps = \
        get_component_distances(array_positions, bond_array, 0, eq=False)
    all_eucl_d = get_euclidean_distance(dict_dist_comps)

    # Now impose a distance cutoff and update the indicator array with non-bond
    # interactions, as long as they are not already bonded
    nonbond_indices = (all_eucl_d <= nbCutoff) & (bond_array < IND_BOND) 
    np.fill_diagonal(nonbond_indices, False) # no bonding to self either

    num_nonbonds = np.nansum(nonbond_indices) / 2 # each is represented 2x
    bond_array[nonbond_indices] = IND_NONBOND

    stop = dt.datetime.now()
    logger.info("%i non-bonded interactions assigned" % num_nonbonds)
    logger.debug("Make nonbond pairs done in %ius" % ((stop-start).microseconds))
    return

def update_velocities(accel_array, velocity_array, args, KE):
    """This function updates Atom velocities for all Atoms in atom_dict for a 
       specific time step dt. It returns the total KE for all atoms if specified

       @param accel_array    nx3 array of x,y,z atom accelerations
       @param velocity_array nx3 array of x,y,z atom velocities
       @param args      input arguments
       @param KE        bool, whether KE should be calculated and returned
       @return          float< total KE > if KE = True else nothing returned
    """
    start = dt.datetime.now()
    total_KE = None
    velocity_array += 0.5 * accel_array * args.dt

    if KE:
        total_KE = np.nansum(0.5 * args.m * velocity_array**2) 

    stop = dt.datetime.now()
    logger.debug("Vel update done in %ius" % (stop-start).microseconds)
    return total_KE

def update_positions(position_array, velocity_array, args):
    """This function updates Atom positions for all Atoms in atom_dict for a
       specified time step dt.

       @param position_array nx3 array of x,y,z positions
       @param velocity_array nx3 array of x,y,z atom velocities
       @param args      program input arguments
    """
    start = dt.datetime.now()
    position_array += velocity_array * args.dt
    stop = dt.datetime.now()
    logger.debug("Position update done in %ius" % ((stop-start).microseconds))
    return

def update_forces(bonds, positions, forces, ref_dists, args):
    """This function updates Fx, Fy, and Fz components for all atoms

       @param bonds     nxn bond indicator array
       @param positions nx3 array of x,y,z atom positions
       @param forces    nx3 array of x,y,z atom forces
       @param ref_dists dict< 'bonded'  : nxn array of bonded ref distances,
                              'nonboned': nxn array of nonbonded ref distances >
       @param args      program input arguments
       @return dict< "bonded" :    euclid_dist_array,
                     "nonbonded" " euclide_dist_array >
    """
    start = dt.datetime.now()
    # First compute current component and euclidean distances
    bond_comp, nbond_comp, euclid = \
        get_distances(positions, bonds, return_comps=True)

    # Compute Force magnitudes
    mag_F_bond  = args.kB * (euclid["bonded"] - ref_dists["bonded"])
    mag_F_nbond = args.kN * (euclid["nonbonded"] - ref_dists["nonbonded"])
   
    # Zero Forces, make a function for summing
    forces *= 0
    modsum  = lambda a: np.nan_to_num(np.nansum(a, axis=1))

    # Update force components with forces from bonded interactions
    # sum along axis = 1 
    forces[:,0] += modsum( mag_F_bond * ( bond_comp["x"] / euclid["bonded"]))
    forces[:,1] += modsum( mag_F_bond * ( bond_comp["y"] / euclid["bonded"]))
    forces[:,2] += modsum( mag_F_bond * ( bond_comp["z"] / euclid["bonded"]))
    #forces[np.isnan(forces)] = 0 # avoid nans

    # Update force components with forces from bonded interactions
    forces[:,0] += modsum( mag_F_nbond*(nbond_comp["x"] / euclid["nonbonded"]))
    forces[:,1] += modsum( mag_F_nbond*(nbond_comp["y"] / euclid["nonbonded"]))
    forces[:,2] += modsum( mag_F_nbond*(nbond_comp["z"] / euclid["nonbonded"]))

    stop = dt.datetime.now()
    logger.debug("Force update done in %ius" % ((stop-start).microseconds))
    return euclid

def get_PEs(curr_dists, ref_dists, args):
    """This function computes the total bonded and nonbonded potential energy
       for all interacting (bonded/non-bonded) atom pairs. 
       @param curr_dists dict< 'bonded'  : nxn array of current bonded distances,
                               'nonboned': nxn array of current nonbonded distances >
       @param ref_dists  dict< 'bonded'  : nxn array of bonded ref distances,
                               'nonboned': nxn array of nonbonded ref distances >
       @param args       program input arguments
       @return dict< "bonded" :    float< bonded PE>, 
                     "nonbonded" : float< nonbonded PE> >
    """
    start = dt.datetime.now()
    PEs   = {}
    # Compoute PE as sum( 1/2 * K * (curr_dist - ref_dist)**2 )
    # Only count half the triangle so no double counting
    PEs["bonded"]    = \
            np.nansum( np.tril( 0.5*args.kB*(curr_dists["bonded"] - \
                                             ref_dists["bonded"])**2) )
    PEs["nonbonded"] = \
            np.nansum( np.tril( 0.5*args.kN*(curr_dists["nonbonded"] - \
                                             ref_dists["nonbonded"])**2) )

    stop  = dt.datetime.now()
    logger.debug("PE update done in %ius" % ((stop-start).microseconds))
    return PEs

def update_accelerations(forces, accelerations, args):
    """This function updates atom accelerations according to:
       F = 1/m * F

       @param forces        nx3 array of atom x,y,z forces
       @param accelerations nx3 array of atom x,y,z accelerations
       @param args          program input arguments
    """
    start = dt.datetime.now()
    accelerations *= 0 # update in place instead of copying
    accelerations += (1/args.m) * forces
    stop = dt.datetime.now()
    logger.debug("Accel. update done in %ius" % ((stop-start).microseconds))
    return

def init_outfiles(args, temp, pdb, atoms):
    """This function initializes the file connections for the two output files 
       (.erg and .rvc). It writes headers and returns the file handles.
    
       @param args      argparse argument list, used for header information
       @param temp      Temperature of simulation, from input file
       @param atoms     dict< atom_ID : Atom >
       @return fhandle_rvc, fhandle_erg
    """
    start = dt.datetime.now()
    # Initialize files
    fh_rvc    = open(args.out + "_out.rvc", "w")
    fh_erg    = open(args.out + "_out.erg", "w")
    track_tup = init_track_file(args, atoms) if args.track != None else False

    # Write headers
    fh_erg.write("# step\tE_k\tE_b\tE_nB\tE_tot\n")
    rvc_hd = "# %s: kB=%s kN=%s  nbCutoff=%.2f dt=%.4f  mass=%.1f  T=%.1f\n" % \
             (pdb, args.kB, args.kN, args.nbCutoff, args.dt, args.m, temp)
    fh_rvc.write(rvc_hd)
    # We also have to copy the initial .rvc data to the ouput .rvc
    for atom in atoms.values(): fh_rvc.write( str(atom) )
    
    logger.info("Opened output file streams: %s, %s" % (fh_rvc.name, 
                                                        fh_erg.name) )
    stop = dt.datetime.now()
    logger.debug("Init outfiles done in %ius" % ((stop-start).microseconds))
    return fh_rvc, fh_erg, track_tup

def init_track_file(args, atoms):
    """This function initializes a file for tracking the distances between 
       atoms specified in args.
       @param args  argparse arguments
       @param atoms dict< atom_ID : Atom >
       @return tuple< list< list<atom pair> >, tracking filehandle >
    """
    fh_track = open(args.out + "_".join(args.track) + "_track.txt", 'w')
    fh_track.write("#Step\t%s\n" % "\t".join(args.track))

    f = lambda str_pair: [ int(atom) for atom in str_pair ]
    to_track = [ f(string.split("-")) for string in args.track ]

    line = "0"
    for ID1, ID2 in to_track:
        line += "\t%.4f" % atoms[ID1].get_distance(atoms[ID2])
    fh_track.write(line + "\n")

    logger.info("Initializing track file at %s" % fh_track.name)
    return (to_track, fh_track)


def write_output(fh_rvc, fh_erg, track_tup, atom_dict, curr_step, PE_dict, KE):
    """This function updates the output files with the status of the simulation
       at the current time.

       @param fh_rvc    file handle for the output .rvc file
       @param fh_erg    file handle for the output .erg file
       @param track_tup tuple< list< list<atom pair> >, tracking filehandle >
       @param atom_dict dict< atom_ID : Atom >
       @param curr_step int, the current step in the simulation (1-based)
       @param PE_dict   dict<"bonded": float<PEb>, "nonbonded" : float<PEnb> >
       @param KE        float< KE >
    """
    start = dt.datetime.now()
    PE_b  = PE_dict["bonded"]
    PE_nb = PE_dict["nonbonded"]
    E_tot = KE + PE_b + PE_nb
    rvc_sep = \
        "#At time step %i,energy = %.3f%s\n" % (curr_step, E_tot, OUTPUT_UNITS)
    erg_line = \
        "%i\t%.1f\t%.1f\t%.1f\t%.1f\n" % (curr_step, KE, PE_b, PE_nb, E_tot)

    # Write line to .erg file
    fh_erg.write(erg_line)
    # Write seperator and information for each atom to .rvc file
    fh_rvc.write(rvc_sep)

    # Ensure atom output is in order
    atoms = range(1, len(atom_dict)+1)
    for ID in atoms:
        fh_rvc.write( str(atom_dict[ID]) )
    # If specified, write to atom tracking file
    if track_tup:
        line = str(curr_step)
        for ID1, ID2 in track_tup[0]:
            line += \
                "\t%.4f" % atom_dict[ID1].get_distance(atom_dict[ID2])
        track_tup[1].write(line + "\n")

    stop = dt.datetime.now()
    logger.debug("Write output done in %ius" % ((stop-start).microseconds))
    return

def plot_energies(outbase, energy_dict, time_step):
    """This function plots Energies over time.
       @param outbase     Basename for output file
       @param energy_dict dict keyed on KE/PEb/PEnb/tot with lists of energy 
                          values over the simulation
       @param time_step   The time step used for each step in the 
                          simulation
    """
    outname = outbase + "_energyplot.pdf"
    logger.info("Saving Energy plot to file %s" % outname)
    fig, ax = plt.subplots()

    # Convert inf vals to real numbers
    energy_dict = { energy : np.nan_to_num(vals) \
                    for energy, vals in energy_dict.items() }
    time  = np.arange(0, len(energy_dict["KE"])*time_step, time_step)
    # Plot data
    ax.plot(time, energy_dict["KE"],   color="dodgerblue", label="KE")
    ax.plot(time, energy_dict["PEb"],  color="orangered",  label="PE-bond")
    ax.plot(time, energy_dict["PEnb"], color="grey",       label="PE-nonbond")
    ax.plot(time, energy_dict["tot"], 'k--',               label="Total") 
    # Add axis labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (%s)" % OUTPUT_UNITS)
    legend = ax.legend(loc="best")
    fig.savefig(outname)
    return

#...............................................................................
#   Main
def main():
    # Time simulation, flag overflows
    start = dt.datetime.now()

    # Keep track of all energies over time
    energies = { Etype : [] for Etype in ["KE", "PEb", "PEnb", "tot"] }

    # Initialize atoms and bonding interactions
    atoms, \
    positions, velocities, accels, forces, bond_indicator, \
    temperature, pdb = init_atoms(args.iF, args.m, args.kB)

    # Make non-bond interactions based on length
    make_nonbond_pairs(positions, bond_indicator, args.nbCutoff) 

    # Determine reference lengths for bond and non-bond interactions
    ref_dists = get_distances(positions, bond_indicator)
    
    # Initialize the output files, which are written to every few steps
    fh_rvc, fh_erg, track_tup = \
        init_outfiles(args, temperature, pdb, atoms)

    # The time step loop
    time = 0
    for time_step in range(1, args.n + 1): # make time 1-based idx
        time += args.dt

        # Catch OverFlowErrors
        try:
            # The Velocity Verlet algorithm
            update_velocities(accels, velocities, args, KE=False)
            update_positions(positions, velocities, args)
            curr_dists = update_forces(bond_indicator, positions, 
                                       forces, ref_dists, args)
            update_accelerations(forces, accels, args)
            KE = update_velocities(accels, velocities, args, KE=True)
            PE = get_PEs(ref_dists, curr_dists, args)

        except FloatingPointError:
            logger.warning(\
                "Overflow error at time step %i (time=%.4f)" % (time_step, time))
            break

        # Track energies
        energies["KE"].append(KE)
        energies["PEb"].append(PE["bonded"])
        energies["PEnb"].append(PE["nonbonded"])
        energies["tot"].append(KE + PE["bonded"] + PE["nonbonded"])

        if time_step % WRITE_FREQ == 0:
            write_output(fh_rvc, fh_erg, track_tup, atoms, time_step, PE, KE)
        if time_step % 100 == 0:
            logger.info("Step %i of %i complete ..." % (time_step, args.n))

    fh_rvc.close()
    fh_erg.close()
    if track_tup: track_tup[1].close()
    if args.plot: plot_energies(args.out, energies, args.dt)
    stop = dt.datetime.now()
    logger.info("Simulation complete in %.1fs. Exiting!" % (stop-start).seconds)
#...............................................................................
#   Flow determinant
if __name__ == "__main__":
    os.chdir(os.getcwd())
    args   = prsr.parse_args()
    if args.out == None:
        args.out = re.findall(REGEX_BASENAME, args.iF)[0] 
    logger = get_logger(args.logger_level)
    main()
