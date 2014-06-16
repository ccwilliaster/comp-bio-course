#!/usr/bin/env python
info="""This is an implementation of a simplified molecular dynamics simulation.
        It uses a simplified force field function that includes only two terms:
        1. bonding interaction and 2. non-bonding interaction. Furthermore, it
        treats all atoms the same (same mass, no charges considered, etc.) and
        makes the assumption that non-bonding interactions are constant for 
        the entirety of the simulation. It has default parameters for the # of
        time steps in the simulation, the size of each step, bonding and non-
        bonding spring constants, but any of these can be specified. It takes 
        as input a .rvc file (see --iF for details) and writes to a similarly
        formatted .rvc file and a .erg file containing kinetic, bonding and 
        non-bonding potential energy, and total energy, every 10 steps of
        the simulation. If the simulation becomes unstable, a warning is thrown
        and the simulation will terminate. Additionally, options are available
        for generating a plot of all energy terms over all steps of the 
        simulation, as well as tracking the euclidean distance between specified
        atom pairs every 10 steps (see --plot and --track).
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
np.seterr(over='raise') # Raise FloatingPointError for overflow errors

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
    def __init__(self, rvc_line, mass):
        self.ID        = int( rvc_line[0] ) 
        self.mass      = mass
        self.coord     = np.array( [ float(pos) for pos in rvc_line[1:4] ])
        self.velocity  = np.array( [ float(vel) for vel in rvc_line[4:7] ])
        self.accel     = np.zeros(3)
        self.force     = np.zeros(3)
        self.bonds     = { "bonded": set(), "nonbonded": set() }
        self.atoms     = { "bonded": set(), "nonbonded": set() }
        self.to_bond   = set([int(ID) for ID in rvc_line[7:]])

    def __eq__(self, other):
        """Tests Atom equality
        """
        if isinstance(other, Atom):
            return self.ID == other.ID
        return NotImplemented

    def __str__(self):
        """This method returns a string describing the Atom. It happens to be
           in the format fit for output to a .rvc file:
                atom-ID\trx\try\trz\tvx\tvy\tvz\tbonded-atom-IDs\n
           All values except IDs have four decimal points
        """
        positions  = "\t".join(["%.4f" % pos for pos in self.coord]) 
        velocities = "\t".join(["%.4f" % vel for vel in self.velocity])
        atoms      = \
            "\t".join([str(ID) for ID in sorted(list(self.atoms["bonded"]))])
        line = "%i\t%s\t%s\t%s\n" % (self.ID, positions, velocities, atoms)
        return line

    def init_bonds(self, Atom_dict, bond_type, spring_constant):
        """Given a dictionary of Atoms, this method initiates bonds of type
           bond_type with all atoms in the self.to_bond set attribute. It
           returns the number of Bonds initialized in the process

           @param Atom_dict         dict< atom_ID : Atom >
           @param bond_type         str, bonded or nonbonded, all bonds formed
                                    will be of this type
           @param spring_constant   Constant reflecting Bond stiffness
           @return int< number of Bonds initialized >
        """
        ct = 0
        IDs_to_bond = list( self.to_bond ) # so we don't remove from set as we go
        for atom_ID in IDs_to_bond:
            self.init_bond(Atom_dict[atom_ID], bond_type, spring_constant)
            ct += 1
        return ct

    def init_bond(self, other, bond_type, spring_constant, distance=False):
        """The method forms a bond of the specified type between self and other
           It updates the bonds and to_bond attributes of both Atom objects.

           @param other             Atom to form a Bond with
           @param bond_type         Type of bond, 'bonded' or 'nonbonded'
           @param spring_constant   float<constant representing bond stiffness>
           @param distance          Specify the inter-atom distance, if known
                                    else it is computed
        """
        newbond = Bond(self, other, bond_type, spring_constant, distance)

        # Update Atom attributes to reflect bond formation
        self.bonds[bond_type].add(newbond)
        if other.ID in self.to_bond: self.to_bond.remove(other.ID)
        self.atoms[bond_type].add(other.ID)

        other.bonds[bond_type].add(newbond)
        if self.ID in other.to_bond: other.to_bond.remove(self.ID)
        other.atoms[bond_type].add(self.ID)

    def get_distance(self, other, euclidean=True):
        """Returns the distance between self and other, with respect to self
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

    def get_kinetic_energy(self):
        """This method computes and returns the Atom's current KE, the sum of
           the KEs in each velocity direction:
                1/2*mass*(v)^2
           @return float< kinetic energy > #nb: NOT AN ARRAY
        """
        return sum( 0.5 * self.mass * self.velocity**2  )

    def update_velocity(self, dt, KE):
        """This method updates the Atom's velocity attribute according to:
                v := v(curr_v) + 1/2*(curr_a)*dt
           It returns KE if specified
           @param dt    The timestep dt used in the velocity update above
           @param KE    bool, whether to update return the KE corresponding to
                        the updated velocity
           @return      float< KE > if KE = True else np.nan
        """
        self.velocity = self.velocity + (0.5 * self.accel * dt)
        return self.get_kinetic_energy() if KE else np.nan

    def update_position(self, dt):
        """This method updates the Atom's coord attribute according to:
                coord := curr_coord + curr_vel * dt 
           where curr_vel is the half time velocity in the Velocity Verlet alg

           @param dt    The timestep dt used in the coordinate update above
        """
        self.coord = self.coord + (self.velocity * dt)
        return 

    def update_force(self, curr_time):
        """This method updates the Atom's force attribute vector according to:
                Fx := sum over all bonded atoms ( Fx(self, other_Atom) )
                Fy, Fz updated similarly
           @param curr_time     float, the current time of simulation, used to
                                prevent duplicate force updates computations
        """
        # First, reset the force components to 0
        self.force *= 0
        # Now iterate over all bonds, summing the force contribution of each
        # nb: the Bond get_force() method prevents duplicate calculations and
        #     abstracts the force direction
        for bond_type, bond_set in self.bonds.items():
            for bond in bond_set:
                #if (3 in [bond.atoms[0].ID, bond.atoms[1].ID]) and \
                #   (4 in [bond.atoms[0].ID, bond.atoms[1].ID]): 
                #       print "atom 3-4 ref dist %s" % str(bond.ref_length)     
                self.force += bond.get_force(self, curr_time)
        return

    def update_acceleration(self):
        """This method updates the Atom's accel attribute according to:
                a(t+dt) := 1/mass * F(t+dt)
        """
        self.accel = (1/self.mass) * self.force
        return

class Bond(object):
    """A class representing a bond, bonding or non-bonding.
    """
    def __init__(self, atom1, atom2, bond_type, spring_constant, distance=False):
        self.bond_type    = bond_type
        self.atoms        = (atom1, atom2)
        self.stiffness    = spring_constant
        self.ref_length   = distance if distance else atom1.get_distance(atom2)
        self.length       = distance if distance else atom1.get_distance(atom2)
        self.PE           = 0
        self.time_updated = {"PE":0, "force":0} # to prevent redundant calcs
        self._force       = np.zeros(3) # Fx, Fy, Fz
        self._length_comp = np.zeros(3) # distx, disty, distz 
        # nb: private _force and _length_comp are relative to atom 1
        #     e.g., atom2.x - atom1.x

    def __str__(self):
        """String representation for Bonds
        """
        out = "B<%s, k=%.3f, atoms=%i,%i, len,ref=%0.4f,%.4f, updated@t=%s>" \
              % (self.bond_type, self.stiffness, self.atoms[0].ID, 
                 self.atoms[1].ID, self.length, self.ref_length, 
                 str(self.time_updated))
        return out

    def get_force(self, atom, curr_time):
        """This method returns a numpy array containing Fx, Fy, Fz force
           components, experience by the Atom specified by atom. 
           @param atom      Atom object for which forces are desired, this
                            will determine the direction of the forces
           @param curr_time Current time in simulation, used to prevent
                            duplicate computations
        """
        # Distances used are assumed to reflect the current state
        assert self.time_updated["PE"] == curr_time

        # Don't calculate force twice if it has already been updated 
        if self.time_updated["force"] != curr_time:
            # First calculate F magnitude, then components relative to atom 1
            mag_force = self.stiffness * (self.length - self.ref_length)
            for i in range(3):
                self._force[i] = mag_force * self._length_comp[i] / self.length

            self.time_updated["force"] = curr_time

        return self._force if atom == self.atoms[0] else -self._force

    def update_length(self):
        """This method updates the Bond's length attribute based on current
           Atom positions.
        """
        # Public euclidean distance
        self.length = self.atoms[0].get_distance(self.atoms[1])
        # Private distance vector, sign comes from atom2 - atom1
        self._length_comp = self.atoms[0].get_distance(self.atoms[1], 
                                                       euclidean=False)
        return

    def get_PE(self, curr_time):
        """This method computes and returns the potential energy of the Bond
           according to:
                PE = 1/2*K*(length - ref_length)^2
                where K is the bond stiffness constant
           @param curr_time     float< current time in simulation> This is used
                                to prevent double counting of PEs if this 
                                method is called on both atoms in the Bond.
           @return float< PE >
        """
        # Don't count potential energy twice
        assert curr_time != self.time_updated["PE"]

        # Compute PE after updating bond length, then update time attribute
        self.update_length()
        self.PE = 0.5 * self.stiffness * (self.length - self.ref_length)**2
        self.time_updated["PE"] = curr_time
        return self.PE

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
       @return dict< atom_ID: Atom >, float<input_temp>, str<input_pdb>
    """
    start = dt.datetime.now()
    logger.info("Initializing atoms from file '%s'" % inputfile)
    atoms   = {} # key on atom ID for easy lookup
    bond_ct = 0
    fhandle = open(inputfile, 'r')

    # Initialize all atoms (no bonds formed here)
    for line in fhandle.readlines():
        if line[0] == "#": 
            pdb         = re.findall(REGEX_PDB, line)[0]
            temperature = float(re.findall(REGEX_TEMPERATURE, line)[0])
            continue
        info    = re.findall(REGEX_INPUT, line)
        atom_ID = int( info[0] )
        atoms[atom_ID] = Atom(info, mass)
    logger.info("%i atoms initialized" % len(atoms))
    fhandle.close()

    # Now initialize bonds
    for atom in atoms.values():
        bond_ct += atom.init_bonds(atoms, "bonded", spring_constant)

    logger.info("%i bonded Bonds initialized" % bond_ct)
    stop = dt.datetime.now()
    logger.debug("Init atoms completed in %ius" % ((stop-start).microseconds))
    return atoms, temperature, pdb

def make_nonbond_pairs(nbCutoff, atoms, spring_constant):
    """This function computes the distance between all atom pairs to deterimine
       if they fall within the distance cutoff for being a non-bond pair. If
       so, a nonbonded Bond is initialized and Atom attributes are updated to
       reflect the change

       @param nbCutoff  float, max distance allowed for a nonbonded pair
       @param atoms     dict< atom_ID : Atom >
       @param spring_constant float, representing the stiffness of bonded 
                              atom bonds
    """
    start = dt.datetime.now()
    ct = 0 
    # Track tested pairs to decrease number of steps
    tested = {}
    # Iterate over all atom pairs and initialize nonbond Bonds if meet criteria
    for atom in atoms.values():
        for ID, test_atom in atoms.items():
            # Non-distance-based cases in which Bond should not be formed
            if ID == atom.ID: 
                continue
            if (ID in atom.atoms['bonded']) or (ID in atom.atoms['nonbonded']): 
                continue
            if (tested.has_key(atom.ID)) and (ID in tested[atom.ID]):
                continue
            # See if nonbond distance criteria is met, form Bond if so
            dist = atom.get_distance(test_atom)
            if (dist - nbCutoff) > EPSILON:
                # Too far, keep track of attempted formation
                if tested.has_key(ID): tested[ID].add(atom.ID)
                else:                  tested[ID] = set([atom.ID])
            else:
                atom.init_bond(test_atom, "nonbonded", spring_constant, dist)
                ct += 1

    logger.info("%i non-bonded pair 'Bonds' initialized" % ct)
    stop = dt.datetime.now()
    logger.debug("Make nonbond pairs done in %ius" % ((stop-start).microseconds))
    return

def update_velocities(atom_dict, delta_t, KE):
    """This function updates Atom velocities for all Atoms in atom_dict for a 
       specific time step dt. It returns the total KE for all atoms if specified

       @param atom_dict dict< atom_ID, Atom >
       @param dt        float, the time step relative to the last acceleration
                        update
       @param KE        bool, whether KE should be calculated and returned
       @return          float< total KE > if KE = True else nothing returned
    """
    start = dt.datetime.now()
    total_KE = 0
    for atom in atom_dict.values():
        total_KE += atom.update_velocity(delta_t, KE)
    stop = dt.datetime.now()
    logger.debug("Vel update done in %ius" % (stop-start).microseconds)
    if KE: return total_KE
    return

def update_positions(atom_dict, delta_t):
    """This function updates Atom positions for all Atoms in atom_dict for a
       specified time step dt.

       @param atom_dict dict< atom_ID, Atom >
       @param dt        float, the time step over which to update Atom position
    """
    start = dt.datetime.now()
    for atom in atom_dict.values():
        atom.update_position(delta_t)
    stop = dt.datetime.now()
    logger.debug("Position update done in %ius" % ((stop-start).microseconds))
    return

def update_PEs(atom_dict, curr_time):
    """This function computes the total bonded and nonbonded potential energy
       for all interacting (bonded/non-bonded) atom pairs. 
    """
    start = dt.datetime.now()
    #updates = {"bonded" : 0, "nonbonded" : 0} # for book keeping
    PEs     = {"bonded" : 0, "nonbonded" : 0}
    for atom in atom_dict.values():
        for bond_type, bond_set in atom.bonds.items():
            for bond in bond_set:
                if bond.time_updated["PE"]  == curr_time: continue # Don't ct 2x
                PEs[bond_type]     += bond.get_PE(curr_time)
                #updates[bond_type] += 1

    #out = ", ".join(["%s\t%i" % (btype, num) for btype, num in updates.items()])
    #logger.debug("# of PEs updated @ time %.4f: %s" % (curr_time, out))
    stop = dt.datetime.now()
    logger.debug("PE update done in %ius" % ((stop-start).microseconds))
    return PEs

def update_forces_and_accelerations(atoms, curr_time):
    """This function iterates over all atoms and updates their forces and
       accelerations for the current time.
       @param atoms         dict< atom_ID, Atom >
       @param curr_time     The current simulation time
    """
    start = dt.datetime.now()

    #print "start atom 3 force %s" % str(atoms[3].force)
    #print "start atom 3-4 dist %s, %s" % (str(atoms[3].get_distance(atoms[4], False)),str(atoms[3].get_distance(atoms[4])))
    #print "start atom 4-3 dist %s, %s" % (str(atoms[4].get_distance(atoms[3], False)),str(atoms[4].get_distance(atoms[3])))
    # Iterate over all atoms, first updating Force and using that to 
    # to update acceleration
    for atom in atoms.values():
        atom.update_force(curr_time)
        atom.update_acceleration()

    #print "start atom 3 force %s" % str(atoms[3].force)
    #print "start atom 3-4 dist %s, %s" % (str(atoms[3].get_distance(atoms[4], False)),str(atoms[3].get_distance(atoms[4])))
    #print "start atom 4-3 dist %s, %s" % (str(atoms[4].get_distance(atoms[3], False)),str(atoms[4].get_distance(atoms[3])))
    stop = dt.datetime.now()
    logger.debug("F/accel update done in %ius" % ((stop-start).microseconds))
    return

def init_outfiles(args, temp, pdb, atoms):
    """This function initializes the file connections for the two output files 
       (.erg and .rvc). It writes headers and returns the file handles.
    
       @param args      argparse argument list, used for header information
       @param temp      Temperature of simulation, from input file
       @param atoms     dict< atom_ID : Atom >
       @return fhandle_rvc, fhandle_erg
    """
    # Initialize files
    fh_rvc    = open(args.out + "_out.rvc", "w")
    fh_erg    = open(args.out + "_out.erg", "w")
    track_tup = init_track_file(args, atoms) if args.track != None else False

    # Write headers
    fh_erg.write("# step\tE_k\tE_b\tE_nB\tEtot\n")
    rvc_hd = "# %s: kB=%s kN=%s nbCutoff=%.2f dt=%.4f  mass=%.1f  T=%.1f\n" % \
             (pdb, args.kB, args.kN, args.nbCutoff, args.dt, args.m, temp)
    fh_rvc.write(rvc_hd)
    # We also have to copy the initial .rvc data to the ouput .rvc
    for atom in atoms.values(): fh_rvc.write( str(atom) )
    
    logger.info("Opened output file streams: %s, %s" % (fh_rvc.name, 
                                                        fh_erg.name) )
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

    # Initialize atoms and bond / non-bond pairs
    atoms, temperature, pdb = init_atoms(args.iF, args.m, args.kB)
    make_nonbond_pairs(args.nbCutoff, atoms, args.kN)

    # Initialize the output files, which are written to every few steps
    fh_rvc, fh_erg, track_tup = init_outfiles(args, temperature, pdb, atoms)

    # The time step loop
    time = 0
    for time_step in range(1, args.n + 1): # make time 1-based idx
        time += args.dt
       
        # Catch OverFlowErrors
        try:
            # The Velocity Verlet algorithm
            update_velocities(atoms, args.dt, KE=False)     # 1/2*dt velocity
            update_positions(atoms,  args.dt)
            PE = update_PEs(atoms, time)
            update_forces_and_accelerations(atoms, time)
            KE = update_velocities(atoms, args.dt, KE=True) # full dt velocity

        except FloatingPointError:
            logger.warning(\
                "Overflow error at time step %i (time=%.4f)" % (time_step, time))
            break
        
        t=open("test_old","w")
        for atom in atoms.values():
            t.write("%.12f, %.12f, %.12f\n" % (atom.velocity[0], atom.velocity[1], atom.velocity[2]))

        
        #print KE, PE
        # Track energies
        energies["KE"].append(KE)
        energies["PEb"].append(PE["bonded"])
        energies["PEnb"].append(PE["nonbonded"])
        energies["tot"].append(KE + PE["bonded"] + PE["nonbonded"])

        if time_step % WRITE_FREQ == 0:
            write_output(fh_rvc, fh_erg, track_tup, atoms, time_step, PE, KE)
        if time_step % 100 == 0:
            logger.info("Step %i of %i complete ..." % (time_step, args.n))
    t.close()
    fh_rvc.close()
    fh_erg.close()
    if track_tup: track_tup[1].close()
    if args.plot: plot_energies(args.out, energies, args.dt)
    stop = dt.datetime.now()
    logger.info("Simulation complete in %.1fs. Exiting!" % (stop-start).seconds)

#...............................................................................
#   Unit tests
def test_all():
    logger.warning("Running all unit tests ...\n")
    test_atoms_and_bonds()
    test_atom_str()
    test_make_nonbond_pairs()
    test_update_params()
    logger.warning("All tests passed, returning.")

def make_atoms():
    atoms, m = {}, DEFAULT_MASS

    atoms[1] = Atom(re.findall(REGEX_INPUT, "1 1 1 1 0.1 0.01 0.001 2 3"), m)
    atoms[2] = Atom(re.findall(REGEX_INPUT, "2 2 2 2 0.2 0.02 0.002 1 3"), m)
    atoms[3] = Atom(re.findall(REGEX_INPUT, "3 3 3 3 0.3 0.03 0.003 1 2"), m)
    atoms[4] = Atom(re.findall(REGEX_INPUT, "4 4 4 4 4 4 4 5"),   m)
    atoms[5] = Atom(re.findall(REGEX_INPUT, "5 5 5 5 10 10 10 4"),         m)
    atoms[6] = Atom(re.findall(REGEX_INPUT, "6 60 60 60 100 100 100"),        m)

    for atom in atoms.values():
        atom.init_bonds(atoms, "bonded", DEFAULT_KB)
   
    return atoms

def test_atoms_and_bonds():
    logger.warning("Testing Atom and Bond initiation and methods ...")
    atoms = make_atoms()

    assert atoms[1].atoms['bonded'] == set([2,3])
    assert atoms[2].atoms['bonded'] == set([1,3])
    assert atoms[3].atoms['bonded'] == set([1,2])
    assert atoms[4].atoms['bonded'] == set([5])
    assert atoms[5].atoms['bonded'] == set([4])
    assert atoms[6].atoms['bonded'] == set()

    assert (atoms[1].get_distance(atoms[2]) - np.sqrt(3)) < EPSILON 
    assert (atoms[2].get_kinetic_energy() - \
            sum(0.5*DEFAULT_MASS*np.array([0.2, 0.02, 0.002])**2)) < EPSILON

    logger.warning("Tests passed.\n")
    return

def test_atom_str():
    logger.warning("Testing atom string ...")
    atoms = make_atoms()
    assert str(atoms[1]) == "1\t1.0000\t1.0000\t1.0000\t0.1000\t0.0100\t0.0010\t2\t3\n"
    assert str(atoms[6]) == "6\t60.0000\t60.0000\t60.0000\t100.0000\t100.0000\t100.0000\t\n"
    logger.warning("Tests passed.\n")
    return   

def test_make_nonbond_pairs():
    logger.warning("Testing make_nonbond_pairs() ...")
    atoms = make_atoms()

    make_nonbond_pairs(0, atoms, DEFAULT_KN)
    assert atoms[1].bonds["nonbonded"] == set()
    
    make_nonbond_pairs(10, atoms, DEFAULT_KN)
    assert 4 in atoms[1].atoms["nonbonded"] 
    assert 1 in atoms[4].atoms["nonbonded"] 
    assert 1 not in atoms[2].atoms['nonbonded']
    assert atoms[6].atoms['nonbonded'] == atoms[6].atoms['bonded'] == set()

    make_nonbond_pairs(1000, atoms, DEFAULT_KN)
    assert 1 in atoms[6].atoms["nonbonded"]
    assert 6 in atoms[4].atoms["nonbonded"]

    logger.warning("Tests passed.\n")
    return

def test_update_params():
    logger.warning("Testing update_params() ...")
    atoms = make_atoms()

    curr_velocity = atoms[1].velocity
    atoms[1].update_velocity(100, KE=False)
    assert curr_velocity[0] == atoms[1].velocity[0]

    atoms[5].update_position(10) # 5dist + 10 dist/s * 10s = 105
    atoms[5].update_force(0)
    new_length = np.sqrt(3*101**2)
    new_force  = DEFAULT_KB*( new_length - np.sqrt(3) )
    assert atoms[5].coord[0] == atoms[5].coord[1] == atoms[5].coord[2] == 105

    for bond in atoms[5].bonds['bonded']:
        force_rel_to_4 = bond.get_force(atoms[4],0)[0]
        force_rel_to_5 = bond.get_force(atoms[5],0)[0]
        assert (force_rel_to_4 - (new_force*101/new_length)) < EPSILON
        assert  force_rel_to_4 == - force_rel_to_5

    PE = update_PEs(atoms, 10)

    logger.warning("Tests passed.\n")
    return

#...............................................................................
#   Flow determinant
if __name__ == "__main__":
    os.chdir(os.getcwd())
    args   = prsr.parse_args()
    if args.out == None:
        args.out = re.findall(REGEX_BASENAME, args.iF)[0] 
    logger = get_logger(args.logger_level)
    if args.tests_only: test_all()
    else: main()
