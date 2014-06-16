# This script computes the FEATURE score for the abbreviated 1FW4 protein
# fragment.  It reads in the pdb file from the local directory and takes in a
# .rcv file that is the result of a molecular dynamics simulation run and then
# uses that extract the feature score for that position.

# python computeFEATUREScore1FW4.py <infile> <outfile>



center1 = ( 5.514,  21.470,  -5.834)
center2 = ( 6.707,  32.045,  -1.096)

siteArray = [{'CYS': 0.002, 'ASP': 0.002, 'SER': 0.002, 'GLN': 0.002, 'LYS': 0.002, 'ILE': 0.002, 'PRO': 0.002, 'THR': 0.002, 'PHE': 0.002, 'ALA': 0.002, 'GLY': 0.002, 'HIS': 0.002, 'GLU': 0.002, 'LEU': 0.002, 'ARG': 0.002, 'TRP': 0.002, 'VAL': 0.002, 'ASN': 0.002, 'TYR': 0.002, 'MET': 0.002}, {'CYS': 0.002, 'ASP': 0.002, 'SER': 0.002, 'GLN': 0.002, 'LYS': 0.002, 'ILE': 0.002, 'PRO': 0.002, 'THR': 0.002, 'PHE': 0.002, 'ALA': 0.002, 'GLY': 0.002, 'HIS': 0.002, 'GLU': 0.002, 'LEU': 0.002, 'ARG': 0.002, 'TRP': 0.002, 'VAL': 0.002, 'ASN': 0.002, 'TYR': 0.002, 'MET': 0.002}, {'CYS': 0.002, 'ASP': 0.88700000000000001, 'SER': 0.113, 'GLN': 0.094, 'LYS': 0.094, 'ILE': 0.5, 'PRO': 0.02, 'THR': 0.075999999999999998, 'PHE': 0.075999999999999998, 'ALA': 0.075999999999999998, 'GLY': 0.039, 'HIS': 0.057000000000000002, 'GLU': 0.02, 'LEU': 0.13100000000000001, 'ARG': 0.039, 'TRP': 0.02, 'VAL': 0.20499999999999999, 'ASN': 0.057000000000000002, 'TYR': 0.075999999999999998, 'MET': 0.002}, {'CYS': 0.002, 'ASP': 0.92400000000000004, 'SER': 0.371, 'GLN': 0.094, 'LYS': 0.35199999999999998, 'ILE': 0.223, 'PRO': 0.02, 'THR': 0.38900000000000001, 'PHE': 0.094, 'ALA': 0.223, 'GLY': 0.75800000000000001, 'HIS': 0.02, 'GLU': 0.13100000000000001, 'LEU': 0.094, 'ARG': 0.186, 'TRP': 0.002, 'VAL': 0.13100000000000001, 'ASN': 0.70299999999999996, 'TYR': 0.02, 'MET': 0.039}, {'CYS': 0.002, 'ASP': 0.14899999999999999, 'SER': 0.094, 'GLN': 0.039, 'LYS': 0.02, 'ILE': 0.094, 'PRO': 0.002, 'THR': 0.094, 'PHE': 0.16800000000000001, 'ALA': 0.113, 'GLY': 0.094, 'HIS': 0.002, 'GLU': 0.86899999999999999, 'LEU': 0.094, 'ARG': 0.002, 'TRP': 0.02, 'VAL': 0.094, 'ASN': 0.039, 'TYR': 0.075999999999999998, 'MET': 0.094}]
nonSiteArray = [{'CYS': 0.002, 'ASP': 0.002, 'SER': 0.002, 'GLN': 0.002, 'LYS': 0.002, 'ILE': 0.002, 'PRO': 0.002, 'THR': 0.027, 'PHE': 0.002, 'ALA': 0.002, 'GLY': 0.002, 'HIS': 0.002, 'GLU': 0.027, 'LEU': 0.002, 'ARG': 0.002, 'TRP': 0.002, 'VAL': 0.027, 'ASN': 0.002, 'TYR': 0.027, 'MET': 0.002}, {'CYS': 0.002, 'ASP': 0.027, 'SER': 0.027, 'GLN': 0.076999999999999999, 'LYS': 0.027, 'ILE': 0.051999999999999998, 'PRO': 0.002, 'THR': 0.027, 'PHE': 0.027, 'ALA': 0.027, 'GLY': 0.10199999999999999, 'HIS': 0.027, 'GLU': 0.051999999999999998, 'LEU': 0.10199999999999999, 'ARG': 0.002, 'TRP': 0.002, 'VAL': 0.027, 'ASN': 0.076999999999999999, 'TYR': 0.002, 'MET': 0.027}, {'CYS': 0.027, 'ASP': 0.152, 'SER': 0.051999999999999998, 'GLN': 0.051999999999999998, 'LYS': 0.10199999999999999, 'ILE': 0.051999999999999998, 'PRO': 0.152, 'THR': 0.10199999999999999, 'PHE': 0.10199999999999999, 'ALA': 0.076999999999999999, 'GLY': 0.076999999999999999, 'HIS': 0.051999999999999998, 'GLU': 0.027, 'LEU': 0.152, 'ARG': 0.027, 'TRP': 0.027, 'VAL': 0.127, 'ASN': 0.051999999999999998, 'TYR': 0.027, 'MET': 0.027}, {'CYS': 0.002, 'ASP': 0.076999999999999999, 'SER': 0.152, 'GLN': 0.027, 'LYS': 0.10199999999999999, 'ILE': 0.152, 'PRO': 0.10199999999999999, 'THR': 0.152, 'PHE': 0.051999999999999998, 'ALA': 0.251, 'GLY': 0.152, 'HIS': 0.002, 'GLU': 0.27600000000000002, 'LEU': 0.17699999999999999, 'ARG': 0.10199999999999999, 'TRP': 0.051999999999999998, 'VAL': 0.20100000000000001, 'ASN': 0.027, 'TYR': 0.127, 'MET': 0.027}, {'CYS': 0.027, 'ASP': 0.152, 'SER': 0.20100000000000001, 'GLN': 0.10199999999999999, 'LYS': 0.27600000000000002, 'ILE': 0.17699999999999999, 'PRO': 0.076999999999999999, 'THR': 0.32600000000000001, 'PHE': 0.22600000000000001, 'ALA': 0.17699999999999999, 'GLY': 0.127, 'HIS': 0.10199999999999999, 'GLU': 0.22600000000000001, 'LEU': 0.27600000000000002, 'ARG': 0.152, 'TRP': 0.10199999999999999, 'VAL': 0.42499999999999999, 'ASN': 0.10199999999999999, 'TYR': 0.20100000000000001, 'MET': 0.17699999999999999}]


backboneD = {128: 'VAL', 2: 'PHE', 294: 'ASP', 388: 'PHE', 261: 'ILE', 135: 'MET', 13: 'ASP', 143: 'THR', 21: 'LYS', 150: 'ASN', 280: 'GLU', 158: 'LEU', 415: 'MET', 289: 'ALA', 38: 'GLY', 423: 'MET', 42: 'ASN', 302: 'ILE', 50: 'GLY', 179: 'LYS', 30: 'ASP', 54: 'TYR', 188: 'LEU', 330: 'GLY', 318: 'GLY', 406: 'GLN', 66: 'ILE', 196: 'THR', 310: 'ASP', 74: 'SER', 203: 'ASP', 322: 'ASP', 334: 'GLN', 269: 'ARG', 80: 'ALA', 211: 'GLU', 85: 'ALA', 343: 'VAL', 90: 'GLU', 399: 'VAL', 220: 'GLU', 350: 'ASN', 229: 'VAL', 99: 'LEU', 166: 'GLY', 358: 'TYR', 107: 'ARG', 236: 'ASP', 253: 'MET', 370: 'GLU', 244: 'GLU', 118: 'HIS', 379: 'GLU', 170: 'GLU'}


import sys, os, string, math

aminoL = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

shellL = [(0, 1.5), (1.5, 3), (3, 4.5), (4.5, 6), (6, 7.5)]

sqdF = lambda x, y: 1.0*(x-y)*(x-y)

def eucDF(a, b):
    "Returns the Euclidean distance between tuples of same length"
    if len(a) != len(b) and len(a) > 0:
        print >>sys.stderr, "Asked to compute distance between two different dimension vectors %s %s" % (repr(a), repr(b))
        return None
    else:
        dist = math.sqrt(sum(map(sqdF, a, b)))
        return dist



class Protein(object):

    def __init__(self):
        self.aaTypeL = [] 
        self.aaPosL = []


    def parse_rvc_line(self, line):
        itemL = line.split()
        index = string.atoi(itemL[0])
        x = string.atof(itemL[1]) *10
        y = string.atof(itemL[2])* 10
        z = string.atof(itemL[3]) *10
        self.add_aa(index, x, y, z)

    def add_aa(self, index, x, y, z):
        if backboneD.has_key(index):
            aaType = backboneD[index]
            self.aaTypeL.append(aaType)
            self.aaPosL.append( (x, y, z))


    def __repr__(self):
        repL = []
        for i in range(0, len(self.aaPosL)):
            aa = self.aaTypeL[i]
            pos = self.aaPosL[i]
            rline = "%s %5.3f %5.3f %5.3f" % (aa, pos[0], pos[1], pos[2])
            repL.append(rline)
        return(string.join(repL, "\n"))


    def shellContents(self, ri, rout, center):
        "This function returns a hash of the amino acids in the shell defined by the inner and outer radius and the position; returns only a single response if it is there (possible could return actual counts if done another way)"
        aaShellContentsD = {}
        for i in range(0, len(self.aaPosL)):
            dist = eucDF(self.aaPosL[i], center)
            if dist >= ri and dist < rout:
                aaShellContentsD[self.aaTypeL[i]] = None
        return aaShellContentsD

    def scoreConfiguration(self, center):
        score = 0
        for s in range(0, len(shellL)):
            ri, ro = shellL[s]
            shellAAD = self.shellContents(ri, ro, center)
            for aa in aminoL:
                pas = siteArray[s][aa]
                pans = nonSiteArray[s][aa]
                if shellAAD.has_key(aa):
                    score += (math.log(pas) - math.log(pans))
                else:
                    score += (math.log(1-pas) - math.log(1-pans))

        return score       


def main():
    i = 0
    inFile = open(sys.argv[1])
    outFile = open(sys.argv[2], "w")

    outFile.write("TimeStep\tCenter1\tCenter2\n")
    curConfig = None
    for line in inFile:
        if line[0] == "#":
            if curConfig:
                s1 = curConfig.scoreConfiguration(center1)
                s2 = curConfig.scoreConfiguration(center2)
                outFile.write("%d\t%.3f\t%.3f\n" % (i, s1, s2))
                curConfig = Protein()
                i += 10
            else:
                curConfig = Protein()
        else:
            if line:
                curConfig.parse_rvc_line(line)



if __name__=='__main__':
    main()
