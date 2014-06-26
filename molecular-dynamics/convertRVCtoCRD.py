#!/usr/bin/python
#
#
 
#author block
"""RVC to CRD file converter"""
__author__ = "Randy Radmer"
__version__ = "1.0"
#
 
#import block
import sys, os
import getopt
#

def readRVC(inFilename):
    fIn = open(inFilename)
    coords=[]
    header = ''
    headerfound = 0
    for line in fIn:
	if (line[0] == '#') & (headerfound == 0):
		header = line.strip()
		headerfound = 1
        if line.strip()=='' or line[0]=='#': continue
        items=line.split()
        coords.append([float(items[1]),
                       float(items[2]),
                       float(items[3])])
    fIn.close()
    return coords, header


def writeCRD(outFilename, title, coords):
    fOut = open(outFilename, 'w')
    fOut.write('%s\n' % title)
    line=''
    for x, y, z in coords:
        line+='%8.3f' % (10*x)
        if len(line)>=80:
            fOut.write('%s\n' % line)
            line=''
        line+='%8.3f' % (10*y)
        if len(line)>=80:
            fOut.write('%s\n' % line)
            line=''
        line+='%8.3f' % (10*z)
        if len(line)>=80:
            fOut.write('%s\n' % line)
            line=''
    if len(line)>0: fOut.write('%s\n' % line)
    fOut.close()

def main():
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    coords, header = readRVC(inFile)
    writeCRD(outFile, header, coords)
    

if __name__=='__main__':
    main()
