#!/bin/sh
#$ -N ¿name2identify?
#$ -j y
#$ -cwd
#$ -pe 56cpn 56
####$ -l mf=15G
#$ -q IFC

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

module use /Dedicated/IFC/.argon/modules
module load asynch

mpirun -np ¿nprocess? asynch ¿global?

