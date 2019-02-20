#!/bin/sh
#$ -N out_190b_2013_s4
#$ -j y
#$ -cwd
#$ -pe 56cpn 56
####$ -l mf=16G
#$ -q IFC

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

module use /Dedicated/IFC/.argon/modules
module load asynch

mpirun -np 56 asynch ¿global?

