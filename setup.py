#!/usr/bin/env python
import os
from numpy.distutils.core import setup, Extension

#os.system('jupytext --set-formats jupyter_scripts//ipynb,ifis_tools//py --sync jupyter_scripts/*.ipynb')

setup(
    name='ifis_tools',
    version='0.1.7',
    author='Nicolas Velasquez G',
    author_email='nicolas.velasquezgiron@gmail.com',    
    packages=['ifis_tools'],
    package_data={'ifis_tools':['evap.mon','60XBaseGlobal.gbl','254BaseGlobal.gbl','190BaseGlobal.gbl','BaseInitial.dbc','BaseRun.sh','ControlPoints.sav','BaseInitial_link.dbc']},
    url='https://github.com/nicolas998/ifis_tools.git',
    license='LICENSE.txt',
    description='Watershed Modelling Framework',
    long_description=open('README.md').read(),
	)
