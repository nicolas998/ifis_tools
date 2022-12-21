# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jupyter_scripts//ipynb,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # asynch_manager
#
# SEt of tools to get data from asynch and set the variables to run the model in an easy way from python.

import pandas as pd 
from datetime import timezone, datetime
import os 
import fileinput
import numpy as np 
import auxiliar as aux

# # Global variables 

# +
# 190 original parameters
Param190 = [6, 0.75, 0.33, -0.20, 0.50, 0.1, 2.2917e-5]
Param254 = [0.33, 0.2, -0.1, 0.02, 2.0425e-6, 0.02, 0.5, 0.10, 0.0, 99.0, 3.0, 0.75]

# 190 global base format 
f = open('190BaseGlobal.gbl','r')
Global190 = f.readlines()
f.close()

# 190 global base format 
f = open('254BaseGlobal.gbl','r')
Global254 = f.readlines()
f.close()


# -

# # ASYNCH tools  
#
# ## Asynch results reader 

class ASYNCH_results:
    
    def __init__(self,path):
        '''Reads a .dat file produced by ASYNCH'''
        #Open the file 
        f = open(path,'r')
        self.dat = f.readlines()
        f.close()

    def ASYNCH_dat2Serie(self, linkID, date1, freq):
        '''From the data readed with ASYNCH_read_dat reads the serie 
        corresponding to the linkID and retrieves a pandas Series object.
        Parameters:
            - linkID: the number of the linkID to obtain.
            - date1: the start date of the simulation.
            - freq: time frequency of the data (ej. 15min).
        Returns: 
            - Pandas series with the simulated data.'''
        #Number of records
        self.Nlinks = int(self.dat[0])
        self.Nrec = int(self.dat[3].split()[1])
        Start = np.arange(3, (self.Nrec+2)*self.Nlinks, self.Nrec+2)
        End = Start + self.Nrec
        #Search the IDS
        self.Ids = [l.split()[0] for l in self.dat[3::self.Nrec+2]]
        #Search the position of the usgs station to be analyzed.
        PosStat = self.Ids.index(str(linkID))
        #Retrieve the data.
        Data = np.array([float(l) for l in self.dat[Start[PosStat]+1:End[PosStat]+1]])
        #return self.dat[Start[PosStat]+1:End[PosStat]+1]
        Dates = pd.date_range(date1, periods=self.Nrec, freq=freq)
        return pd.Series(Data, Dates)


# ## Asynch project manager

# +
class ASYNCH_project:
    
    def __init__(self, name = None, date1 = None, date2 = None, linkID = None, initial = 'Initial',
        unix1 = None, unix2 = None, gblBase = '190BaseGlobal.gbl',
        initialBase = '190BaseInitial.dbc', parameters = None, links2save = 'PeakFlows.sav'):
        '''ASYNCH project constructor, this class creates the folders and files 
        for an ASYNCH run, and also eventually runs asynch from python (this is not
        a warp from C)
        Parameters:
            - name: The name of the project with the path to save it
            - date1: the initial date of the simulation (YYYY-MM-DD HH:MM)
            - date2: the initial date of the simulation (YYYY-MM-DD HH:MM)
            - linkID: the number of the output link to make the simulation.
            - glbOut: rute and name of the output global file for asynch.
            - output: name of the file with the outputs.
            - peakflow: name of the file containing the links where to save.
        '''
        #Define initial values for the asynch project.
        if os.path.isdir(name):
            print('Warning: Project already exists, it will be loaded (not implemented)')
        else:
            #Define parameters of the new project
            self.path = os.getcwd() + '/' + name
            self.date1 = date1
            self.date2 = date2
            self.linkID = linkID
            self.initial = initial
            self.unix1 = unix1
            self.unix2 = unix2
            self.gblBase = gblBase
            self.initialBase = initialBase
            self.parameters = parameters
            self.links2save = links2save
            #Creates the new project on a file.
            self.__ASYNC_createProject__()

    def __ASYNC_createProject__(self):
        '''Creates a new directory for asynch runs.
        Parameters:
            - pathProject: path to the new folder conaining the new asynch project
        Return:
            - Creates a new folder that contains sub-folders and files required 
                for the asynch run'''
        #Creates the main folder of the proyect and inputs and outputs.
        aux.__make_folder__(self.path)
        self.path_in = self.path + '/AsynchInputs'
        aux.__make_folder__(self.path_in)
        self.path_out = self.path + '/AsynchOutputs'
        aux.__make_folder__(self.path_out)
        self.path_scratch = self.path_out+'/Scratch'
        aux.__make_folder__(self.path_scratch)
    
    def __ASYNCH_setRunFile__(self, OutRun = 'Run.sh',runBase = '190BaseRun.sh'):
        '''Writes the runfile to the AsynchInput directory of the project'''
        #Copy the run file from the base 
        self.path_in_run = self.path_in + '/' + OutRun
        comand = 'cp '+runBase+' '+self.path_in_run
        os.system(comand)
        # Filename to write the new runfile
        filename = self.path_in_run
        #Dictionary with the words to search and change in the new runfile
        DicToReplace = {'glbFile':{'to_search': '¿global?', 'to_put': self.path_in_global}}
        #Changing the runfile.    
        for k in DicToReplace:            
            with fileinput.FileInput(filename, inplace=True) as file:
                for line in file:                    
                    text_to_search = DicToReplace[k]['to_search']
                    replacement_text = str(DicToReplace[k]['to_put'])
                    print(line.replace(text_to_search, replacement_text), end='')
            
    def ASYNCH_setProject(self, GlobalName='GlobalFile.gbl',Links2SaveName = 'Links2Save.sav',
        OutStatesName = 'OutputStates.dat', InitialName = 'InitialFile.dbc', setRunFile = True):
        '''Edit the global file for 190 asynch run.
        Parameters:
            - date1: the initial date of the simulation (YYYY-MM-DD HH:MM)
            - date2: the initial date of the simulation (YYYY-MM-DD HH:MM)
            - linkID: the number of the output link to make the simulation.
            - glbOut: rute and name of the output global file for asynch.
            - output: name of the file with the outputs.
            - peakflow: name of the file containing the links where to save.
        Optional:
            - parameters: running parameters for ASYNCH model.
            - unix1: start time of the execution.
            - unix2: end time of the execution.
            - glbBase: rute and name of the base globl file used for the excecutions.
        Outputs:
            This function writes a gbl file where gblOut indicates.'''
        # Copy the base global to a glbOut
        self.path_in_global = self.path_in + '/'+ GlobalName
        comand = 'cp '+self.gblBase+' '+self.path_in_global
        os.system(comand)
        #Copy the links2save file 
        self.path_in_links2save = self.path_in + '/'+ Links2SaveName
        comand = 'cp '+self.links2save+' '+self.path_in_links2save
        os.system(comand)
        #Set of the initial file for that link 
        self.path_in_initial = self.path_in+ '/'+ InitialName
        aux.__ASYNCH_initialFile__(self.path_in_initial,
            self.linkID, self.initialBase)
        #Set the name of the file with the output of the streamflow
        self.path_out_states = self.path_out+'/'+ OutStatesName
        # Unix time are equal to date
        if self.unix1 is None:
            self.unix1 = aux.__datetime2unix__(self.date1)
        textUnix1 = '%d' % self.unix1
        if self.unix2 is None:
            self.unix2 =aux.__datetime2unix__(self.date2)
        textUnix2 = '%d' % self.unix2
        # Parameters 
        if self.parameters is None:
            self.Param = [str(i) for i in Param190]
        else:
            self.Param = [str(i) for i in self.parameters]
        Param = ' '.join(self.Param)+'\n'
        # Replace parameters in the global file        
        DicToreplace = {'date1':{'to_search': '¿date1?', 'to_put': self.date1},
            'date2':{'to_search': '¿date2?', 'to_put': self.date2},
            'unix1':{'to_search': '¿unix1?', 'to_put': textUnix1},
            'unix2':{'to_search': '¿unix2?', 'to_put': textUnix2},
            'linkID':{'to_search': '¿linkID?', 'to_put': self.linkID},
            'parameters':{'to_search': '¿Parameters?', 'to_put': Param},
            'output':{'to_search': '¿output?', 'to_put': self.path_out_states},
            'peakflow':{'to_search': '¿peakflow?', 'to_put': self.path_in_links2save},
            'initial':{'to_search': '¿initial?', 'to_put': self.path_in_initial},
            'scratch':{'to_search': '¿scratch?', 'to_put': self.path_scratch}}
        # Replacement in the document.
        filename = self.path_in_global
        for k in DicToreplace:            
            with fileinput.FileInput(filename, inplace=True) as file:
                for line in file:                    
                    text_to_search = DicToreplace[k]['to_search']
                    replacement_text = str(DicToreplace[k]['to_put'])
                    print(line.replace(text_to_search, replacement_text), end='')
        # SEt the runfile for the project.
        if setRunFile:
            self.__ASYNCH_setRunFile__()

   
# -


