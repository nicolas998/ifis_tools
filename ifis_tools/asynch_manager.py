# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: jupyter_scripts//ipynb,ifis_tools//py
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
from ifis_tools import auxiliar as aux

# # Global variables 

# +
#SEt of parameters for the different configurations of the model 
Parameters = {'190': [6, 0.75, 0.33, -0.20, 0.50, 0.1, 2.2917e-5],
    '254':[12, 0.33, 0.2, -0.1, 0.02, 2.0425e-6, 0.02, 0.5, 0.10, 0.0, 99.0, 3.0, 0.75]}

#Path of the file 
Path = __file__.split('/')
Path = '/'.join(Path[:-1])+'/'

#Read the global files that are used to generate new globals
Globals = {}
for g in ['190','254']:    
    # 190 global base format 
    f = open(Path+g+'BaseGlobal.gbl','r')
    Globals.update({g:f.readlines()}) 
    f.close()


# -

# # ASYNCH tools  
#
# ## Asynch results reader 

def UpdateGlobal(filename, DictUpdates):
    #Iterate in the updates keys
    for k in DictUpdates:            
        with fileinput.FileInput(filename, inplace=True) as file:
            for line in file:                    
                print(line.replace(DictUpdates[k]['old'], DictUpdates[k]['new']), end='')
        

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

class ASYNCH_project:
    
    def __init__(self, path_in, path_out, name = None, date1 = None, date2 = None, 
        linkID = 0, unix1 = None, unix2 = None, model = '190',
        parameters = None, links2save = 'ControlPoints.sav'):
        '''ASYNCH project constructor, this class creates the folders and files 
        for an ASYNCH run, and also eventually runs asynch from python (this is not
        a warp from C)
        Parameters:
            - path_in: path to store: Global file, initial file and run file.
            - path_out: path to store: hydrographs.
            - name: The name of the project with the path to save it
            - date1: the initial date of the simulation (YYYY-MM-DD HH:MM)
            - date2: the initial date of the simulation (YYYY-MM-DD HH:MM)
            - linkID: the number of the output link to make the simulation.
            - glbOut: rute and name of the output global file for asynch.
            - output: name of the file with the outputs.
            - peakflow: name of the file containing the links where to save.
        '''
        #Paths and name of the project
        self.path_in = path_in
        self.path_out = path_out
        self.name = name
        # Dates and linkID of the outlet 
        self.date1 = date1
        self.date2 = date2
        self.linkID = linkID
        self.unix1 = unix1
        self.unix2 = unix2
        #Model to use, parameters and where to save
        self.model = model
        self.parameters = [str(i) for i in Parameters[model]]
        self.links2save = links2save
        
    def ASYNCH_setRunFile(self, runBase = 'BaseRun.sh', path2gbl = None, nprocess = 28):
        '''Writes the runfile to the AsynchInput directory of the project'''
        #Copy the run file from the base 
        self.path_in_run = self.path_in + '/' + self.name + '.sh'
        comand = 'cp '+Path+runBase+' '+self.path_in_run
        os.system(comand)
        # Filename to write the new runfile
        filename = self.path_in_run
        #Dictionary with the words to search and change in the new runfile
        if path2gbl is None:
            DicToReplace = {'glbFile':{'to_search': '¿global?', 'to_put': self.name+'.gbl'}}
        else:
            DicToReplace = {'glbFile':{'to_search': '¿global?', 'to_put': path2gbl + self.name+'.gbl'}}
        DicToReplace.update({'nProcess':{'to_search': '¿nprocess?', 'to_put': str(nprocess)}})
        DicToReplace.update({'name2identify':{'to_search': '¿name2identify?', 'to_put': 'r'+self.name}})
        #Changing the runfile.    
        for k in DicToReplace:            
            with fileinput.FileInput(filename, inplace=True) as file:
                for line in file:                    
                    text_to_search = DicToReplace[k]['to_search']
                    replacement_text = str(DicToReplace[k]['to_put'])
                    print(line.replace(text_to_search, replacement_text), end='')
            
    def ASYNCH_setGlobal(self, gblBase = 'BaseGlobal.gbl', Links2SaveName = 'ControlPoints.sav',
        OutStatesName = 'OutputStates.dat', createInitial = True, oldInitial = None,
        snapName = None, snapTime = None, createSav = False):
        '''Edit the global file for asynch run.
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
        self.path_in_global = self.path_in +  self.name + '.gbl'
        comand = 'cp '+Path+self.model+gblBase+' '+self.path_in_global
        os.system(comand)
        #Copy the links2save file 
        if createSav is False:
            self.path_in_links2save = self.path_in + Links2SaveName
            comand = 'cp '+Path+self.links2save+' '+self.path_in_links2save
            os.system(comand)
        else:
            self.path_in_links2save = self.path_in+'control.sav'
            f = open(self.path_in_links2save,'w')
            f.write('%s' % self.linkID)
            f.close()
        #Set of the initial file for that link 
        if createInitial:
            self.path_in_initial = self.path_in + self.name + '.dbc'
            self.__ASYNCH_setInitialFile__(self.path_in_initial,self.date1[:4],
                self.linkID)
        else:
            self.path_in_initial = oldInitial
        #Set the number of the initial depending on its extension 
        InitNumber = str(self.__ASYNCH_get_number(self.path_in_initial, whatfor='initial'))
        
        #Set the snapshot info
        if snapName is not None:
            Snap_flag = self.__ASYNCH_get_number(snapName, whatfor='snapshot')
            Snap_name = snapName
            if snapTime is not None:
                Snap_time = str(snapTime)
                if Snap_flag == 3:
                    Snap_flag = str(4)
            else:
                Snap_time = ''
            Snap_flag = str(Snap_flag)
        else:
            Snap_flag = 0; Snap_name = ''; Snap_time = ''
        
        #Set the name of the file with the output of the streamflow
        self.path_out_states = self.path_out + OutStatesName
        # Unix time are equal to date
        if self.unix1 is None:
            self.unix1 = aux.__datetime2unix__(self.date1) + 12*3600.
        textUnix1 = '%d' % self.unix1
        if self.unix2 is None:
            self.unix2 =aux.__datetime2unix__(self.date2) + 12*3600
        textUnix2 = '%d' % self.unix2
        # Parameters 
        Param = ' '.join(self.parameters)+'\n'
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
            'initial_flag': {'to_search': '¿initialflag?', 'to_put': InitNumber},
            'snapshot_flag': {'to_search': '¿snapflag?', 'to_put': Snap_flag},
            'snapshot_time': {'to_search': '¿snaptime?', 'to_put': Snap_time},
            'snapthot_name': {'to_search': '¿snapshot?', 'to_put': Snap_name},}
        # Replacement in the document.
        filename = self.path_in_global
        for k in DicToreplace:            
            with fileinput.FileInput(filename, inplace=True) as file:
                for line in file:                    
                    text_to_search = DicToreplace[k]['to_search']
                    replacement_text = str(DicToreplace[k]['to_put'])
                    print(line.replace(text_to_search, replacement_text), end='')
        
    def __ASYNCH_setInitialFile__(self, InitialOut, year, linkID):
        '''Set the dbc query for the initial conditions for asynch
        Parameters:
            - InitialOut: The path and name of the outlet initial file for the dbc.
            - linkID: the link in which is going to be placed the initial file.
            - InicialBase: Path to the basefile for the stablishment of the initial file.
        Results:
            - Writes the initial file at InitialOut.'''
        # Copy the initial file state in
        if linkID == 0:
            comando = 'cp '+Path+'BaseInitial.dbc'+' '+InitialOut
        else:
            comando = 'cp '+Path+'BaseInitial_link.dbc'+' '+InitialOut
        os.system(comando)
        #Dict with words to replace
        DicToreplace = {'link':{'to_search': '¿linkID?', 'to_put': linkID},
            'date':{'to_search': 'YYYY', 'to_put': year}}
        # Replace the linkID in the initial file so asynch knows.
        filename = InitialOut
        for k in DicToreplace:          
            with fileinput.FileInput(filename, inplace = True) as file:
                for line in file:
                    text_to_search = DicToreplace[k]['to_search']
                    replacement_text = str(DicToreplace[k]['to_put'])
                    print(line.replace(text_to_search, replacement_text), end='')
               
    def __ASYNCH_get_number(self, path, whatfor='initial'):
        '''Get the number to set into the globabl based on the extension name 
        of the file'''
        #Get the extention from the file 
        extension = os.path.splitext(path)[1]
        #Find the number for the case of init states
        if whatfor == 'initial':            
            if extension == '.ini':
                return 0
            elif extension == '.uini':
                return 1
            elif extension == '.rec':
                return 2
            elif extension == '.dbc':
                return 3
            elif extension == '.h5':
                return 4
        # Fing the number for the case of snapshot
        if whatfor == 'snapshot':
            if extension == '':
                return 0
            elif extension == '.rec':
                return 1
            elif extension == '.dbc':
                return 2
            elif extension == '.h5':
                return 3
        
# # Deprecated


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


