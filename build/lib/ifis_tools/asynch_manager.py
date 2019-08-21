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
from ifis_tools import database_tools as db 
from wmf import wmf
from string import Template
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
for g in ['190','254','60X']:    
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
        
class hlmModel:
    
    def __init__(self,linkid=None, path = None, ExtraParams = None):
        '''Depending on the linkid or in the path the class starts a table 
        to set up a new project fro HLM model.
            - linkid = number of link id to search for in the database.
            - path =  path to search for a WMF.SimuBasin project .nc
        Optional:
            -ExtraParams: List with the names of the extra params extracted from the database'''
        #Make an action depending on each case.
        if linkid is not None and path is None:
            self.Table = db.SQL_Get_WatershedFromMaster(linkid, ExtraParams)
            self.linkid = linkid
        elif path is not None and linkid is None:
            self.wmfBasin = wmf.SimuBasin(rute=path)
            self.wmfBasin.GetGeo_Cell_Basics()
            self.Table = cu.Transform_Basin2Asnych('/tmp/tmp.rvr',
                lookup='/tmp/tmp.look',
                prm='/tmp/tmp.prm') 

    def write_rvr(self, path):
        #conncet to the database
        con = db.DataBaseConnect(database='restore_res_env_92')
        #Query to ask for the link ids and the topology
        q = db.sql.SQL("WITH all_links(id) AS (SELECT link_id FROM public.env_master_km) \
         SELECT all_links.id,public.env_master_km.link_id FROM public.env_master_km,all_links \
         WHERE (all_links.id IN (SELECT nodeX.link_id FROM public.env_master_km AS nodeX, public.env_master_km AS parentX \
         WHERE (nodeX.left BETWEEN parentX.left AND parentX.right) AND parentX.link_id = "+str(self.linkid)+")) AND public.env_master_km.parent_link = all_links.id ORDER BY all_links.id")            
        self.topo = pd.read_sql(q, con)
        con.close()
        topo = self.topo.values.T
        #Convert the query to a rvr file 
        f = open(path,'w')
        f.write('%d\n\n' % topo.shape[1])
        #List = self.Table.index.tolist()
        for t in topo[1]:
            #List.index(t)
            f.write('%d'% t)
            p = np.where(topo[0] == t)[0]
            if len(p)>0:
                for i in p:
                    f.write(' %d' % topo[1][i])
                f.write('\n')
                f.write('%d\n\n' % p.size)
            else:
                f.write('\n0\n\n')    
        f.close()        
            
    def write_Global(self, path2global, model_uid = 604,
        date1 = None, date2 = None, rvrFile = None, prmFile = None, initialFile = None,
        rainType = 5, rainPath = None, evpFile = 'evap.mon', datResults = None, 
        controlFile = None, baseGlobal = None, noWarning = False):
        '''Creates a global file for the current project.
            - model_uid: is the number of hte model goes from 601 to 604.
            - date1 and date2: initial date and end date
            - rvrFile: path to rvr file.
            - prmFile: path to prm file.
            - initialFile: path to file with initial conditions.
            - rainType: number inficating the type of the rain to be used.
            - rainPath: path to the folder containning the binary files of the rain.
            - evpFile: path to the file with the values of the evp.
            - datResults: File where .dat files will be written.
            - controlFile: File with the number of the links to write.
            - baseGlobal: give the option to use a base global that is not the default'''
        #Open the base global file and creates tyhe template
        if baseGlobal is not None:
            f = open(baseGlobal, 'r')
            L = f.readlines()
            f.close()
        else:
            L = Globals['60X']
        t = []
        for i in L:
            t += i
        Base = Template(''.join(t))
        # Creates the default Dictionary.
        Default = {
            'model_uid' : model_uid,
            'date1': date1,
            'date2': date2,
            'rvrFile': rvrFile,
            'prmFile': prmFile,
            'initialFile': initialFile,
            'rainType': str(rainType),
            'rainPath': rainPath,
            'evpFile': Path + evpFile,
            'datResults': datResults,
            'controlFile': controlFile,
        }
        if date1 is not None:
            Default.update({'unix1': aux.__datetime2unix__(Default['date1'])})
        else:
            Default.update({'unix1': '$'+'unix1'})
        if date2 is not None:
            Default.update({'unix2': aux.__datetime2unix__(Default['date2'])})
        else:
            Default.update({'unix2': '$'+'unix2'})
        #Check for parameters left undefined
        D = {}
        for k in Default.keys():
            if Default[k] is not None:
                D.update({k: Default[k]})
            else:
                if noWarning:
                    print('Warning: parameter ' + k +' left undefined model wont run')
                D.update({k: '$'+k})
        #Update parameter on the base and write global 
        f = open(path2global,'w')
        f.writelines(Base.substitute(D))
        f.close()

    def set_parameters(self):
        self.Formats = []
        self.Table['Vr'] = 0.0041
        self.Formats.append('%.4f')
        self.Table['ar'] = 1.67
        self.Formats.append('%.2f')
        self.Table['Vs1'] = 2.04e-7#2.04e-6
        self.Formats.append('%.2e')
        self.Table['Vs2'] = 8.11e-6#1.11e-3#3.11e-3
        self.Formats.append('%.2e')
        self.Table['k1'] = 0.0067
        self.Formats.append('%.4f')
        self.Table['k2'] = 2.0e-4#1.66e-4
        self.Formats.append('%.2e')
        self.Table['tl'] = 0.12
        self.Formats.append('%.2f')
        self.Table['bl'] = 1.55
        self.Formats.append('%.2f')
        self.Table['l1'] = 0.25
        self.Formats.append('%.2f')
        self.Table['l2'] = -0.1
        self.Formats.append('%.1f')
        self.Table['vo'] = 0.4
        self.Formats.append('%.1f')
    
    
    def write_prm(self, ruta, extraNames = None, extraFormats = None):
        '''Writes the distributed prm file used for the 6XX model family'''
        #Converts the dataFrame to dict 
        D = self.Table.T.to_dict()
        # arregla la ruta 
        path, ext = os.path.splitext(ruta)
        if ext != '.prm':
            ruta = path + '.prm'
        #Escritura 
        f = open(ruta, 'w')
        f.write('%d\n\n' % len(D))
        for k in D.keys():
            f.write('%s\n' % k)
            f.write('%.5f %.5f %.5f ' % (D[k]['Acum'],
                D[k]['Long'],D[k]['Area']))
            if extraNames is not None:
                c = 0
                for k2 in extraNames:
                    try:
                        fo = extraFormats[c]
                    except:
                        fo = '%.5f '
                    f.write(fo % D[k][k2])
                    f.write(' ')
                    c += 1
            f.write('\n\n')
        f.close()

        
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

    def Dat2Msg(link, folder):
        '''converts dat files to a msg serie'''
        L = glob.glob(folder+'254_*.dat')
        L.sort()
        anos = [l.split('_')[1] for l in L]

        dates = pd.date_range('2008-04-01','2019-12-30', freq='15min')
        Qs = pd.Series(np.zeros(dates.size), index=dates)
        for i,a in zip(L, anos):
            d = am.ASYNCH_results(i)
            qs = d.ASYNCH_dat2Serie(link, a+'-04-01', freq='15min')
            Qs[qs.index] = qs.values
        Qs.to_msgpack('/Users/nicolas/BaseData/HLM254/'+str(link)+'_all.msg')

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
        OutStatesName = 'OutputStates.dat', initial_name = None, initial_exist = False,
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
        if initial_name is None:            
            self.path_in_initial = self.path_in + self.name + '.dbc'
        else:
            self.path_in_initial = initial_name
        if initial_exist is False:
            self.__ASYNCH_setInitialFile__(self.path_in_initial,self.date1[:4],
                self.linkID)
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


