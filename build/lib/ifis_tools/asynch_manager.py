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

import fileinput
import io
import os
import sys
from datetime import datetime, timezone
from string import Template
from struct import pack, unpack
import re
import numpy as np
import pandas as pd
import glob
from ifis_tools import auxiliar as aux
from ifis_tools import database_tools as db

try:
    from wmf import wmf
except:
    print('Unable to import WMF, cant create basins whit it')

try:
    from ipywidgets import FloatProgress
    from IPython.display import display
    floatBar = True
except:
    print('Unable to import FloatProgress and Ipython.display, it seems that you are not in a Jupyter-notebook')
    floatBar = False

def __saveBin__(lid, lid_vals, count, fn):
    io_buffer_size = 4+4*100000
    if count > 0:
        lid = (lid[lid_vals > 0.005])
        lid_vals = (lid_vals[lid_vals > 0.005])
    fh = io.open(fn, 'wb', io_buffer_size)
    fh.write(pack('<I', count))
    for vals in zip(lid, lid_vals):
        fh.write(pack('<If', *vals))
    fh.close()
# # Global variables 

# +
#SEt of parameters for the different configurations of the model 
Parameters = {'190': [6, 0.75, 0.33, -0.20, 0.50, 0.1, 2.2917e-5],
    '254':[12, 0.33, 0.2, -0.1, 0.02, 2.0425e-6, 0.02, 0.5, 0.10, 0.0, 99.0, 3.0, 0.75]}

#Path of the file 
if os.name == 'posix':
    Path = __file__.split('/')
    Path = '/'.join(Path[:-1])+'/'
elif os.name == 'nt':
    ListPaths = sys.path
    for l in ListPaths:
        if l.split('\\')[-1] == 'site-packages':
            Path = l
    Path += '\\ifis_tools\\'
#Read the global files that are used to generate new globals
#try:
Globals = {}
for g in ['190','254','60X']:
    # 190 global base format 
    f = open(Path+g+'BaseGlobal.gbl','r')
    Globals.update({g:f.readlines()})
    f.close()
#except:
 #   print('Warning: no base globals copied, you have to use your own')

# -

# # ASYNCH tools  
#
# ## Asynch results reader 

def UpdateGlobal(filename, DictUpdates):
   #Iterate in the updates keys
   for k in DictUpdates:
       with fileinput.FileInput(filename, inplace=True) as file:
           for line in file:
               print(line.replace(DictUpdates[k]['old'],DictUpdates[k]['new']), end = '')

class hlmModel:

    def __init__(self,linkid=None, path = None, ExtraParams = None, model_uid = 604):
        '''Depending on the linkid or in the path the class starts a table
        to set up a new project fro HLM model.
            - linkid = number of link id to search for in the database.
            - path =  path to search for a WMF.SimuBasin project .nc
        Optional:
            -ExtraParams: List with the names of the extra params extracted from the database'''
        #Type of model to be used 
        self.model_uid = model_uid
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

    def write_control(self, path , linkList = None):
        '''Writes the control.sav file used by the model to determine at which links store the
        obtained results.'''
        #If no link list it tries to obtain the control fronm the database
        if linkList is None:
            con = db.DataBaseConnect(user='nicolas', password='10A28Gir0')
            if self.linkid > 0:
                q = db.sql.SQL("SELECT distinct \
                    us.link_id \
                FROM \
                    pers_nico.iowa_usgs_stations us, \
                    pers_nico.subbasin("+str(self.linkid)+") sub \
                where us.link_id = sub")
            elif self.linkid == 0:
                q = db.sql.SQL("SELECT distinct \
                    us.link_id \
                FROM \
                    pers_nico.iowa_usgs_stations us, \
                    pers_nico.master_lambda_vo mas \
                where us.link_id = mas.link_id")
            linkList = pd.read_sql(q, con).values.T.astype(int).tolist()[0]
        #Opens the file 
        f = open(path,'w',newline = '\n')
        Links = self.Table.index.tolist()
        for l in linkList:
            try:
                Links.index(l)
                f.write('%d\n' % l)
            except:
                pass
        f.close()
    
    def write_rainfall(self, date1, date2, path):
        '''Writes binary files for the rainfall of the defined watershed.
            - date1: initial date of the data to write.
            - date2: end date
            - path: where to store binary files eg: /home/user/basin/rain/BasinStage4_
        Returns:
            - the mean rainfall of the watershed for the selected period'''
        #Databse connection and query.
        unix1 = str(aux.__datetime2unix__(date1))
        unix2 = str(aux.__datetime2unix__(date2))
        con = db.DataBaseConnect(database='rt_precipitation')
        if self.linkid > 0:
            q = db.sql.SQL("WITH subbasin AS (SELECT nodeX.link_id AS link_id FROM students.env_master_km \
         AS nodeX, students.env_master_km AS parentX WHERE (nodeX.left BETWEEN parentX.left AND parentX.right) \
         AND parentX.link_id = "+str(self.linkid)+") SELECT A.unix_time,sum(weight*A.val) as rain,B.link_id FROM stage_4.data AS \
         A,env_lookup_hrap_lid_v4 AS B,subbasin WHERE A.grid_x = B.x AND A.grid_y=B.y AND B.link_id = subbasin.link_id \
         AND A.unix_time >= "+str(unix1)+" AND A.unix_time < "+str(unix2)+" AND A.val < 99.0 GROUP BY B.link_id,A.unix_time ORDER BY A.unix_time")
        else:
            q = db.sql.SQL("SELECT \
                    A.unix_time, \
                    sum(weight*A.val) as rain, \
                    B.link_id \
                FROM \
                    stage_4.data AS A, \
                    env_lookup_hrap_lid_v4 AS B \
                WHERE A.grid_x = B.x AND A.grid_y=B.y AND A.unix_time >= "+str(unix1)+" AND A.unix_time < "+str(unix2)+" AND A.val < 999.0 \
                GROUP BY B.link_id,A.unix_time \
                ORDER BY A.unix_time;")
        E = pd.read_sql(q, con, index_col='unix_time')
        con.close()
        #SEtup for the data with the mean rainfall for that period
        d = pd.date_range(date1, date2, freq='1H')
        MeanRain = pd.Series(np.zeros(d.size), index = d)
        #Rainfall binary files creation
        for i in np.arange(E.index[0], E.index[-1], 3600):
            Rain = E[['link_id','rain']][E.index == i]
            __saveBin__(Rain['link_id'].values, Rain['rain'].values, Rain['rain'].size,path+str(i))
            MeanRain[pd.to_datetime(i,unit='s')] = Rain['rain'].mean()
        MeanRain[np.isnan(MeanRain) == True] = 0.0
        return MeanRain

    def write_rvr(self, path = None, database = 'rt_precipitation'):
        #conncet to the database
        con = db.DataBaseConnect(user = 'nicolas', password = '10A28Gir0',)
        #restore_res_env_92
        #Query to ask for the link ids and the topology
        if self.linkid > 0:
            q = db.sql.SQL("WITH all_links(id) AS (SELECT link_id FROM pers_nico.master_lambda_vo) \
             SELECT all_links.id,pers_nico.master_lambda_vo.link_id FROM pers_nico.master_lambda_vo,all_links \
             WHERE (all_links.id IN (SELECT nodeX.link_id FROM pers_nico.master_lambda_vo AS nodeX, \
             pers_nico.master_lambda_vo AS parentX \
             WHERE (nodeX.left BETWEEN parentX.left AND parentX.right) AND parentX.link_id = "+str(self.linkid)+")) AND pers_nico.master_lambda_vo.parent_link = all_links.id ORDER BY all_links.id")
        elif self.linkid == 0:
            q = db.sql.SQL("WITH all_links(id) AS (SELECT link_id FROM pers_nico.master_lambda_vo) \
            SELECT DISTINCT all_links.id,pers_nico.master_lambda_vo.link_id FROM pers_nico.master_lambda_vo,all_links \
            WHERE all_links.id > 1 AND pers_nico.master_lambda_vo.model AND \
            pers_nico.master_lambda_vo.parent_link = all_links.id ORDER BY all_links.id;")

        self.topo = pd.read_sql(q, con)
        con.close()
        topo = self.topo.values.T
        #Convert the query to a rvr file 
        if path is not None:
            f = open(path,'w',  newline='\n')
            f.write('%d\n\n' % topo.shape[1])
            #List = self.Table.index.tolist()
            for t in topo[1]:
                #List.index(t)
                f.write('%d\n'% t)
                p = np.where(topo[0] == t)[0]
                if len(p)>0:
                    f.write('%d ' % p.size)
                    for i in p:
                        f.write('%d ' % topo[1][i])
                    f.write('\n\n')
                else:
                    f.write('0\n\n')
            f.close()
        #Check for consistency with the Table
        a = pd.Series(self.topo.shape[0], self.topo['link_id'].values)
        b = 0
        for i in self.Table.index:
            if i in a.index:
                b += 1
            else:
                self.Table = self.Table.drop(i, axis = 0)

    def write_initial(self, path, initial = [1e-6, 0.0001, 0.05, 1.0], kind = 'uini',year = 2012):
        '''Writes an initial file for the model
        parameters:
            - path: where to store the file.
            - initial: vector for the uini or for the ini files.
            - kind: defines the type of initial conditions:
                - uini: uniform initial conditions.
                - ini: non-uniform text based initial conditions.
                - dbase: retrieve initial conditions from data base.
            - year: initial year for the case of the kind =  dbase'''
        #opens the file
        if kind == 'uini':
            f = open(path, 'w',  newline='\n')
            f.write('%d\n' % self.model_uid)
            f.write('0.000000\n')
            for i in initial:
                f.write('%.3e ' % i)
            f.close()
        elif kind == 'dbase':
            t = "dbname=research_environment host=s-iihr51.iihr.uiowa.edu \
port=5435 user=nicolas password=10A28Gir0 \
\n\n1 \n\nSELECT link_id, state_0, state_1, state_2, state_3 \
FROM pers_felipe_initial_conditions.initialconditions_"+str(year)+" order by link_id"
            f = open(path, 'w')
            f.write(t)
            f.close()


    def write_Global(self, path2global, model_uid = 604,
        date1 = None, date2 = None, rvrFile = None, rvrType = 0, rvrLink = 0, prmFile = None, prmType = 0, initialFile = None,
        initialType = 1,rainType = 5, rainPath = None, evpFile = 'evap.mon', datResults = None,
        nComponents = 1, Components = [0], controlFile = None, baseGlobal = None, noWarning = False, snapType = 0,
        snapPath = '', snapTime = '', evpFromSysPath = False):
        '''Creates a global file for the current project.
            - model_uid: is the number of hte model goes from 601 to 604.
            - date1 and date2: initial date and end date
            - rvrFile: path to rvr file.
            - rvrType: 0: .rvr file, 1: databse .dbc file.
            - rvrLink: 0: all the domain, N: number of the linkid.
            - prmFile: path to prm file.
            - prmType: 0: .prm file, 1: databse .dbc file.
            - initialFile: path to file with initial conditions.
            - initialType: type of initial file:
                - 0: ini, 1: uini, 2: rec, 3: .dbc
            - rainType: number inficating the type of the rain to be used.
                - 1: plain text with rainfall data for each link.
                - 3: Database.
                - 4: Uniform storm file: .ustr
                - 5: Binary data with the unix time
            - rainPath: path to the folder containning the binary files of the rain.
                or path to the file with the dabase
            - evpFile: path to the file with the values of the evp.
            - datResults: File where .dat files will be written.
            - nComponents: Number of results to put in the .dat file.
            - Components: Number of each component to write: [0,1,2,...,N]
            - controlFile: File with the number of the links to write.
            - baseGlobal: give the option to use a base global that is not the default
            - snapType: type of snapshot to make with the model:
                - 0: no snapshot, 1: .rec file, 2: to database, 3: to hdf5, 4:
                    recurrent hdf5
            - snapPath: path to the snapshot.
            - snapTime: time interval between snapshots (min)
            - evpFromSysPath: add the path of the system to the evp file.'''
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
        # Databse rainfall 
        if rainType == 3 and rainPath is None:
            rainPath = '/Dedicated/IFC/model_eval/forcing_rain51_5435_s4.dbc'
        if rvrType == 1 and rvrFile is None:
            rvrFile = '/Dedicated/IFC/model_eval/topo51.dbc'
        #Chang  the evp path 
        if evpFromSysPath:
            evpFile = Path + evpFile
        # Creates the default Dictionary.
        Default = {
            'model_uid' : model_uid,
            'date1': date1,
            'date2': date2,
            'rvrFile': rvrFile,
            'rvrType': str(rvrType),
            'rvrLink': str(rvrLink),
            'prmFile': prmFile,
            'prmType': str(prmType),
            'initialFile': initialFile,
            'initialType': initialType,
            'rainType': str(rainType),
            'rainPath': rainPath,
            'evpFile': evpFile,
            'datResults': datResults,
            'controlFile': controlFile,
            'snapType': str(snapType),
            'snapPath': snapPath,
            'snapTime': str(snapTime),
            'nComp': str(nComponents)
        }
        if date1 is not None:
            Default.update({'unix1': aux.__datetime2unix__(Default['date1'])})
        else:
            Default.update({'unix1': '$'+'unix1'})
        if date2 is not None:
            Default.update({'unix2': aux.__datetime2unix__(Default['date2'])})
        else:
            Default.update({'unix2': '$'+'unix2'})
        #Update the list of components to write
        for n, c in enumerate(Components):
            Default.update({'Comp'+str(n): 'State'+str(c)})
        if nComponents <= 9:
            for c in range(9-nComponents):
                Default.update({'Comp'+str(8-c): 'XXXXX'})
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
        f = open(path2global,'w', newline='\n')
        f.writelines(Base.substitute(D))
        f.close()
        #Erase unused print components
        f = open(path2global,'r')
        L = f.readlines()
        f.close()
        flag = True
        while flag:
            try:
                L.remove('XXXXX\n')
            except:
                flag = False
        f = open(path2global,'w', newline='\n')
        f.writelines(L)
        f.close()


    def write_runfile(self, path, process, jobName = 'job',nCores = 56, nSplit = 1):
        '''Writes the .sh file that runs the model
        Parameters:
            - path: path where the run file is stored.
            - process: dictionary with the parameters for each process to be launch:
                eg: proc = {'Global1.gbl':{'nproc': 12, 'secondplane': True}}
            - ncores: Number of cores.
            - nsplit: Total number of cores for each group.'''
        #Define the size of the group of cores
        if nCores%nSplit == 0:
            Groups = int(nCores / nSplit)
        else:
            Groups = int(nCores / 2)
        #Define the header text.
        L = ['#!/bin/sh\n#$ -N '+jobName+'\n#$ -j y\n#$ -cwd\n#$ -pe '+str(Groups)+'cpn '+str(nCores)+'\n####$ -l mf=16G\n#$ -q IFC\n\n\
/bin/echo Running on host: `hostname`.\n\
/bin/echo In directory: `pwd`\n\
/bin/echo Starting on: `date`\n']

        f = open(path,'w',  newline='\n')
        f.write(L[0])
        f.write('\n')

        for k in process.keys():
            secondplane = ' \n'
            if process[k]['secondplane']:
                secondplane = ' &\n'
            if process[k]['nproc'] > nCores:
                process[k]['nproc'] = nCores        
            f.write('mpirun -np '+str(process[k]['nproc'])+' /Users/nicolas/Tiles/dist/bin/asynch '+k+secondplane)
        f.close()

    def set_parameters(self, Vr = 0.0041, ar = 1.67, Vs1 = 2.04e-7, Vs2 = 8.11e-6,
        k1 = 0.0067, k2 = 2.0e-4, tl = 0.1, bl = 1.0, l1 = 0.2, l2 = -0.1, vo = 0.4,
        VrF = '%.4f', arF = '%.2f', Vs1F = '%.2e', Vs2F = '%.2e', k1F = '%.4f', k2F = '%.2e',
        tlF = '%.2f', blF = '%.2f', l1F = '%.2f', l2F = '%.2f', voF = '%.2f', DictP = None, DicOrder = None):
        '''The parameters correspond to the model 604 if you want to put another parameters you
        must use the variable DictP as an example this variable must be like:
            - DictP['a']['value'] = [1,2,3,4,..., N],
            - DictP['a']['format'] = '%.2f' 
        The DicOrder vartiable must be use in order to determine the order to write the variables in the prm file
        as an examples DicOrder = ['a', 'c','j','b','z'] otherwise the function probably will write it in disorder'''
        self.Formats = []
        self.Table['Vr'] = Vr
        self.Formats.append(VrF)
        self.Table['ar'] = ar
        self.Formats.append(arF)
        self.Table['Vs1'] = Vs1
        self.Formats.append(Vs1F)
        self.Table['Vs2'] = Vs2
        self.Formats.append(Vs2F)
        self.Table['k1'] = k1
        self.Formats.append(k1F)
        self.Table['k2'] = k2
        self.Formats.append(k2F)
        self.Table['tl'] = tl
        self.Formats.append(tlF)
        self.Table['bl'] = bl
        self.Formats.append(blF)
        self.Table['l1'] = l1
        self.Formats.append(l1F)
        self.Table['l2'] = l2
        self.Formats.append(l2F)
        self.Table['vo'] = vo
        self.Formats.append(voF)
        if DictP is not None:
            for k in DicOrder:
                self.Table[k] = DictP[k]['value']
                self.Formats.append(DictP[k]['format'])

    def write_prm(self, ruta, extraNames = None, extraFormats = None):
        '''Writes the distributed prm file used for the 6XX model family'''
        #Converts the dataFrame to dict 
        D = self.Table.T.to_dict()
        # arregla la ruta 
        path, ext = os.path.splitext(ruta)
        if ext != '.prm':
            ruta = path + '.prm'
        #Escritura 
        f = open(ruta, 'w', newline = '\n')
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


class hlm_dat_process:

    def __init__(self,sav_path = None):
        '''Reads a .sav file to know wich links to extract'''
        #Open the file 
        if sav_path is not None:
            self.links = self.eval_links(sav_path)
        else:
            self.links = []

    def eval_links(self, sav_path):
        '''Reads a .sav file with the number of the links to read from a .dat
        (or eventually from a h5 file)
        Params:
            - sav_path: the path to the plain text .sav
        Returns:
            - self.links'''
        f = open(sav_path)
        links = f.readlines()
        f.close()
        return [int(i) for i in links]

    def read_dat_file(self, path):
        '''Reads a dat file and stores the information of it'''
        f = open(path,'r')
        self.dat = f.readlines()
        f.close()

    def dat_record2pandas(self, linkID, date1, freq):
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
        Data = np.array([np.array(l.split()).astype(float) for l in self.dat[Start[PosStat]+1:End[PosStat]+1]])
        #return self.dat[Start[PosStat]+1:End[PosStat]+1]
        Dates = pd.date_range(date1, periods=self.Nrec, freq=freq)
        return pd.DataFrame(Data, Dates)
    
    def dat_all2pandas(self, path_in, path_out, sim_name=None, initial_name = '',
        start_year = '2000', start_date = '-04-01 01:00', freq = '1H', stages = 'all', stages_names = None,
        nickname = None):
        '''Takes a list of dat files or a dat file and extracts the simulated streamflow to records
        Parameters:
            - path_in: path with the .dat files, no extension if the user want to process 
                multiple years and mus use sim_name. Otherwise give the full path to 
                the .dat and dont use sim_name
            - sim_name: name of the simulations that correspond to the .dat files
                eg. if in the folder there are: hlm254_2012.dat, hlm254_2013.dat, hlm604_2012.dat and 
                sim_name = hlm254, the function will only eval the hlm254* cases.
                WARNING: this works for files of the same setup with several years.
            - path_out: path to put outfiles.
            - initial_name: initial strings of the files that are goinig to be analyzed.
            - start_date: -mm-dd HH:MM of the initial date of the simulation.
            - start_year: the year to start if cant find the yar from the .dat files name
            - freq: frequency of the simulation period.
            - stages: storage of the dat file to be transformed, all stores all, otherwise it stores
                just some of them.
            - stages_names: names to put in the DataFrame for each column
            - nickname: name to put to the output files instead of the sim_name.
        Results:
            - Writes an msgpack with pandas Series item for each link in the dat system'''
        #In functions
        def find_sim_name(dat_name, sim_name):
            '''Fuinds the name of a dat_name extracted from the .dat file'''
            return dat_name[dat_name.index(sim_name[0]):dat_name.index(sim_name[-1])+1]
        def find_year(name, date_format = '\d{4}'):
            '''Finds the year of the .dat file, right now only supports files with the year on them'''
            m = re.search('\d{4}', name)
            return m.group()
        #extension end Dataframe based on the stages
        if type(stages) == list:
            if type(stages_names) == list:
                end_name = '-'.join([str(i) for i in stages_names])
            else:
                end_name = '-'.join([str(i) for i in stages])
        else:
            end_name = ''
        #Defines dat lists
        dat_names = []
        dat_paths = []
        dat_years = []
        #Raw dat files in a folder
        if sim_name is None:
            dat_list_raw = glob.glob(path_in + '*.dat')
            dat_list_raw.sort()
            print(dat_list_raw)
            #Finds ths dat files that has the specified name
            for i in dat_list_raw:
                try:
                    if os.name == 'nt':
                        #Windows
                        name = find_sim_name(i.split('\\')[1], initial_name)
                        print(name)
                    if os.name == 'posix':
                        #Linux
                        name = find_sim_name(i.split('/')[-1], initial_name)
                    if name == initial_name:
                        dat_names.append(name)
                        dat_paths.append(i)
                        dat_years.append(find_year(i.split('/')[-1]))
                except OSError as err:
                    print("OS error: {0}".format(err))
        else:
            if os.name == 'nt':
                dat_names = [path_in.split('\\')[-1].split('.')[0]]
                try:                
                    dat_years = [find_year(path_in).split('\\')[-1].split('.')[0]]
                except:
                    dat_years = [start_year,]
            elif os.name == 'posix':
                dat_names = [path_in.split('/')[-1].split('.')[0],]
                try:                
                    dat_years = [find_year(path_in).split('/')[-1].split('.')[0],]
                except:
                    dat_years = [start_year,]
            dat_paths = [path_in,]
        #Reads the dat files and writes the msg packs.
        if floatBar:
            f1 = FloatProgress(min =0, max = len(dat_paths))
            display(f1)
        #Goes for every year
        first = True
        for path,name,year in zip(dat_paths, dat_names, dat_years):
            #Reads the dat file 
            self.read_dat_file(path)
            #Put the nickname
            if nickname is not None:
                name = nickname
            #Process each link
            for link in self.links:
                #extract the info from the dat file
                if stages is 'all':
                    q = self.dat_record2pandas(link, str(year)+start_date, freq)
                else:
                    q = self.dat_record2pandas(link, str(year)+start_date, freq)[stages]
                #Change column names 
                if type(stages_names) == list:
                    for i,j in zip(q.columns.values.tolist(),stages_names):
                        q = q.rename(columns={i:j})
                #Writes the maskpack
                if first:
                    q.to_msgpack(path_out+str(link)+'_'+name+'_'+end_name+'.msg')
                else:
                    q_old = pd.read_msgpack(path_out+str(link)+'_'+name+'_'+end_name+'.msg')
                    qtot = q_old.append(q)
                    qtot.to_msgpack(path_out+str(link)+'_'+name+'_'+end_name+'.msg')
            if floatBar:
                f1.value+=1
            first = False

# ## Asynch project manager

class hlm_project:
    
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
