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

# # auxiliar
#
# set of auxiliar functions that support the rest of the scripts.

import os
from datetime import timezone, datetime
import fileinput

# +

def __make_folder__(pathFolder):
    '''Creates a new folder in the specify path'''
    #Check that there is no extension
    pathFolder = list(os.path.splitext(pathFolder))
    if len(pathFolder[-1])>0:
        return 'Error: path to folder must not contain an extension'
    else:
        #check if the directory already exists
        if os.path.isdir(pathFolder[0]):            
            return 'Error: the folder already exists'
        else:
            #Comand to create folder.
            comando = 'mkdir '+pathFolder[0]
            os.system(comando)
            return 0            

def __check_extension__(path, ext, putExtension = True):
    '''Check if the path or file given ends with the right extension
    Parameters:
        - path: the path of the file to eval.
        - ext: extension (with the period ".ext) to compare with.
        - putExtension: Put or not the new extension to the file.
    Returns:
        - 0 if has the extension, 1 if not.
        - the path with the extension.'''
    #Obtains the extension 
    path = list(os.path.splitext(path))
    if path[-1] == ext:
        return 0
    else:
        if putExtension:
            path[-1] = ext
            return ''.join(path)
        else:
            return 1

####################################################################################################################
# Date and time utilities.
####################################################################################################################

def __datetime2unix__(date):
    '''Converts a datetime object into a unix time
    this is usefull for SQL queries.
    Parameters:
        - date: a datetime object datetime.datetime(YYYY,MM,DD,HH,MM)
            or a string object with the date format.
    Returns:
        - unix time: the total elapsed time in unix format'''
    #Check format 
    formato = '%Y-%m-%d %H:%M '
    flag = True
    i = 1
    while flag:
        try:
            date = datetime.strptime(date, formato[:-i])
            flag = False
        except:
            i += 1
        if i>len(formato): flag = False
    #make the conversion
    timestamp = date.replace(tzinfo=timezone.utc).timestamp()
    return int(timestamp)

####################################################################################################################
# Asynch utilities
####################################################################################################################

def __ASYNCH_initialFile__(InitialOut, linkID, InitialBase = '190BaseInitial.dbc'):
    '''Set the dbc query for the initial conditions for asynch
    Parameters:
        - InitialOut: The path and name of the outlet initial file for the dbc.
        - linkID: the link in which is going to be placed the initial file.
        - InicialBase: Path to the basefile for the stablishment of the initial file.
    Results:
        - Writes the initial file at InitialOut.'''
    #Chekc the extension of hte initial file 
    __check_extension__(InitialOut, '.dbc')
    # Copy the initial file state in
    comando = 'cp '+InitialBase+' '+InitialOut
    os.system(comando)
    # Replace the linkID in the initial file so asynch knows.
    filename = InitialOut
    with fileinput.FileInput(filename, inplace = True) as file:
        for line in file:
            text_to_search = '¿linkID?'
            replacement_text = str(linkID)
            print(line.replace(text_to_search, replacement_text), end='')
# -


