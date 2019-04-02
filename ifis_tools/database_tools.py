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

# # database_tools:
#
# Set of tools to connect to the data base, put and get data from them.

import psycopg2
from psycopg2 import sql
import pandas as pd 
from datetime import datetime
import numpy as np 
from ifis_tools import auxiliar as aux


# +
def DataBaseConnect(user = "iihr_student", password = "iihr.student", host = "s-iihr51.iihr.uiowa.edu",
    port = "5435", database = "research_environment"):
    '''Connect to the database that hsa stored the usgs information'''
    con = psycopg2.connect(user = user,
        password = password,
        host = host,
        port = port,
        database = database)
    return con

def SQL_read_USGS_Streamflow(usgs_id, date1, date2, schema = 'pers_nico', 
    table = 'data_usgs', time_name = 'unix_time', data_name = 'val', usgs_name = 'usgs_id'):
    '''Read streamflow data from IIHR database "research_environment" 
    and returns it as a pandas.DataFrame element.
    Parameters:
        - usgs_id: code of the usgs.
        - date1: initial date of the query.
        - date2: final date of the query.
    Optional:
        - schema: where to obtain data in the databse.
        - table: master table with the usgs data.
        - time_name: the name of the column that has the time.
        - data_name: the name of the column that has the data.
        - usgs_name: the name of the column that has the id of the usgs stations.
    Returns:
        - pandas.DataFrame containing the streamflow data.'''
    #make the connection
    con = DataBaseConnect(user = 'nicolas', password = '10A28Gir0')
    #Work with dates and usgs id
    date1 = str(aux.__datetime2unix__(date1))
    date2 = str(aux.__datetime2unix__(date2))
    if type(usgs_id) is not str:
        usgs_id = str(usgs_id)
    #make the querty
    query = sql.SQL("SELECT "+time_name+", "+data_name+" FROM "+schema+"."+table+" WHERE "+time_name+" BETWEEN "+date1+" and "+date2+" AND "+usgs_name+"='"+usgs_id+"'")
    #Make the consult.
    Data = pd.read_sql(query, con, index_col='unix_time',parse_dates={'unix_time':{'unit':'s'}})
    con.close()
    return Data

#SQL Query to obtain the data from per_felipe.pois_adv_geom
def SQL_USGS_at_IFIS():
    '''Return the list of the usgs stations in the IFIS system and the linkID where they 
    belong.'''
    #make the connection
    con = DataBaseConnect(user = 'nicolas', password = '10A28Gir0')
    #Query for the stations
    query = sql.SQL("SELECT foreign_id,link_id FROM pers_felipe.pois_adv_geom where type in (2,3) and foreign_id like '0%' AND link_id < 620000")
    #make the consult
    cur = con.cursor()
    cur.execute(query)
    L = cur.fetchall()
    cur.close()
    con.close()
    #Obtains a dictionary in which stations are the key
    DicUSGSinIFIS = {}
    for l in L:
        DicUSGSinIFIS.update({l[0]:l[1]})
    return DicUSGSinIFIS

def SQL_USGS_at_MATC():
    '''Return the list of stations that are in the databse pers_nico (matc).'''
    #make the connection
    con = DataBaseConnect(user = 'nicolas', password = '10A28Gir0')
    #Make the query
    query = sql.SQL("SELECT DISTINCT(usgs_id) FROM pers_nico.data_usgs_2008")
    cur = con.cursor()
    cur.execute(query)
    L = cur.fetchall()
    cur.close()
    con.close()
    return [l[0] for l in L]

def SQL_Get_linkArea(linkID):
    '''Obtains the up area for a link ID'''
    #The query and the obtentions
    con = DataBaseConnect('nicolas','10A28Gir0')
    cur = con.cursor()
    q = sql.SQL("SELECT upstream_area FROM pers_felipe.pois_adv_geom WHERE link_id = "+str(linkID))
    cur.execute(q)
    A = cur.fetchall()
    cur.close()
    con.close()
    return A[0][0]*2.583

def SQL_Get_Coordinates(linkID):
    con = DataBaseConnect(user='nicolas',password='10A28Gir0')
    cur = con.cursor()
    LatLng = {}
    query = sql.SQL('SELECT lat, lng FROM pers_felipe.pois_adv_geom where link_id = '+str(linkID))
    cur.execute(query)
    Coord = cur.fetchall()
    con.close()
    return float(Coord[0][0]),float(Coord[0][1])       
    
def SQL_Read_MeanRainfall(link_id, date1, date2, schema = 'pers_nico', 
    table = 's4mrain', time_name = 'unix_time', data_name = 'rain', linkid_name = 'link_id'):
    '''Read streamflow data from IIHR database "research_environment" 
    and returns it as a pandas.DataFrame element.
    Parameters:
        - usgs_id: code of the usgs.
        - date1: initial date of the query.
        - date2: final date of the query.
    Optional:
        - schema: where to obtain data in the databse.
        - table: master table with the usgs data.
        - time_name: the name of the column that has the time.
        - data_name: the name of the column that has the data.
        - usgs_name: the name of the column that has the id of the usgs stations.
    Returns:
        - pandas.DataFrame containing the streamflow data.'''
    #make the connection
    con = DataBaseConnect(user = 'nicolas', password = '10A28Gir0')
    #Work with dates and usgs id
    date1 = str(aux.__datetime2unix__(date1))
    date2 = str(aux.__datetime2unix__(date2))
    if type(link_id) is not str:
        link_id = str(link_id)
    #make the querty
    query = sql.SQL("SELECT "+time_name+", "+data_name+" FROM "+schema+"."+table+" WHERE "+time_name+" BETWEEN "+date1+" and "+date2+" AND "+linkid_name+"='"+link_id+"'")
    #Make the consult.
    Data = pd.read_sql(query, con, index_col='unix_time',parse_dates={'unix_time':{'unit':'s'}})
    con.close()
    #Organize rainfall 
    Data = Data.sort_index()
    Dates = pd.date_range(Data.index[0], Data.index[-1], freq='1h')
    Rain = pd.Series(np.zeros(Dates.size), Dates)
    Rain[Data.index] = Data['rain'].values
    Rain[Rain>1000] = 0.0
    return Rain

def SQL_Get_MeanRainfall(linkID, date1, date2):
    '''Obtains the mean rainfall for the watershed associated to 
    a given linkID.
    Parameters:
        - linkID: linkID of the outlet of the basin.
        - date1: initial date (YYYY-MM-DD HH:MM).
        - date2: end date (YYYY-MM-DD HH:MM).
    Returns:
        - Rainfall: Pandas series with the mean rainfall in the basin.'''
    #SEt the connection
    con = DataBaseConnect(user='nicolas', password='10A28Gir0', database='rt_precipitation')
    #Transform dates to unix 
    unix1 = str(aux.__datetime2unix__(date1))
    unix2 = str(aux.__datetime2unix__(date2))
    linkID = str(linkID)
    #Set the query and obtains data
    q = sql.SQL("WITH subbasin AS (SELECT nodeX.link_id AS link_id FROM students.env_master_km AS nodeX, students.env_master_km AS parentX WHERE (nodeX.left BETWEEN parentX.left AND parentX.right) AND parentX.link_id = "+str(linkID)+"), uparea as (SELECT up_area FROM students.env_master_km WHERE link_id= "+str(linkID)+"), lut as (SELECT x, y FROM env_lookup_hrap_lid_v4 WHERE link_id IN (SELECT * FROM subbasin) group by x, y) SELECT unix_time, sum(val)/(SELECT count(*) FROM lut) as rain FROM stage_4.data WHERE grid_x IN (SELECT x FROM lut) AND grid_y IN (SELECT y from lut) AND unix_time between "+unix1+" AND "+unix2+" group by unix_time order by unix_time;")
    Data = pd.read_sql(q, con, index_col='unix_time',parse_dates={'unix_time':{'unit':'s'}})
    #close connection
    con.close()
    #Pos process data 
    dates = pd.date_range(date1, date2, freq='1h')
    Rain = pd.Series(np.zeros(dates.size), dates)
    Rain[Data.index] = Data['rain'] 
    return Rain

# -


