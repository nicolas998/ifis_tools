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
try:
    from climata.usgs import InstantValueIO, DailyValueIO
except:
    print('Warning: climata not installed cant use the functions to get USGS data')
import numpy as np 
from ifis_tools import auxiliar as aux
import sqlalchemy
import sqlalchemy.types as sqt

print('warning: to use the data base tool you must set the following variables: data_usr, data_pass. \nAlso you can change:data_host, data_base, data_port')
data_usr = None
data_pass = None
data_host = "s-iihr51.iihr.uiowa.edu"
data_base = "research_environment"
data_port = "5435"

# +
def DataBaseConnect(user = "iihr_student", password =data_pass, host = data_host,
    port = "5435", database = "research_environment"):
    '''Connect to the database that hsa stored the usgs information'''
    con = psycopg2.connect(user = user,
        password = password,
        host = host,
        port = port,
        database = database)
    return con

def SQL_getSubLinks(linkid):
    '''returns the list of links that belong to a certain link.'''
    con = DataBaseConnect(user='nicolas',password='10A28Gir0',database='rt_precipitation')
    query = 'SELECT nodeX.link_id AS link_id FROM students.env_master_km AS nodeX, students.env_master_km AS parentX WHERE (nodeX.left BETWEEN parentX.left AND parentX.right) AND parentX.link_id = '+str(linkid)
    Data = pd.read_sql(query, con)
    Data = Data.values.T[0]
    Data.sort()
    con.close()
    return Data

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

def WEB_Get_USGS(usgs_code, date1, date2, variable = '00060'):
    '''Get USGS data from the web using the climdata interface
    Parameters (debe ser probado):
        - usgs_code: the code of the station to obtain.
        - date1: initial date.
        - date2: final date.
        - variable: 
            - 00060 for streamflow.
            - 00065 for height'''
    #Get the data form the web     
    if variable =='00060':
        convert = 0.02832
    else:
        convert = 1
    data = InstantValueIO(
        start_date = pd.Timestamp(date1),
        end_date = pd.Timestamp(date2),
        station = usgs_code,
        parameter = variable)
    try:
        #Convert the data into a pandas series 
        for series in data:
            flow = [r[0] for r in series.data]
            dates = [r[1] for r in series.data]
        #Obtain the series of pandas
        Q = pd.Series(flow, pd.to_datetime(dates, utc=True)) * convert
        Index = [d.replace(tzinfo = None) for d in Q.index]
        Q.index = Index
    except:
        #Convert the data into a pandas series 
        for series in data:
            flow = [r[1] for r in series.data]
            dates = [r[0] for r in series.data]
        #Obtain the series of pandas
        Q = pd.Series(flow, pd.to_datetime(dates, utc=True)) * convert
        Index = [d.replace(tzinfo = None) for d in Q.index]
        Q.index = Index
    return Q

#SQL Query to obtain the data from per_felipe.pois_adv_geom
def SQL_USGS_at_IFIS():
    '''Return the list of the usgs stations in the IFIS system and the linkID where they 
    belong.'''
    #make the connection
    con = DataBaseConnect(user = data_usr, password = data_pass)
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
    con = DataBaseConnect(user = data_usr, password = data_pass)
    #Make the query
    query = sql.SQL("SELECT DISTINCT(usgs_id) FROM pers_nico.data_usgs_2008")
    cur = con.cursor()
    cur.execute(query)
    L = cur.fetchall()
    cur.close()
    con.close()
    return [l[0] for l in L]

def SQL_Get_linkArea(linkID, upArea = True):
    '''Obtains the up area for a link ID'''
    #The query and the obtentions
    con = DataBaseConnect(data_usr,data_pass,database='restore_res_env_92')
    cur = con.cursor()
    if upArea:
        q = sql.SQL("SELECT up_area FROM public.env_master_km where link_id="+str(linkID))
    else:
        q = sql.SQL("SELECT area FROM public.env_master_km where link_id="+str(linkID))
    cur.execute(q)
    A = cur.fetchall()
    cur.close()
    con.close()
    return A[0][0]

def SQL_Get_Coordinates(linkID):
    con = DataBaseConnect(user=data_usr,password=data_pass)
    cur = con.cursor()
    LatLng = {}
    query = sql.SQL('SELECT lat, lng FROM pers_felipe.pois_adv_geom where link_id = '+str(linkID))
    cur.execute(query)
    Coord = cur.fetchall()
    con.close()
    return float(Coord[0][0]),float(Coord[0][1])       
    
def SQL_Read_MeanRainfall(link_id, date1, date2, schema = 'pers_nico', 
    table = 's4mrain', time_name = 'unix_time', data_name = 'rain', linkid_name = 'link_id'):
    '''DEPRECATED Read streamflow data from IIHR database "research_environment" 
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
    con = DataBaseConnect(user=data_usr, password=data_pass, database='rt_precipitation')
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

def SQL_Get_WatershedFromMaster(linkID, otherParams = None, data_usr = data_usr, data_pass = data_pwd):
    '''Obtains the params files records for a watershed based on its linkID.
    The otherParams is a list with the names stand for other parameters that can also be obtained from the querty
    Other names are: [k_i,k_dry, h_b, topsoil_thickness, k_d, slope]'''
    #Obtains the connection 
    con = DataBaseConnect(user=data_usr, password=data_pass)
    #Set up the data that will ask for 
    text1 = "WITH subbasin AS (SELECT nodeX.link_id AS link_id FROM pers_nico.master_lambda_vo AS nodeX, pers_nico.master_lambda_vo AS parentX WHERE (nodeX.left BETWEEN parentX.left AND parentX.right) AND parentX.link_id = "+str(linkID)+") SELECT link_id, up_area/1e6 as up_area, area/1e6 as area, length/1000. as length" 
    text2 = "FROM pers_nico.master_lambda_vo WHERE link_id IN (SELECT * FROM subbasin)"
    if otherParams is None:
        if linkID>0:
            q = sql.SQL(text1+' '+text2) 
        elif linkID == 0:
            text = "SELECT DISTINCT link_id, up_area/1e6 as up_area, area/1e6 as area, length/1000. as length FROM pers_nico.master_lambda_vo as mas WHERE mas.model"
            q = sql.SQL(text)
    else:
        if linkID>0:
            for l in otherParams:
                text1+= ',' + l +' '
            q = sql.SQL(text1+text2) 
        elif linkID == 0:
            text1 = "SELECT DISTINCT link_id, up_area/1e6 as up_area, area/1e6 as area, length/1000. as length "
            text2 = "FROM pers_nico.master_lambda_vo AS mas WHERE mas.model"
            for l in otherParams:
                text1+= ',' + l +' '
            q = sql.SQL(text1+text2) 
    #Get the data
    BasinData = pd.read_sql(q, con, index_col='link_id')
    con.close()
    return BasinData.rename(columns={'up_area': "Acum", 'area':'Area', 'length':'Long'})


def WEB_usgs4assim(usgs_code, link, fi, ff):
    qu = WEB_Get_USGS(usgs_code, fi,ff)
    qu = qu.interpolate().resample('1H').mean()
    qu = qu.to_frame()
    qu.rename(columns={0:'discharge'}, inplace=True)
    qu['discharge'] = qu['discharge'].interpolate()
    u = list(map(aux.__datetime2unix__,qu.index))
    qu['link'] = link
    qu['unix_time'] = u
    return qu[['unix_time','discharge','link']]
    
def SQL_Write_stream4HLM(q, year,table_name, schema, fi = None, ff = None):
    if fi is None:
        fi = str(year) + '-01-01 00:00'
    if ff is None:
        ff = str(year) + '-12-31 23:59'
    #Reads the data from the web
    eng = sqlalchemy.create_engine("postgresql://"+data_usr+":"+data_pass+"@"+data_host+":"+data_port+"/"+data_base)
    conn = eng.connect()
    q.to_sql(table_name,conn, schema = schema, index = False, chunksize=1000, if_exists = 'append',
        dtype = {'unix_time': sqt.INTEGER,
        'discharge':sqt.FLOAT,
        'link':sqt.INTEGER})
    conn.close()