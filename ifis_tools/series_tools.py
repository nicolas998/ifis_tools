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

# # series_tools:
#
# set of tools that work with streamflow records.
# - Identify events.
# - Identidy baseflow and runoff.
#

import pandas as pd 
import numpy as np 
from read_dat import *
from scipy import stats as sta
from multiprocessing import Pool
import glob
from scipy.signal import find_peaks as __find_peaks__
from hydroeval import evaluator, kge, nse, pbias
# ## Digital filters
#
# Collection of functions to separate runoff from baseflow.

# +

def percentiles(obs, sim, steps = 10, bins = None, perc = 50, percMax = 99.5):
    '''Obtains the percentile for the sim value that corresponds to
    an oberved value.
    Parameters:
        -obs: observed peaks or time serie.
        -sim: simulated peaks or time serie.
        -steps: number of splits to estimate the percentile.
        -perc: percentile to estimate
        -percMax: Max value to divide the observations.
    Returns:
        -A vector (2,N) where the first row corresponds to the percentiles
        at the observation and the second to the percentiles at the simulation'''
    if bins is None:
        bins = np.linspace(obs.min(), np.percentile(obs, percMax), steps)
    X = []; Y = []
    for i,j in zip(bins[:-1], bins[1:]):
        Y.append(np.percentile(sim[(obs>i) & (obs<=j)], perc))
        X.append((i+j)/2.)
        #Y.append(np.percentile(sim[(obs>i) & (obs<=j)], perc))
    return np.vstack([X,Y])

def read_dat(path, date_start,state = 0, freq = '60min'):
    '''Read fast a .dat file produced by HLM and returns a pandas DataFrame
    Parameters:
        -path: path to the .dat file.
        -date_start: starting date of the simulation
        -State: model state.
        -freq: Time step of the .dat given in min (ej. 60min).'''
    Ncon, Nrec,Nstat = read_dat.get_size(path)
    cont, data = read_dat.get_data(path, Ncon, Nrec, Nstat)
    dates = pd.date_range(date_start, periods=Nrec, freq=freq)
    
    return pd.DataFrame(data[:,state,:].T, index=dates, columns=cont)


class performance:

    #Initialization and hide functions
    def __init__(self, temp_scale = '1H', links_prop = None, 
        prop_col_names = None, rain_path = None, rain_ending = ''):
        '''class for the performance analysos of model simulations
        Params:
        - temp_scale: temporal scale at which the analysis will be done.
        -links_prop: a data frame that must have link_id as index, and columns 
        related with the area and the travel time of the watershed.
        -prop_col_names: dictionary with the actual name in the link_prop and 
        the name for this tool. which are 'area' for the acumulated area and 
        'ttime' for the travel time
        -rain_path: the path to find the mean reainfall series of each watershed
            This is relevant for the some of the performance metrics, if not given they
            will be deactivated.
        -rain_ending: the name at the end of the rainfall files (if exist)'''
        #Define initial things of the class        
        self.analysis_dic = {}
        self.base_name = None
        self.temp_scale = temp_scale
        self.link_act = None
        self.tocompare = []
        #Get link props that is a dataFrame
        if links_prop is not None:
            self.link_prop = links_prop
            self.link_prop = self.link_prop.rename(columns = prop_col_names)[[prop_col_names[k] for k in prop_col_names.keys()]]
        #Rainfall path 
        if rain_path is not None:
            self.update_dic('rain', False, path = rain_path, abr = 'r', file_ending = rain_ending)
            self.__has_rain__ = True
        else:
            self.__has_rain__ = False
  
    def __mapVar__(self, x, new_min = 0, new_max = 1):
        '''Function to map any vatriable between a range'''
        return ((x-x.min())/(x.max()-x.min()))*(new_max - new_min) + new_min    

    def __intersection__(self, n_list1, n_list2): 
        '''Get the intersection of two links lists'''
        lst1 = self.analysis_dic[n_list1]['link_names']
        lst2 = self.analysis_dic[n_list2]['link_names'] 
        paths2 = self.analysis_dic[n_list2]['link_paths'] 
        valueN = []
        pathsN = []
        for value,path in zip(lst2, paths2):
            if value in lst1:
                valueN.append(value)
                pathsN.append(path)
        return valueN, pathsN 

    def __path2name__(self, paths):
        '''Extracts the link numbers from a list of paths.'''
        names = []
        for i in paths:
            j = i.split('/')[-1]
            names.append(''.join([s for s in j if s.isdigit()]))
        return names

    def __func_qpeakTimeDiff__(self,qo,qs, ttime = None):
        '''Calc the time difference between two hydrographs
        parameters:
            - qo: observed hydrograph.
            - qs: simulated hydrograph
            - ttime: reference time for the hydrograph
        returns:
            - dt: delta time difference, 
                - negative stands for time(sim) > time(obs): late arrival
                - positibve time(sim) < time(obs): early arrival'''
        to = qo.idxmax()
        ts = qs.idxmax()
        dt = to - ts
        dt = dt.seconds / 3600.
        if ttime is not None:
            dt = dt / ttime
        if to > ts:
            return dt*-1
        else:
            return dt

    def __func_qpeakMagDiff__(self,qo,qs):
        '''Calc the magnitude difference between two hydrographs'''
        qmo = qo.max()
        qms = qs.max()
        return (qmo - qms) / qmo

    def __func_qpeakMagDiff2(self, qo,qs):
        '''Same as func_qpeakMagDiff but put first the simulated'''
        qmo = qo.max()
        qms = qs.max()
        return (qms - qmo) / qmo

    def __func_HyM__(self, qo, qs, flood):
        '''Calculates the amount of Hits and misses compared to certain flood
        value'''
        a = qo[(qo >= flood) & (qs >= flood)].size
        c = qo[qo>=flood].size
        if c ==0:
            return np.nan
        else:
            return float(a)/float(c)

    def __func_KGE__(self, qo, qs):
        '''Gets the KGE for an event'''
        return evaluator(kge, qs.values, qo.values)[0][0]

    def __func_KGE_mean__(self, qo,qs):
        '''Gets the mean ratio difference of the KGE'''
        qo.dropna(inplace = True)
        qs.dropna(inplace = True)
        idx = qo.index.intersection(qs.index)
        return qs.loc[idx].mean() / qo.loc[idx].mean()

    def __func_KGE_std__(self, qo,qs):
        '''Gets the std ratio of the KGE'''
        qo.dropna(inplace = True)
        qs.dropna(inplace = True)
        idx = qo.index.intersection(qs.index)
        return qs.loc[idx].std() / qo.loc[idx].std()

    def __func_pbias__(self, qo, qs):
        '''Gets the Percent bias for an event'''
        return evaluator(pbias, qs.values, qo.values)[0] * (-1)

    def __func_nse__(self, qo, qs):
        '''Gets the Nash for an event'''
        return evaluator(nse, qs.values, qo.values)[0]

    def __func_qpeakTravelTime__(self, q):
        ttime = int(self.link_tt)*4
        dt = pd.Timedelta(str(ttime)+'H')
        peakMax = q.idxmax()
        max_r_idx = self.link_r[peakMax-dt:peakMax].idxmax()
        max_r_val = self.link_r[peakMax-dt:peakMax].max()
        ttime = peakMax - max_r_idx
        return max_r_val, ttime.seconds/3600.

    def __func_find_max_corr(self, qo,qs,w = 100):
        #Get only the data that is not nan
        qo.dropna(inplace=True)
        qs.dropna(inplace=True)
        idx = qo.index.intersection(qs.index)
        qo = qo[idx]
        qs = qs[idx]
        #find the best correlation 
        bestCorr = -9999
        bestMove = -9999
        zeroCorr = 0.0
        for move in np.arange(-w+1,w):
            ct = sta.pearsonr(qo.values[w:-w],qs.values[w+move:-w+move])[0]
            if ct > bestCorr:
                bestCorr = ct
                bestMove = move
            if move == 0:
                zeroCorr = ct
        return bestCorr, zeroCorr, bestMove


    def set_link2analyze(self, link, min4event = 'P90', link_tt = 30.):
        '''For a link read the data of the different options
        Parameters:
        - link: the link to analyze.
        - min4event: the minim value to consider an event, could be a number or
        the percentile (eg P50, P60, P90)
        - link_tt: traver time of the link is to determine the length of the hydrograph.
        Returns:
        - Updates the analysis_dic of the performance class'''
        self.link_act = link
        self.link_tt = link_tt
        for k in self.analysis_dic:
            pos = self.analysis_dic[k]['link_names'].index(str(link))
            #try:
            #reads the data
            q = pd.read_msgpack(self.analysis_dic[k]['link_paths'][pos])
            if self.analysis_dic[k]['isDataFrame']:
                q = q[self.analysis_dic[k]['DataFrameColumn']]
            self.analysis_dic[k]['data']['q'] = q.resample(self.temp_scale).mean()
            self.link_q = q.resample(self.temp_scale).mean()
            #rainfall data 
            if self.__has_rain__:
                self.link_r = pd.read_msgpack(self.analysis_dic['rain']['link_paths'][pos])
            #Gert the events for the base
            if k == self.base_name:
                #Set the type of minimum value
                if type(min4event) == int or type(min4event) == float:
                    qmin = min4event
                elif min4event.startswith('P'):
                    per = float(min4event[1:])
                    if per>1.0: per = per/100.
                    min4event = q[q.notnull()].quantile(per)      
                #Get tyhe events for that station
                pos,ph = __find_peaks__(q, min4event, distance = link_tt)
                self.analysis_dic[k]['data']['peaks'] = q.index[pos]
                self.link_mpeak = ph['peak_heights']
                self.link_tpeak = q.index[pos]
            #except:
                #If any error just put nan
             #   self.analysis_dic[k]['data']['q'] = np.nan

    #Function to update the main analysiis dic.
    def update_dic(self, name, base = False, path = 'not set', abr = 'not set', file_ending = '',
        path2linkFunc = None, isDataFrame = False, DataFrameColumn = ''):
        '''Creates a dictionary for the performance of the model analysis:
        Parameters:
            -name: name to the model run or observed serie.
            -base: is this an observation or not?.
            -path: directory with the msgpack series of the links.
            -abr: small name.
            -file_ending: text at the end of the files for that run
            -path2fileFunc: optinal function to extract the links from the given paths.
            -isDataFrame: if the data came from a data frame.
            -DataFrameColumn: the name of the column to extract from the data. 
        Returns (no explicit return):
            Updates the dictionary with the information for the analysis with
            series.'''
        #Defines if it is the base key
        if base:
            self.base_name = name
        else:
            self.tocompare.append(name)
        #Get the links and paths to the links of that class
        links_paths = glob.glob(path+'.msg')
        if path2linkFunc is None:
            links_names = self.__path2name__(links_paths)
        else:
            links_names = path2linkFunc(links_paths)
        #Updates the information of the performance dictionary
        self.analysis_dic.update({
            name: {'base': base,
                'path': path,
                'abr': abr,
                'file_ending': file_ending,
                'isDataFrame': isDataFrame,
                'DataFrameColumn': DataFrameColumn,
                'link_paths': links_paths,
                'link_names': links_names,
                'data':{
                    'q': None,
                    'peaks': None
                }},
        })
        #If has the base makes the intersect to hav just the good links
        if self.base_name is not None and name != self.base_name:
            # Gets the intersection of the links
            links_names, links_paths = self.__intersection__(self.base_name, name)
            self.analysis_dic[name]['link_paths'] = links_paths
            self.analysis_dic[name]['link_names'] = links_names
            #Updates the intersection of all the evaluable links
            self.links_eval = links_names

  #To make ealuations by events
    def eval_by_events(self, link = None, min4peak = None):
        '''Get the performance of multiple excecutions for all the events of that link'''

        if link is not None:
            args = {'link_tt': self.link_prop['ttime'][int(link)]}
            if min4peak is not None:
                args.update({'min4event' : min4peak})
            self.set_link2analyze(link, **args)
        area_link = self.link_prop.loc[float(link),'area'] / 1e6

        #Define list to fill 
        Dates = []
        Qpeak = []
        QpeakMDiff = []
        QpeakTDiff = []
        KGE = []
        PBIAS = []
        NASH = []
        product = []
        if self.__has_rain__:
            RainPeak = []
            TimePeak = []
        Qmean = []
        Area = []

        #Iterate in all the models
        for k in self.analysis_dic.keys():
            if k != 'rain':
                #Get data
                qs = self.analysis_dic[k]['data']['q']
                qo = self.analysis_dic[self.base_name]['data']['q']
                dt = pd.Timedelta(str(self.link_tt)+'H')
                qmean = qs.mean()

            for date in self.analysis_dic[self.base_name]['data']['peaks']:
                qot = qo[date-dt:date+dt]
                qst = qs[date-dt:date+dt]
                if self.__has_rain__:
                    rt = self.link_r[date-dt*4:date+dt]
                if len(qot)>0 and len(qst)>0:
                    good_o = qot[qot.notnull()].size / qot.size
                    good_s = qst[qst.notnull()].size / qst.size
                #Only makes the calculus if both series have more than half of their records
                if good_o > 0.5 and good_s > 0.5 and len(qot) == len(qst):
                    #Good date
                    Dates.append(date)
                    product.append(k)
                    Qmean.append(qmean)
                    Area.append(area_link)
                    #Get the performance of the qpeak max
                    QpeakMDiff.append(self.__func_qpeakMagDiff__(qot,qst))
                    Qpeak.append(np.nanmax(qst))
                    #Time performance            
                    travelDif = self.__func_qpeakTimeDiff__(qot, qst)
                    QpeakTDiff.append(travelDif)
                    #Get the oberved and simulated travel time
                    if self.__has_rain__:
                        i_max,tpeak = self.__func_qpeakTravelTime__(qst)
                        TimePeak.append(tpeak)
                        RainPeak.append(i_max)
                    #Overall performance
                    KGE.append(self.__func_KGE__(qot, qst))
                    PBIAS.append(self.__func_pbias__(qot, qst)*-1)
                    NASH.append(self.__func_nse__(qot, qst))

        #Set up to include or not to include rain analysis.
        if self.__has_rain__:
            columns = ['product','qpeak','qmean','Imax','qpeakDiff',
                       'tpeak','tpeakDiff','kge','nse', 'pbias','up_area']
            ListProducts = [product, Qpeak, Qmean,RainPeak, QpeakMDiff,
                            TimePeak,QpeakTDiff, KGE, NASH, PBIAS, Area]
            formats = {'qpeak':'float','tpeak':'float', 'qpeakDiff':'float','kge':'float', 'tpeakDiff':'float', 
                       'nse':'float', 'pbias':'float',
                       'Imax':'float','qmean':'float', 'up_area':'float'}
        else:
            columns = ['product','qpeak','qmean','qpeakDiff',
                       'tpeakDiff','kge','nse', 'pbias','up_area']
            ListProducts = [product, Qpeak, Qmean,QpeakMDiff,
                            QpeakTDiff, KGE, NASH, PBIAS, Area]
            formats = {'qpeak':'float','qpeakDiff':'float','kge':'float', 'tpeakDiff':'float', 
                       'nse':'float',
                       'pbias':'float','qmean':'float','up_area':'float'}
        #Convert to a Data frame with the results.
        D = pd.DataFrame(np.array(ListProducts).T, index = Dates, columns = columns, )
        D = D.astype(formats)
        D['link'] = self.link_act
        return D

    def eval_years(self, link = None, usgs = None, fi = '',
                   flood = None, Pflood = 96, window = 100):
        '''Eval the performance of the model results for every year'''

        if link is not None:
            self.set_link2analyze(link)

        Vol = []
        KGE = []
        NSE = []
        Hits = []
        Misses = []
        Pbias = []
        PD = []
        QP = []
        Years = []
        Product = []
        bestCorr = []
        zeroCorr = []
        bestMove = []
        meanRatio = []
        stdRatio = []
        for k in self.analysis_dic.keys():
            qs = self.analysis_dic[k]['data']['q']
            qo = self.analysis_dic[self.base_name]['data']['q']

            idx = qo.index.intersection(qs.index)
            qot = qo[idx]
            qst = qs[idx]
            qA = qot.resample('A').mean()

            if flood is None:
                flood = np.percentile(qo[qo>0], Pflood)

            for y in qA.index.year.values:
                #Position of the numeric values
                p = np.where((np.isnan(qot[str(y)]) == False) & (np.isnan(qst[str(y)]) == False))[0]
                #Performance
                KGE.append(self.__func_KGE__(qot[str(y)][p],qst[str(y)][p]))
                NSE.append(self.__func_nse__(qot[str(y)][p],qst[str(y)][p]))
                Pbias.append(self.__func_pbias__(qot[str(y)][p],qst[str(y)][p]))
                Hits.append(self.__func_HyM__(qot[str(y)][p],qst[str(y)][p],flood))
                PD.append(self.__func_qpeakMagDiff2(qot[str(y)][p],qst[str(y)][p]))
                #Mean sn std ratios
                meanRatio.append(self.__func_KGE_mean__(qot[str(y)],qst[str(y)]))
                stdRatio.append(self.__func_KGE_std__(qot[str(y)],qst[str(y)]))
                #Best correlation and correlatoin
                try:
                    beCorr, zeCorr, beMove = self.__func_find_max_corr(qo[str(y)],qs[str(y)],window)
                    bestCorr.append(beCorr)
                    zeroCorr.append(zeCorr)
                    bestMove.append(beMove)
                except:
                    bestCorr.append(0.0)
                    zeroCorr.append(0.0)
                    bestMove.append(0)
                #Prop
                Vol.append(qst[str(y)][p].sum())
                QP.append(qst[str(y)][p].max())
                #Add the year
                Years.append(y)
                Product.append(k)

        #Convert to a Data frame with the results.
        ListProducts = [Product, KGE, NSE, Vol, Pbias, Hits, PD, QP,
                       zeroCorr, bestCorr, bestMove, meanRatio,
                       stdRatio]
        columns = ['product','kge','nse','vol','pbias', 'Hits',
                   'PeakDif','Qpeak',
                  'corr','best_corr','moves','meanRatio','stdRatio']
        formats = {'kge':'float', 'nse':'float','vol':'float','pbias':'float',
                   'Hits':'float', 'PeakDif':'float','Qpeak':'float',
                  'corr':'float','best_corr':'float','moves':'int',
                   'meanRatio':'float','stdRatio':'float'}
        D = pd.DataFrame(np.array(ListProducts).T, index = Years, columns = columns, )
        D = D.astype(formats)
        D['Misses'] = 1 - D['Hits']
        D['link'] = self.link_act
        if usgs is not None:
            D['usgs'] = usgs
        return D

    # class performance:

    #     def __init__(self):
    #         '''class for the performance analysos of model simulations'''        
    #         self.analysis_dic = {}

    #     def mapVar(self, x, new_min = 0, new_max = 1):
    #         '''Function to map any vatriable between a range'''
    #         return ((x-x.min())/(x.max()-x.min()))*(new_max - new_min) + new_min

    #     def func_qpeakTimeDiff(self,qo,qs, ttime = None):
    #         '''Calc the time difference between two hydrographs
    #         parameters:
    #             - qo: observed hydrograph.
    #             - qs: simulated hydrograph
    #             - ttime: reference time for the hydrograph
    #         returns:
    #             - dt: delta time difference, 
    #                 - negative stands for time(sim) > time(obs): late arrival
    #                 - positibve time(sim) < time(obs): early arrival'''
    #         to = qo.idxmax()
    #         ts = qs.idxmax()
    #         dt = to - ts
    #         dt = dt.seconds / 3600.
    #         if ttime is not None:
    #             dt = dt / ttime
    #         if to > ts:
    #             return dt*-1
    #         else:
    #             return dt

    #     def func_qpeakMagDiff(self,qo,qs):
    #         '''Calc the magnitude difference between two hydrographs'''
    #         qmo = qo.max()
    #         qms = qs.max()
    #         return (qmo - qms) / qmo

    #     def update_dic(self, name, base = False, path = 'not set', abr = 'not set', file_ending = '',
    #         isDataFrame = False, DataFrameColumn = ''):
    #         '''Creates a dictionary for the performance of the model analysis:
    #         Parameters:
    #             -name: name to the model run or observed serie.
    #             -base: is this an observation or not?.
    #             -path: directory with the msgpack series of the links.
    #             -abr: small name.
    #             -file_ending: text at the end of the files for that run
    #             -isDataFrame: if the data came from a data frame.
    #             -DataFrameColumn: the name of the column to extract from the data. 
    #         Returns (no explicit return):
    #             Updates the dictionary with the information for the analysis with
    #             series.'''
    #         self.analysis_dic.update({
    #             name: {'base': base,
    #                 'path': path,
    #                 'abr': abr,
    #                 'file_ending': file_ending,
    #                 'isDataFrame': isDataFrame,
    #                 'DataFrameColumn': DataFrameColumn},
    #         })

    #     def percentiles(self, obs, sim, steps = 10, bins = None, perc = 50, percMax = 99.5):
    #         '''Obtains the percentile for the sim value that corresponds to
    #         an oberved value.
    #         Parameters:
    #             -obs: observed peaks or time serie.
    #             -sim: simulated peaks or time serie.
    #             -steps: number of splits to estimate the percentile.
    #             -perc: percentile to estimate
    #             -percMax: Max value to divide the observations.
    #         Returns:
    #             -A vector (2,N) where the first row corresponds to the percentiles
    #             at the observation and the second to the percentiles at the simulation'''
    #         if bins is None:
    #             bins = np.linspace(obs.min(), np.percentile(obs, percMax), steps)
    #         X = []; Y = []
    #         for i,j in zip(bins[:-1], bins[1:]):
    #             Y.append(np.percentile(sim[(obs>i) & (obs<=j)], perc))
    #             X.append((i+j)/2.)
    #             #Y.append(np.percentile(sim[(obs>i) & (obs<=j)], perc))
    #         return np.vstack([X,Y])

    #     def perform_analysis(self, link, qpeakmin, keys = None, yi = 2012, yf = 2018):
    #         '''Produce the performance analysis of several simulation by taking a 
    #         dictionary with the information of the series names, path and abreviation.
    #         Parameters:
    #             -ForAnalysis: Dictionary with the structure given by Dic4Analysis.
    #             -link: number of the link to be analyzed, the link must be saved in a msgpack
    #                 format under the path given by the dictionary of ForAnalysis.
    #             -qpeakmin: minimum value to consider a qpeak.
    #             -keys: (optional) if not given it takes the keys from the dictionary.
    #             - yi and yf: initial and end year of the anlysis.
    #         Returns:
    #             - Metrics: DataFrame with the metrics for the models.
    #             - QpeakAnalysis: DataFrmae with the metrics of all the peaks found.'''
    #         #Define function to obtain median value
    #         def median(x):
    #             try:
    #                 return np.percentile(x, 50)
    #             except:
    #                 pass 
    #         def FindMax(Serie, Smax, tw = '1W'):
    #             index = []
    #             values = []
    #             shared = []
    #             dt = pd.Timedelta(tw)
    #             for i in Smax.index:
    #                 try:
    #                     val = Serie[i-dt:i+dt].max()
    #                     if val > 0:
    #                         index.append(Serie[i-dt:i+dt].idxmax())
    #                         values.append(val)
    #                         shared.append(i)
    #                 except:
    #                     pass
    #             return pd.Series(values, index), shared       
    #         #Read the data
    #         if keys is None:
    #             keys = [k for k in self.analysis_dic.keys()]
    #         yf += 1
    #         Data = {}
    #         for k in keys:
    #             if self.analysis_dic[k]['isDataFrame']:
    #                 q = pd.read_msgpack(self.analysis_dic[k]['path']+str(link)+self.analysis_dic[k]['file_ending']+'.msg')
    #                 q = q[self.analysis_dic[k]['DataFrameColumn']]
    #                 print(q.type)
    #             else:
    #                 q = pd.read_msgpack(self.analysis_dic[k]['path']+str(link)+self.analysis_dic[k]['file_ending']+'.msg')
    #             #Updates data in a dictionary
    #             D = {k: {'q': q,
    #                     'base': self.analysis_dic[k]['base'],
    #                     'abr': self.analysis_dic[k]['abr']}}
    #             #If it is the base obtains the peak values
    #             if self.analysis_dic[k]['base']:
    #                 qo = q.copy()
    #                 qo_max = Events_Get_Peaks(q, qpeakmin, tw = pd.Timedelta('5d'))
    #                 D[k].update({'qmax': qo_max})
    #             Data.update(D)
    #         #Produce the analysis
    #         Metrics = {}    
    #         for k in keys:
    #             if self.analysis_dic[k]['base'] is False:
    #                 #Obtains the peaks in the simulation
    #                 key_sim = k
    #                 qs = Data[k]['q']
    #                 qs_max, sh = FindMax(qs, qo_max)
    #                 #Obtain peak differences metrics 
    #                 qpeakMagDiff = (qo_max[sh].values - qs_max.values) / qo_max[sh].values
    #                 #Differences at the time to peak
    #                 qpeakTimeDiff = qo_max[sh].index - qs_max.index
    #                 pos = np.where(qo_max[sh].index > qs_max.index)[0]
    #                 qpeakTimeDiff = qpeakTimeDiff.seconds / 3600.
    #                 qpeakTimeDiff = qpeakTimeDiff.values
    #                 qpeakTimeDiff[pos] = -1*qpeakTimeDiff[pos]
    #                 #Yearly differences
    #                 kge_val = []
    #                 VolDif = []
    #                 Qannual = []
    #                 Dtannual = []
    #                 Anos = []
    #                 qpAnnual = pd.Series(qpeakMagDiff, index=qo_max[sh].index)
    #                 qpAnnual = qpAnnual.resample('A').apply(median)
    #                 dtAnnual = pd.Series(qpeakTimeDiff, index=qo_max[sh].index)
    #                 dtAnnual = dtAnnual.resample('A').apply(median)            
    #                 for i in range(yi,yf):
    #                     pos = np.where(np.isnan(qs[str(i)]) == False)[0]
    #                     qsimOver = qo[qs[str(i)].index[pos]].values
    #                     kge_temp = evaluator(kge, qs[str(i)].values[pos], qo[qs[str(i)].index[pos]].values)[0][0]
    #                     voldif_temp = ((np.nansum(qsimOver)*3600/1e6) -(qs[str(i)].sum()*3600/1e6)) / (np.nansum(qsimOver)*3600/1e6)
    #                     if kge_temp > -999:
    #                         try:
    #                             #print(i,qpAnnual[str(i)].values[0])
    #                             Qannual.append(qpAnnual[str(i)].values[0])
    #                             Dtannual.append(dtAnnual[str(i)].values[0])
    #                         except:
    #                             Qannual.append(np.nan)
    #                             Dtannual.append(np.nan)
    #                         kge_val.append(kge_temp)
    #                         VolDif.append(voldif_temp)
    #                         Anos.append(pd.Timestamp(year = i, month = 12, day = 31))            
    #             else:            
    #                 key_base = k
    #                 qs_max = qo_max.copy()
    #                 sh = qs_max.index
    #                 qpeakMagDiff = np.zeros(qs_max.size)
    #                 qpeakTimeDiff = np.zeros(qs_max.size)
    #                 kge_val = np.ones(np.arange(yi,yf).size) 
    #                 VolDif =  np.zeros(np.arange(yi,yf).size)
    #                 Qannual = np.zeros(np.arange(yi,yf).size)
    #                 Dtannual = np.zeros(np.arange(yi,yf).size)
    #                 Anos = []
    #                 for i in range(yi,yf):
    #                     Anos.append(pd.Timestamp(year = i, month = 12, day = 31))
                    
    #             #Update the dictionary.
    #             MetQp = np.vstack([qs_max, qpeakMagDiff, qpeakTimeDiff]).T
    #             idxQp = qo_max[sh].index
    #             columnsQp = ['qpeak','qpeakMagDif', 'qpeakTimeDif']
    #             MetEff = np.vstack([kge_val, VolDif, Qannual, Dtannual]).T
    #             idxEff = Anos
    #             columnsEff = ['kge', 'VolDif', 'qpeakMagDif', 'qpeakTimeDif']
    #             Metrics.update({
    #                 k:{
    #                     'link': link,
    #                     'qpeakMetrics': pd.DataFrame(MetQp, index = idxQp, columns = columnsQp),
    #                     'effMetrics': pd.DataFrame(MetEff, index = idxEff, columns = columnsEff)
    #                 }
    #             })
                
    #         #Converts everything into a DataFrame
    #         usgs_idx = Metrics[key_base]['qpeakMetrics'].index
    #         model_idx = Metrics[key_sim]['qpeakMetrics'].index
    #         shared_idx = model_idx.intersection(usgs_idx)
    #         Metrics[key_base]['qpeakMetrics'] = Metrics[key_base]['qpeakMetrics'].loc[shared_idx]
    #         initial = 'usgs'
    #         QpeakAnalysis = Metrics[key_base]['qpeakMetrics']
    #         QpeakAnalysis['model'] = key_base
    #         QpeakAnalysis['link'] = link
    #         for k in Metrics.keys():
    #             if k != initial:
    #                 a = Metrics[k]['qpeakMetrics']
    #                 a['model'] = k
    #                 a['link'] = link
    #                 QpeakAnalysis = QpeakAnalysis.append(a)
    #         #Converts metrics into a dataFrame
    #         MetrData = Metrics[key_base]['effMetrics']
    #         MetrData['model'] = key_base
    #         MetrData['link'] = link
    #         for k in Metrics.keys():
    #             if k != initial:
    #                 a = Metrics[k]['effMetrics']
    #                 a['model'] = k
    #                 a['link'] = link
    #                 MetrData = MetrData.append(a)
                
    #         return MetrData, QpeakAnalysis





def DigitalFilters(Q,tipo = 'Eckhart', a = 0.98, BFI = 0.8):
    '''Digital filters to separate baseflow from runoff in a continuos time series.
    Parameters:
        - tipo: type of filter to be used.
            - Eckhart o 1.
            - Nathan o 2.
            - Chapman o 3.
        - Q: pandas series with the streamflow records.
        - a: paramter for the filter.
            - Eckhart: 0.98.
            - Nathan: 0.8.
            - Chapman: 0.8.
        - BFI: 0.8 only applies for Eckhart filter.
    Returns:
        - Pandas DataFrame with the Runoff, Baseflow.'''
    #Functions definitions.
    def Nathan1990(Q, a = 0.8):
        '''One parameter digital filter of Nathan and McMahon (1990)'''
        R = np.zeros(Q.size)
        c = 1
        for q1,q2 in zip(Q[:-1], Q[1:]):
            R[c] = a*R[c-1] + ((1+a)/2.)*(q2-q1)
            if R[c]<0: 
                R[c] = 0
            elif R[c]>q2:
                R[c] = q2 
            c += 1
        B = Q - R
        return R, B

    def Eckhart2005(Q, BFI=0.8, a = 0.98):
        '''Two parameter Eckhart digital filter
        Parameters:
            - Q: np.ndarray with the streamflow records.
            - BFI: The maximum amount of baseflow (%).
            - a: parameter alpha (0.98)
        Output: 
            - R: total runoff.
            - B: total baseflow.'''
        #SEparation
        B = np.zeros(Q.size)
        B[0] = Q[0]
        c = 1
        for q in Q[1:]:
            #SEparation equation
            B[c] = ((1.0-BFI)*a*B[c-1]+(1.0-a)*BFI*q)/(1.0-a*BFI)
            #Constrains
            if B[c] > q:
                B[c] = q
            c+=1
        R = Q - B
        return R, B

    def ChapmanMaxwell1996(Q, a = 0.98):
        '''Digital filter proposed by chapman and maxwell (1996)'''
        B = np.zeros(Q.size)
        c = 1
        for q in Q[1:]:
            B[c] = (a / (2.-a))*B[c-1] + ((1.-a)/(2.-a))*q
            c+=1
        R = Q-B
        return R,B
    
    #Cal the filter 
    if tipo == 'Eckhart' or tipo == 1:
        R,B = Eckhart2005(Q.values, a, BFI)
    elif tipo =='Nathan' or tipo == 2:
        R,B = Nathan1990(Q.values, a,)
    elif tipo == 'Chapman' or tipo ==3:
        R,B = ChapmanMaxwell1996(Q.values, a)
    #Returns the serie
    return pd.DataFrame(np.vstack([R,B]).T, index = Q.index, columns = ['Runoff','Baseflow']) 


# -

# ## Events selection functions
#
# Collection of functions to identify peaks in a series and the end of each peak recession.

# +
def Events_Get_Peaks(Q, Qmin = None, tw = pd.Timedelta('12h')):
    '''Find the peack values of the hydrographs of a serie
    Params:
        - Q: Pandas serie with the records.
        - Qmin: The minimum value of Q to be considered a peak.
            if None takes the 99th percentile of the series as the min
        - tw: size of the ime window used to eliminate surrounding maximum values'''
    if Qmin is None:
        Qmin = np.percentile(Q.values[np.isfinite(Q.values)], 99)
    #Find the maximum
    Qmax = Q[Q>Qmin]
    QmaxCopy = Qmax.copy()
    #Search the maxium maximorums
    Flag = True
    PosMax = []
    while Flag:
        MaxIdx = Qmax.idxmax()
        PosMax.append(MaxIdx)
        Qmax[MaxIdx-tw:MaxIdx+tw] = -9
        if Qmax.max() < Qmin: Flag = False
    #Return the result
    return QmaxCopy[PosMax].sort_index()

def Events_Get_End(Q, Qpeaks):
    '''Obtains the end of each event extracted by Events_Get_Peaks'''
    #Expands the Qpeaks
    Qpeaks = Qpeaks.append(Q[-1:])
    Qpeaks.values[-1] = Qpeaks.max()
    #Finds the ends
    dates_min = []
    val_mins = []
    for star, end in zip(Qpeaks.index[:-1], Qpeaks.index[1:]):
        dates_min.append(Q[star:end].idxmin())
        val_mins.append(Q[star:end].min())
    return pd.Series(val_mins, dates_min)

# def Events_Get_End(Q, Qmax, minDif = 0.04, minDistance = None,maxSearch = 10, Window = '1h'):
#     '''Find the end of each selected event in order to know the 
#     longitude of each recession event.
#     Parameters: 
#         - Q: Pandas series with the records.
#         - Qmax: Pandas series with the peak streamflows.
#         - minDif: The minimum difference to consider that a recession is over.
#     Optional:
#         - minDistance: minimum temporal distance between the peak and the end.
#         - maxSearch: maximum number of iterations to search for the end.
#         - Widow: Size of the temporal window used to smooth the streamflow 
#             records before the difference estimation (pandas format).
#     Returns: 
#         - Qend: The point indicating the en of the recession.'''
#     #Obtains the difference
#     X = Q.resample('1h').mean()
#     dX = X.values[1:] - X.values[:-1]
#     dX = pd.Series(dX, index=X.index[:-1])
#     #Obtains the points.
#     DatesEnds = []
#     Correct = []
#     for peakIndex in Qmax.index:
#         try:
#             a = dX[dX.index > peakIndex]
#             if minDistance is None:
#                 DatesEnds.append(a[a>minDif].index[0])
#                 Correct.append(0)
#             else:
#                 Dates = a[a>minDif].index
#                 flag = True
#                 c = 0
#                 while flag:
#                     distancia = Dates[c] - peakIndex
#                     if distancia > minDistance:
#                         DatesEnds.append(Dates[c])
#                         flag= False
#                         Correct.append(0)
#                     c += 1
#                     if c>maxSearch:
#                         flag = False
#                         Correct.append(1)
#         except:            
#             Correct.append(1)
#     #Returns the pandas series with the values and end dates 
#     Correct = np.array(Correct)
#     return pd.Series(Q[DatesEnds], index=DatesEnds), Qmax[Correct == 0]


# -

# ## Runoff analysis 

# +
def Runoff_SeparateBaseflow(Qobs, Qsim):
    '''From observed records obtain the baseflow and runoff streamflow records.
    Parameters:
        - Qobs: Observed record dt < 1h.
        - Qsim: Simulated records dt < 1h.
    Returns: 
        - Qh: Observed records at hourly scale.
        - Qsh: Simulated records at a hourly scale.
        - Qsep: Observed separated records at hourly scale'''
    #Observed series to hourly scale.
    Qh = Qobs.resample('1h').mean()
    Qh.interpolate(method='linear',inplace=True)
    Qsep = DigitalFilters(Qh, tipo = 'Nathan', a = 0.998)
    #Pre-process of simulated series to hourly scale.
    Qsh = Qsim.resample('1h').mean()
    Qsh[np.isnan(Qsh)] = 0.0
    #Return results
    return Qh, Qsh, Qsep

def Runoff_FindEvents(Qobs, Qsim, umbral = None,minTime = 1, minConcav = None, minPeak = None):
    '''Separates runoff from baseflow and finds the events.
    Parameters:
        - Qobs: Hourly obseved streamflow.
        - Qsim: Hourly simulated streamflow.
        - minTime: minimum duration of the event.
        - minConcav: minimum concavity of the event.
        - minPeak: minimum value of the peakflows.
    Returns: 
        - pos1: pandas index lists with the initial positions.
        - pos2: pandas index lists with the end positions.'''
    #Obtain the positions of the start and 
    if umbral is None:
        umbral = np.percentile(Qobs[np.isfinite(Qobs)], 20)
    pos1, pos2 = __Runoff_Get_Events__(Qsim, umbral)
    pos1, pos2 = __Runoff_Del_Events__(Qobs, pos1, pos2, minTime=1, minConcav=minConcav, minPeak = minPeak)
    #Returns results 
    return pos1, pos2

def Runoff_CompleteAnalysis(Area, Qobs, Rain, Qsep, pos1, pos2, N=None, Nant = None):
    '''Obtains the DataFrame with the resume of the RC analysis.
    Parameters:
        - Area: the area of the basin in km2.
        - Qobs: Hourly observed streamflow.
        - Rain: Hourly rainfall.
        - Qsep: Hourly dataFrame with the separated flows.
        - pos1: pandas index lists with the initial positions.
        - pos2: pandas index lists with the end positions.
        - N: Number of days to eval the rainfall between p1-N: p2.
        - Nant: Number of antecedent days to eval the rainfall between p1-Nant : p1-N.
    Results:
        - DataFrame with the columns: RC, RainEvent, RainBefore, RainInt, Qmax'''
    #Search for N
    if N is None:
        #Time window based on the basin area.
        N = Area**0.2
        N = np.floor(N) // 2 * 2 + 1
        if N<3: N = 3
        if N>11: N = 11        
        Ndays = pd.Timedelta(str(N)+'d')
        if Nant is None:
            Nant = pd.Timedelta(str(N+3)+'d')
    else:
        Ndays = N
        if Nant is None:
            Nant = N + pd.Timedelta('3d')
            
    #Lists of data
    RC = []
    RainTot = []
    Date = []
    Qmax = []
    RainInt = []
    RainAnt = []

    #Get Values for events
    for pi,pf in zip(pos1, pos2):
        #General variables obtention
        Runoff = Qsep['Runoff'][pi:pf+Ndays].sum()*3600.
        Rainfall = (Rain[pi-Ndays:pf].sum()/1000.)*(Area*1e6)
        #Runoff and streamflow List updates
        Qmax.append(Qobs[pi:pf].max())
        RC.append(Runoff / Rainfall)
        #Rainfall list updates
        RainTot.append(Rain[pi-Ndays:pf].sum())
        RainInt.append(Rain[pi-Ndays:pf].max())
        RainAnt.append(Rain[pi-Ndays-Nant:pi-Ndays].sum())
        #Dates.
        Date.append(pi)
    #Converts to arrays
    RC = np.array(RC)
    RainTot = np.array(RainTot)
    RainInt = np.array(RainInt)
    RainAnt = np.array(RainAnt)
    Date = np.array(Date)
    Qmax = np.array(Qmax)
    #Select the correct values
    p1 = np.where(np.isfinite(RC))[0]
    p2 = np.where((RC[p1]<=1.0) & (RC[p1]>0.0))[0]
    #Lo que es 
    RC = RC[p1[p2]]
    RainTot = RainTot[p1[p2]]
    RainInt = RainInt[p1[p2]]
    RainAnt = RainAnt[p1[p2]]
    Date = Date[p1[p2]]
    Qmax = Qmax[p1[p2]]
    #Los malos 
    pos = np.where((RC>0.04) & (RainTot<10))[0]
    #Depura de nuevo 
    RC = np.delete(RC, pos)
    RainTot = np.delete(RainTot, pos)
    RainInt = np.delete(RainInt, pos)
    RainAnt = np.delete(RainAnt, pos)
    Date = np.delete(Date, pos)
    Qmax = np.delete(Qmax, pos)
    #Turns things into a DataFrame
    Data = pd.DataFrame(
        np.vstack([RC, RainTot, RainAnt, RainInt, Qmax]).T,
        index= Date, 
        columns=['RC', 'RainEvent', 'RainBefore','RainInt','Qmax'])
    return Data

def Runoff_groupByRain(D, groupby = 'RainEvent' , bins = None,
    Vmin=None, Vmax=None, Nb = 10, logx = True):
    '''Group the values of RC in function of a variable.
    Parameters:
        - D: pandas Dataframe with the results from the RC analysis.
        - groupby: name of the column to use for the groups.
        - Vmin: minimum value to set the groups.
        - Vmax: max value to set the groups.
        - b: number of bins.
        - logx: use or not logaritmic X axis.
    Results:
        - Dictionary with the RC by groups, P25, P50, P90, mean value of the variable
        for grouping, Variable for groups.'''
    #Change if the axis X is logarithm or not
    if logx:
        x = np.log(D[groupby])
    else:
        x = D[groupby]
    #SEt max y min 
    if Vmin is None: Vmin = x.min()
    if Vmax is None: Vmax = x.max()
    #SEt the intervals
    if bins is None:
        b = np.linspace(Vmin, Vmax, Nb)
    else:
        b = bins
    #Make the groups
    DicStats = {'RC':[],'P25':[],'P75':[],'P50':[], 'X': [], groupby: []}    
    for i,j in zip(b[:-1], b[1:]):
        p = np.where((x>=i) & (x<=j))[0]
        if p.size > 0:
            DicStats['RC'].append(D['RC'][p])
            DicStats['P25'].append(np.percentile(D['RC'][p], 25))
            DicStats['P50'].append(np.percentile(D['RC'][p], 50))
            DicStats['P75'].append(np.percentile(D['RC'][p], 75))
            DicStats['X'].append((i+j)/2.)
            DicStats[groupby].append(x[p])
    return DicStats
#-------------------------------------------------------------------------------------------
## Backgroudn functions.

def __Runoff_Get_Events__(Q, Umbral):
    '''Obtais the initia and end dates of the events related to 
    a time series based on the results from the Asynch 190.
    Parameters:
        - Q: pandas series with the streamflow (simulated from asynch 190 no infiltration).
        - perc: percentile used to stablish runoff occurrence.
    Returns:
        - pos1: initial date of each event.
        - pos2: end date of each event'''    
    #Treshold and positions with values over it
    pos = np.where(Q.values > Umbral)[0]
    #Positions start and end.
    Dpos = pos[1:] - pos[:-1]
    Dpos1 = pd.Series(Dpos, Q.index[pos[1:]])
    pos1 = Dpos1[Dpos1>1].index
    pos1 = pos1.insert(0, Q.index[pos][0])
    pos1 = pos1[:-1]
    Dpos2 = pd.Series(Dpos, Q.index[pos[:-1]])
    pos2 = Dpos2[Dpos2>1].index
    #returns results 
    return pos1, pos2

def __Runoff_Get_eventsPeaks__(Q, pos1, pos2):
    '''Obtains the peaks of the observed events selected by the 
    criteria of the asynch 190 model
    PArameters:
        - Q: Pandas series qwith the observed data.
        - pos1: list with the start of the peaks.
        - pos2: list with the end of the peaks.
    Returns:
        - List with the peaks corresponding to the events.'''
    #Peak at each event 
    Peaks = []
    for p1, p2 in zip(pos1, pos2):
        Peaks.append(np.nanmax(Q[p1:p2].values))
    return Peaks

def __Runoff_Del_Events__(Q, pos1, pos2, minTime = 2.5, minPeak = None, minConcav = None):
    '''Eliminates events from the selected initial peaks based on different 
    aspects such as min time of the event, min peak and the concativity.
    Parameters:
        - Q: pandas series with the observed streamflow.
        - pos1: Pandas indexes with the start of the events.
        - pos2: Pandas indexes with the end of the events.
        - minTime: minimum time (days) of the duration  of the hydrographs.
        - minPeak: minim value of the peak at the hydrographs.
        - minConcat: minimum concativity for the hydrograph (suggested: 10).
    Returns:
        - starts: pandas index with the corrected starts.
        - ends: pandas indexes with the corrected ends.'''
    #Eliminates events based on their duration
    if minTime is not None:
        #Obtains the duration
        Td = pos2 - pos1
        Td = Td.total_seconds()/(3600*24)
        Td = Td.values
        #Eliminates
        p = np.where(Td<minTime)[0]
        pos1 = pos1.delete(p)
        pos2 = pos2.delete(p)
    #Eliminates events based on the peak flow
    if minPeak is not None:
        #Obtains peaks
        Peaks = __Runoff_Get_eventsPeaks__(Q, pos1, pos2)
        Peaks = np.array(Peaks)
        #Eliminates
        p = np.where(Peaks<minPeak)[0]
        pos1 = pos1.delete(p)
        pos2 = pos2.delete(p)
    #Eliminates events based on the concavity criterion
    if minConcav is not None:
        #Obtains the concativity series 
        Concav = Q.resample('5h').mean().diff(2)
        Concav = __Runoff_Get_eventsPeaks__(Concav, pos1, pos2)
        #Eliminates
        p = np.where(np.array(Concav)<minConcav)[0]
        pos1 = pos1.delete(p)
        pos2 = pos2.delete(p)
    #Returns the result
    return pos1, pos2


# -

# ## Recession analysis 

# +
#Function to obtain a
def Recession_NDF_method(l):
    '''l[0]: np.ndarray of the streamflow data.
    l[1]: parameter B between 0 and 5'''
    
    # Function to obtains A for a given B (l[1])
    def Estimate_A(Q,B,dt):
        e1 = np.nansum((Q.values[:-1] - Q.values[1:]))
        e2 = dt * np.nansum(((Q.values[:-1] - Q.values[1:])/2.)**B)        
        return e1/e2
    
    # Estimates Q for the pair B and A    
    def Estimate_Q(Q, B, A):
        '''Obtaines the estimated Q for a given A and B
        Parameters:
        - Qo: the initial value of the analyzed peak.
        - t: Vector with the elapsed time.'''
        #Convert time vector to elapsed time in seconds.
        t = Q.index.astype('int64') / 1e9
        t = (t.values - t.values[0])/3600.
        Qo = Q.values[0]
        # Obtains the estimted Qs
        return Qo * (1 - ( (1.-B)*A*t / Qo**(1.-B) )) ** (1./(1.-B))
    
    def Estimate_error(Qobs, Qsim):
        '''Estimates the total percentage error obtained with the pair
        A and B'''
        Vsim = Qsim.sum()
        Vobs = Qobs.sum()
        return (Vsim - Vobs) / Vsim
    
    #Obtains the time delta 
    dt = l[0].index[1] - l[0].index[0]
    dt = dt.value / 1e9
    #Estimates A 
    A = Estimate_A(l[0],l[1],dt)
    #Estimaest Q
    Qsim = Estimate_Q(l[0],l[1], A)
    CountNaN = Qsim[np.isnan(Qsim)].size
    #Estimate error
    if CountNaN  == 0:
        E = Estimate_error(l[0],Qsim)
    else:
        E = 1000
    return A, E, Qsim

# search B for recession 
def Recession_Search_NDF(Q,Initial = 0, Long=1 ,process = 8, Window = 1, step = 0.01):
    '''Search for the optimum value of B and A for a hydrograph
    Parameters:
        - Initial: Initial point oscillates between 0 and 168h.
        - Long: recession longitude oscillates between 4 and 12 days.
        - process: total number of processors to do the analysis.'''
    #Movement of the initial and finish time 
    dis_i = pd.Timedelta(hours = Initial)
    dis_f = pd.Timedelta(hours = 24*Long)
    #Take a portion of the recession curve
    X = Q[Q.idxmax()+dis_i:Q.idxmax()+dis_f+dis_i]
    # Excercise to obtain A and B for a streamflow record.
    L = []
    B = np.arange(0, 5., step)
    for b in B:
        L.append([X, b])
    p = Pool(processes=process)
    Res = p.map(NDF_method, L)
    p.close()
    p.join()
    #Error selection
    Error = np.abs([i[1] for i in Res])
    PosEr = np.argmin(Error)
    #Return: B, A, E and Qsim
    return B[PosEr], Res[PosEr][0], Error[PosEr], pd.Series(Res[PosEr][2], X.index)
# -


