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

# # series_tools:
#
# set of tools that work with streamflow records.
# - Identify events.
# - Identidy baseflow and runoff.
#

import pandas as pd 
import numpy as np 


# ## Digital filters
#
# Collection of functions to separate runoff from baseflow.

# +

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

def Events_Get_End(Q, Qmax, minDif = 0.04, minDistance = None,maxSearch = 10, Window = '1h'):
    '''Find the end of each selected event in order to know the 
    longitude of each recession event.
    Parameters: 
        - Q: Pandas series with the records.
        - Qmax: Pandas series with the peak streamflows.
        - minDif: The minimum difference to consider that a recession is over.
    Optional:
        - minDistance: minimum temporal distance between the peak and the end.
        - maxSearch: maximum number of iterations to search for the end.
        - Widow: Size of the temporal window used to smooth the streamflow 
            records before the difference estimation (pandas format).
    Returns: 
        - Qend: The point indicating the en of the recession.'''
    #Obtains the difference
    X = Q.resample('1h').mean()
    dX = X.values[1:] - X.values[:-1]
    dX = pd.Series(dX, index=X.index[:-1])
    #Obtains the points.
    DatesEnds = []
    Correct = []
    for peakIndex in Qmax.index:
        try:
            a = dX[dX.index > peakIndex]
            if minDistance is None:
                DatesEnds.append(a[a>minDif].index[0])
            else:
                Dates = a[a>minDif].index
                flag = True
                c = 0
                while flag:
                    distancia = Dates[c] - peakIndex
                    if distancia > minDistance:
                        DatesEnds.append(Dates[c])
                        flag= False
                    c += 1
                    if c>maxSearch: flag = False
            Correct.append(0)
        except:
            DatesEnds.append(peakIndex)
            Correct.append(1)
    #Returns the pandas series with the values and end dates 
    Correct = np.array(Correct)
    return pd.Series(Q[DatesEnds], index=DatesEnds), Qmax[Correct == 0]


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
    Qh[np.isnan(Qh)] = Qh.mean()
    Qh[Qh<0] = Qh.mean()
    Qsep = DigitalFilters(Qh, tipo = 'Nathan', a = 0.998)
    #Pre-process of simulated series to hourly scale.
    Qsh = Qsim.resample('1h').mean()
    Qsh[np.isnan(Qsh)] = 0.0
    #Return results
    return Qh, Qsh, Qsep

def Runoff_FindEvents(Qobs, Qsim, minTime = 1, minConcav = None, minPeak = None):
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
    pos1, pos2 = __Runoff_Get_Events__(Qsim, np.percentile(Qobs, 20))
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
        Peaks = Series_Get_eventsPeaks(Q, pos1, pos2)
        Peaks = np.array(Peaks)
        #Eliminates
        p = np.where(Peaks<minPeak)[0]
        pos1 = pos1.delete(p)
        pos2 = pos2.delete(p)
    #Eliminates events based on the concavity criterion
    if minConcav is not None:
        #Obtains the concativity series 
        Concav = Q.resample('5h').mean().diff(2)
        Concav = Series_Get_eventsPeaks(Concav, pos1, pos2)
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


