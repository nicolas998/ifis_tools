import plotly.graph_objects as go 
import pandas as pd 
import numpy as np 
import folium
from folium import plugins
import geopandas

colors = {
    'greens' : ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#006d2c','#00441b'],
    'reds' : ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704'],
    'purples' : ['#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f'],
    'blues' : ['#fff7fb','#ece7f2','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#045a8d','#023858'],
    'contrast' : ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
}

dicQ = {'observed':{'q':Qo,'type':pl.scatter,'args':{'s':130,'label':'obs'}},
       'simulated':{'q':Qs,'type':pl.plot,
                    'args':{'lw':3,'color':'#313695','label':'sim'}},
        'sst1':{'q':Qsce,'type':pl.plot,
                'args':{'lw':3,'color':'#fee090','label':'sst1'}},
       'sst2':{'q':Qsce2,'type':pl.plot,
               'args':{'lw':3,'color':'#f46d43','label':'sst2'}},
       'sst3':{'q':Qsce3,'type':pl.plot,
               'args':{'lw':3,'color':'#a50026','label':'sst3'}}}

def plot2(link,dicQ, step = 500, start =0 , end =-1, intersect = [0,1],path = None, name = None):
    fig = pl.figure(figsize=(15,5))
    ax = fig.add_subplot(111)    
    keys = list(dicQ.keys())
    indx1 = dicQ[keys[intersect[0]]]['q'][link].index
    indx2 = dicQ[keys[intersect[1]]]['q'][link].index
    idx = indx1.intersection(indx2)
    for k in dicQ.keys():
        d = dicQ[k]
        indx = d['q'][link].loc[idx].index
        val = d['q'][link].loc[idx].values
        d['type'](indx,val,**d['args'])
    ax.set_xlim(indx[start], indx[end])
    ax.set_xticks(indx[start:end:step])
    ax.legend(loc = 0, fontsize = 'xx-large')
    if name:
        ax.text(0.005,1.03,name, transform=ax.transAxes, size=20,bbox=dict(facecolor='gray', alpha=0.4))
    ax.set_ylabel('Streamflow [$m^3 \cdot s^{-1}$]', size = 20)    
    ax.tick_params(labelsize = 18)    
    ax.grid()
    if path is not None:
        pl.savefig(path, bbox_inches = 'tight')

def PlotSeries(Data, date_s, date_e, xname = 'Streamflow [m3 s-1]', 
  yname = 'Time [h]', width = 700, height = 400):
    
    fig = go.Figure()

    for k in Data.keys():    
        if Data[k]['mode'] == 'markers':        
            fig.add_trace(go.Scatter(
                x = Data[k]['s'].index, 
                y = Data[k]['s'].values,
                name = k,
                mode = Data[k]['mode'],
                marker = Data[k]['style']
            ))
        elif Data[k]['mode'] == 'lines':
            fig.add_trace(go.Scatter(
                x = Data[k]['s'].index, 
                y = Data[k]['s'].values,
                name = k,
                mode = Data[k]['mode'],
                line = Data[k]['style']
            ))
       
    fig.update_layout(
        width = width,
        height = height,
        margin = go.layout.Margin(
            l = 50,
            r = 20,
            b = 50,
            t = 20
        ),
        xaxis_range=[date_s, date_e],
        xaxis_title = xname,
        yaxis_title = yname,
        font = dict(
            size = 17
        )
    )
    fig.update_xaxes(tickfont=dict(size = 16))
    fig.update_yaxes(tickfont=dict(size = 16))

    fig.show()

def Data4PlotSeries(Dic, file_path, name, year, color, mode = 'lines', width = 4):
    '''Prepare the dictionary for the function PlotSeries'''
    #Updates the dictionary
    q = pd.read_msgpack(file_path)
    if mode == 'lines':
        Dic.update({name: {'s': q[str(year)], 'mode': mode, 
            'style' : {'color': color,  'width': width}}})
    elif mode == 'markers':
        Dic.update({name: {'s': q[str(year)], 'mode': mode, 
        'style' : {'color': color,  'size': width}}})
    return Dic

def values2colors(values, colors, bins):
    col_intervalo = np.array(['#b2182b' for i in range(values.size)])
    for i,j,co in zip(bins[:-1], bins[1:],colors):
        col_intervalo[(values>i) & (values<=j)] = co
    return col_intervalo

def leaflet_map(location = [42, -93], height = '70%', width = '80%', zoom_start = 9,
    geoDataFrame = None, geo_name = None, geo_tool=None, geo_popup=None, geo_color=None):
    #Defines the map 
    m = folium.Map(
            location=location,
            height = height, 
            width = width,
            zoom_start = zoom_start
        )
    #Add maps tiles 
    folium.TileLayer(tiles='Stamen Terrain',name="Stamen Terrain").add_to(m)
    folium.TileLayer(tiles='OpenStreetMap',name="Open Street").add_to(m)
    
    if geoDataFrame is not None:
        #Add features
        fg = folium.FeatureGroup()
        m.add_child(fg)
        g1 = plugins.FeatureGroupSubGroup(fg, geo_name)
        m.add_child(g1)

        #plot the stations
        for i in range(geoDataFrame.shape[0]):
            x = geoDataFrame['geometry'][i].x
            y = geoDataFrame['geometry'][i].y
            td = '%.2f' % geoDataFrame[geo_tool][i]
            g1.add_child(folium.CircleMarker(location=[y,x], radius=10,
                    popup ='<b>Id: </b>%s'%(str(geoDataFrame[geo_popup][i])), 
                    tooltip ='<b>val: </b>%s'%(td), 
                    line_color='#3186cc',
                    line_width = 0.5,
                    fill_color= geoDataFrame[geo_color][i],
                    fill_opacity=0.7, 
                    fill=True))
    folium.LayerControl().add_to(m)
    return m

class leaflet_map_class:

    def __init__(self, location = [42, -93], height = '70%', width = '80%', zoom_start = 9, **kwargs):
        self.m = folium.Map(
            location=location,
            height = height, 
            width = width,
            zoom_start = zoom_start
        )
        #Add maps tiles 
        folium.TileLayer(tiles='Stamen Terrain',name="Stamen Terrain").add_to(self.m)
        folium.TileLayer(tiles='OpenStreetMap',name="Open Street").add_to(self.m)
        folium.LayerControl().add_to(self.m)
        #Add rthe groups option
        

    def add_points(self, geoDataFrame, name_col, color_col = '#2c7fb8', popup_col = None):
        '''Add a set of points to the leaflet map'''
        #Creates a group for that layer
        fg = folium.FeatureGroup()
        self.m.add_child(fg)
        g1 = plugins.FeatureGroupSubGroup(fg, name_col)
        self.m.add_child(g1)

        #plot the stations
        for i in range(geoDataFrame.shape[0]):
            x = geoDataFrame['geometry'][i].x
            y = geoDataFrame['geometry'][i].y
            td = '%.2f' % geoDataFrame[name_col][i]
            g1.add_child(folium.CircleMarker(location=[y,x], radius=10,
                    popup ='<b>Id: </b>%s'%(str(geoDataFrame[popup_col][i])), 
                    tooltip ='<b>val: </b>%s'%(td), 
                    line_color='#3186cc',
                    line_width = 0.5,
                    fill_color=geoDataFrame[color_col][i],
                    fill_opacity=0.7, 
                    fill=True))
        #folium.LayerControl().add_to(self.m)

    def display(self):
        '''Shows the map'''
        self.m