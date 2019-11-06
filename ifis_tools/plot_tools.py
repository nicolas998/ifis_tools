import plotly.graph_objects as go 
import pandas as pd 
import numpy as np 

colors = {
    'greens' : ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#006d2c','#00441b'],
    'reds' : ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704'],
    'purples' : ['#f7f4f9','#e7e1ef','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f'],
    'blues' : ['#fff7fb','#ece7f2','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#045a8d','#023858'],
    'contrast' : ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
}

def PlotSeries(Data, date_s, date_e, xname = 'Streamflow [m3 s-1]', 
  yname = 'Time [h]'):
    
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
