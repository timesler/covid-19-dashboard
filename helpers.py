from functools import lru_cache
import time
import requests
from datetime import datetime, timedelta

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@lru_cache(1)
def get_geojson():
    response = requests.get('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson')
    geojson = response.json()
    for i, gj in enumerate(geojson['features']):
        if 'Yukon' in gj['properties']['name']:
            gj['properties']['name'] = 'Yukon'
            geojson['features'][i] = gj
    return geojson


@lru_cache(1)
def get_data(hour_str):
    data = pd.read_csv('https://health-infobase.canada.ca/src/data/covidLive/covid19.csv')
    data = data[['prname', 'date', 'numdeaths', 'numtotal', 'numtested']]
    data['date_index'] = pd.to_datetime(data.date, format='%d-%m-%Y')
    data.date = data.date_index.dt.strftime('%Y-%m-%d')
    data.set_index('date_index', inplace=True)
    data.columns = ['Province', 'Date', 'Total Deaths', 'Total Cases', 'Total Tests']
    data.sort_index(inplace=True)

    provinces = data.Province.unique()

    return data, provinces


@lru_cache(20)
def filter_province(hour_str, filt='Canada'):
    data, _ = get_data(hour_str)
    filt = 'Canada' if filt is None else filt
    data = data.loc[data.Province == filt]
    data['New Cases'] = data['Total Cases'].diff()
    data['New Deaths'] = data['Total Deaths'].diff()
    return data


def exp_fun(x, a, b):
    return b * np.exp(a * x)


def sig_fun(x, a, b, c):
    return 10 ** a / (1 + np.exp(- b * (x - c)))


@lru_cache(64)
def fit_exponential(x, y):
    popt, _ = curve_fit(exp_fun, x, y, bounds=([0.01, 0.01], [0.4, 500]))
    return lambda x: exp_fun(x, *popt)


@lru_cache(64)
def fit_sigmoid(x, y, popt=None):
    if max(y) < 20:
        return None

    if popt is not None:
        asymp = [np.log10(max(max(y), 10 ** popt[0] * 0.001)), np.log10(10 ** popt[0] * 0.15)]
        slope = [popt[1] * 0.98, popt[1] * 1.02]
        midpt = popt[2] + 6
    else:
        asymp = [min(np.log10(max(y)), 0.1), 6]
        slope = [0.05, 0.9]
        midpt = 10

    x = list(x)[:-1]
    y = list(y)[:-1]

    sses = []
    popts = []
    for time_shift in range(max(60, int(midpt)+1), 301, 60):
        popt, _ = curve_fit(sig_fun, x, y, bounds=([asymp[0], slope[0], midpt], [asymp[1], slope[1], time_shift]))
        sses.append(((y - sig_fun(x, *popt)) ** 2).sum())
        popts.append(popt)
    popt = popts[np.argmin(sses)]
    if 10 ** popt[0] > 10 * max(y):
        return None

    return popt


def generate_plot(data, start, project=1, metric='Cases', sig_fit=None):
    start = datetime.strptime(start, '%Y-%m-%d')

    x = data.index.astype(np.int64) / 10e13
    x_min = x.min().copy()
    x = x - x_min
    y = data[f'Total {metric}']

    sig_fit = fit_sigmoid(tuple(x.tolist()), tuple(y.tolist()), sig_fit)

    trend_dates = pd.date_range(data.index.min(), datetime.now() + timedelta(days=60))
    trend_x = trend_dates.astype(np.int64) / 10e13
    trend_x = trend_x - x_min

    traces_total = []
    traces_new = []
    y_max_total = -1
    y_max_new = -1

    traces_total.append(dict(
        x=data.index,
        y=data[f'Total {metric}'],
        ids=[str(id) for id in range(len(data))],
        mode='lines+markers',
        opacity=0.7,
        marker=dict(size=10),
        line=dict(width=2),
        name=f'Total {metric}'
    ))
    y_max_total = max(y_max_total, data[f'Total {metric}'].max())

    traces_new.append(go.Bar(
        x=data.index,
        y=data[f'New {metric}'],
        ids=[str(id) for id in range(len(data))],
        opacity=0.7,
        name=f'New {metric}'
    ))
    y_max_new = max(y_max_new, data[f'New {metric}'].max())

    if sig_fit is not None:
        sig_fit = tuple(sig_fit.tolist())
        y = sig_fun(trend_x, *sig_fit)
        y_new = np.array([0] + [y[i] - y[i-1] for i in range(1, len(y))])

        traces_total.append(dict(
            x=trend_dates,
            y=y,
            ids=[str(id) for id in range(len(trend_dates))],
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash'),
            name='Trendline (logistic)'
        ))
        y_max_total = max(y_max_total, y.max())

        traces_new.append(dict(
            x=trend_dates,
            y=y_new,
            ids=[str(id) for id in range(len(trend_dates))],
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash'),
            name='Trendline (logistic)'
        ))
        y_max_new = max(y_max_new, y_new.max())
    
    total_graph = dcc.Graph(
        figure={
            'data': traces_total,
            'layout': dict(
                xaxis={'title': 'Date', 'range': [start, datetime.now() + timedelta(days=project)]},
                yaxis={'title': f'{metric}', 'range': [- y_max_total * 0.05, y_max_total * 1.05]},
                hovermode='closest',
                height=500,
                title=f'Total {metric}',
                legend_title='<b>Click to hide</b>',
                legend=dict(x=0.02, y=1, bgcolor="rgba(0,0,0,0)"),
                margin={"r": 50, "t": 30, "l": 50, "b": 70},
                dragmode=False,
            ),
        },
        animate=True,
        config=dict(displayModeBar=False),
        id=f'total-{metric.lower()}'
    )

    new_graph = dcc.Graph(
        figure={
            'data': traces_new,
            'layout': dict(
                xaxis={'title': 'Date', 'range': [start, datetime.now() + timedelta(days=project)], 'showgrid': True},
                yaxis={'title': f'{metric}', 'range': [- y_max_new * 0.05, y_max_new * 1.05]},
                hovermode='closest',
                height=500,
                title=f'New {metric}',
                legend_title='<b>Click to hide</b>',
                legend=dict(x=0.02, y=1, bgcolor="rgba(0,0,0,0)"),
                margin={"r": 50, "t": 30, "l": 50, "b": 70},
                dragmode=False,
            ),
        },
        animate=True,
        config=dict(displayModeBar=False),
        id=f'new-{metric.lower()}'
    )

    return (
        html.Div(
            [
                html.Div([total_graph], className="six columns", style=dict(margin=0, padding=0)),
                html.Div([new_graph], className="six columns", style=dict(margin=0, padding=0))
            ],
            style=dict(marginTop=20),
            className='row'
        ),
        sig_fit
    )


def generate_table(data):
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
        sort_action="native",
        style_as_list_view=True,
        style_cell={'textAlign': 'center'}
    )
    return html.Details([
        html.Summary('Expand for tabular data'),
        html.Div([table], style={'width': '80%', 'marginLeft': '10%', 'marginRight': '10%', 'marginTop': 30}),
    ])


@lru_cache(1)
def generate_map(provinces, total_cases):
    df = pd.DataFrame({'Province': provinces, 'Total Cases': total_cases})
    df = df.groupby('Province').agg({'Total Cases': max}).reset_index()
    df = df.loc[df.Province != 'Canada']

    fig = px.choropleth(
        geojson=get_geojson(), 
        locations=df['Province'],
        color=df['Total Cases'],
        featureidkey="properties.name",
        color_continuous_scale=[
            (0, "lightgrey"), 
            (0.000001, "lightgrey"),
            (0.000001, "rgb(239, 243, 255)"),
            (0.05, "rgb(189, 215, 231)"),
            (0.1, "rgb(107, 174, 214)"),
            (0.25, "rgb(49, 130, 189)"),
            (0.5, "rgb(8, 81, 156)"),
            (0.8, "rgb(5, 51, 97)"),
            (1, "rgb(5, 51, 97)"),
        ],
        projection='orthographic',
        hover_name=df['Province'],
        labels={'color':'Total cases'},
    )

    fig.data[0].hovertemplate = '<b>%{hovertext}</b><br><br>Total cases: %{z}<extra></extra>'

    fig.update_geos(
        lataxis_range=[40, 83],
        lonaxis_range=[-125, -54],
        projection_rotation=dict(lat=30),
        visible=False
    )
    
    fig.update_layout(
        title=dict(text='Total Cases By Province', y=0.9, x=0),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        dragmode=False,
        annotations=[
            dict(
                x=0,
                y=0.85,
                showarrow=False,
                text="Select province to filter charts",
                xref="paper",
                yref="paper"
            )
        ],
        coloraxis_showscale=False
    )

    return dcc.Graph(
        figure=fig,
        style=dict(padding=20, width='40vw', minWidth=400, margin='auto'),
        id='map-graph'
    )


def placeholder_graph(id):
    return dcc.Graph(id=id, style=dict(position='absolute', left='-100vw'))
