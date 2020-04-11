import os
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

COUNTRY = os.environ.get('COUNTRY', 'US')

LAT_RANGES = {
    'Canada': [40, 83],
    'US': [25, 55]
}

LON_RANGES = {
    'Canada': [-125, -54],
    'US': [-120, -73]
}

PROVINCE_NAME = {
    'Canada': 'Province',
    'US': 'State'
}

def get_geojson_canada():
    response = requests.get('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson')
    geojson = response.json()
    for i, gj in enumerate(geojson['features']):
        if 'Yukon' in gj['properties']['name']:
            gj['properties']['name'] = 'Yukon'
            geojson['features'][i] = gj
    return geojson


def get_geojson_us():
    response = requests.get('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/united-states.geojson')
    geojson = response.json()
    return geojson


GEO_FNS = {
    'Canada': get_geojson_canada,
    'US': get_geojson_us,
}


@lru_cache(1)
def get_geojson():
    return GEO_FNS[COUNTRY]()


# TODO: finish global data function
# import pytz
# from tzwhere import tzwhere

# data = pd.read_csv(
#     'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
#     'csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
# )
# data = data.set_index(['Country/Region', 'Province/State'])

# def get_tz(x):
#     try:
#         return pytz.timezone(tzwhere.tzNameAt(*x.values, forceTZ=True))
#     except Exception as e:
#         print(x, x.index)
#         raise e

# coords = data[['Lat', 'Long']]
# tzwhere = tzwhere.tzwhere(forceTZ=True)
# coords['tz'] = coords.apply(get_tz, axis=1)

# data = data.drop(columns=['Lat', 'Long'])
# data = data.transpose()
# data['date_index'] = pd.to_datetime(data.index)
# data = data.set_index('date_index')


def get_data_canada():
    data = pd.read_csv('https://health-infobase.canada.ca/src/data/covidLive/covid19.csv')
    data = data[['prname', 'date', 'numdeaths', 'numtotal', 'numtested']]

    data['date_index'] = pd.to_datetime(data.date, format='%d-%m-%Y')
    data.date = data.date_index.dt.strftime('%Y-%m-%d')
    data.set_index('date_index', inplace=True)
    data.columns = ['Province', 'Date', 'Total Deaths', 'Total Cases', 'Total Tests']
    data.sort_index(inplace=True)

    provinces_totals = (
        data.groupby('Province')
            .agg({'Total Cases': max})
            .reset_index()
            .sort_values('Total Cases', ascending=False)
    )

    return data, provinces_totals


def get_data_us():
    data = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')
    data = data[['state', 'date', 'deaths', 'cases']]

    data_us = data.groupby('date').agg({'deaths': sum, 'cases': sum}).reset_index()
    data_us['state'] = 'US'

    data = pd.concat((data, data_us))

    data['date_index'] = pd.to_datetime(data.date, format='%Y-%m-%d')
    data.date = data.date_index.dt.strftime('%Y-%m-%d')
    data.set_index('date_index', inplace=True)
    data.columns = ['Total Cases', 'Date', 'Total Deaths', 'Province']
    data.sort_index(inplace=True)

    provinces_totals = (
        data.groupby('Province')
            .agg({'Total Cases': max})
            .reset_index()
            .sort_values('Total Cases', ascending=False)
    )

    return data, provinces_totals


DATA_FNS = {
    'Canada': get_data_canada,
    'US': get_data_us,
}


@lru_cache(1)
def get_data(hour_str):
    return DATA_FNS[COUNTRY]()


@lru_cache(20)
def filter_province(hour_str, filt=COUNTRY):
    data, provinces_totals = get_data(hour_str)
    filt = COUNTRY if filt is None else filt
    data = data.loc[data.Province == filt]
    data['New Cases'] = data['Total Cases'].diff()
    data['New Deaths'] = data['Total Deaths'].diff()
    return data, provinces_totals


def exp_fun(x, a, b):
    return b * np.exp(a * x)


def sig_fun(x, a, b, c):
    return 10 ** a / (1 + np.exp(- b * (x - c)))


def gom_fun(x, a, b, c):
    return 10 ** a * np.exp(-np.exp(- b * (x - c)))


@lru_cache(64)
def fit_exponential(x, y, popt=None):
    popt, _ = curve_fit(exp_fun, x, y, bounds=([0.01, 0.01], [0.4, 500]))
    return popt


@lru_cache(64)
def fit_sigmoid(x, y, fn=gom_fun, popt=None):
    if not len(y) > 0:
        return None
    if max(y) < 20:
        return None

    if popt is not None:
        asymp = [max(np.log10(max(y)), popt[0] * 0.001), np.log10(10 ** popt[0] * 0.15)]
        slope = [popt[1] * 0.9, popt[1] * 1.1]
        midpt = popt[2] + 10
    else:
        asymp = [max(np.log10(max(y)), 0.1), 8]
        slope = [0.05, 0.9]
        midpt = 10

    x = np.array(x)[:-1]
    y = np.array(y)[:-1]

    sigma = np.ones(len(x)) * (1 - x / x.max()) * 10 + 1
    
    sses = []
    popts = []
    for time_shift in range(max(60, int(midpt)+1), 301, 60):
        popt, _ = curve_fit(
            fn, x, y,
            bounds=([asymp[0], slope[0], midpt], [asymp[1], slope[1], time_shift]),
            sigma=sigma
        )
        sses.append(((y - fn(x, *popt)) ** 2).sum())
        popts.append(popt)
    popt = popts[np.argmin(sses)]

    if 10 ** popt[0] > 8 * max(y):
        return None

    return popt


def generate_plot(data, start, project=1, metric='Cases', sig_fit=None):
    start = datetime.strptime(start, '%Y-%m-%d')
    end = datetime.now() + timedelta(days=project)

    x = data.index.astype(np.int64) / 1e9 / 60 / 1440
    x_min = x.min()
    x = x - x_min
    y = data[f'Total {metric}']

    fn = gom_fun
    sig_fit = fit_sigmoid(tuple(x.tolist()), tuple(y.tolist()), fn=fn, popt=sig_fit)

    trend_dates = pd.date_range(data.index.min(), datetime.now() + timedelta(days=60))
    trend_x = trend_dates.astype(np.int64) / 1e9 / 60 / 1440
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
        y = fn(trend_x, *sig_fit)
        y_new = np.array([0] + [y[i] - y[i-1] for i in range(1, len(y))])

        traces_total.append(dict(
            x=trend_dates,
            y=y,
            ids=[str(id) for id in range(len(trend_dates))],
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash'),
            name='Trendline (Gompertz)'
        ))
        y_max_total = max(y_max_total, y[trend_dates <= end].max())

        traces_new.append(dict(
            x=trend_dates,
            y=y_new,
            ids=[str(id) for id in range(len(trend_dates))],
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash'),
            name='Trendline (Gompertz)'
        ))
        y_max_new = max(y_max_new, y_new[trend_dates <= end].max())
    
    total_graph = dcc.Graph(
        figure={
            'data': traces_total,
            'layout': dict(
                xaxis=dict(
                    # title='Date',
                    range=[start, end]
                ),
                yaxis=dict(
                    # title=f'{metric}',
                    range=[- y_max_total * 0.02, y_max_total * 1.02]
                ),
                hovermode='closest',
                height=450,
                title=f'Total {metric}',
                legend_title='<b>Click to hide</b>',
                legend=dict(x=0.02, y=1, bgcolor="rgba(0,0,0,0)"),
                margin={"r": 10, "t": 30, "l": 30, "b": 70},
                dragmode=False,
                transition={'duration': 250, 'easing': 'linear-in-out'}
            ),
        },
        # animate=True,
        config=dict(displayModeBar=False),
        id=f'total-{metric.lower()}'
    )

    new_graph = dcc.Graph(
        figure={
            'data': traces_new,
            'layout': dict(
                xaxis=dict(
                    # title='Date',
                    range=[start, end],
                    showgrid=True
                ),
                yaxis=dict(
                    # title=f'{metric}',
                    range=[- y_max_new * 0.02, y_max_new * 1.02]
                ),
                hovermode='closest',
                height=450,
                title=f'New {metric}',
                legend_title='<b>Click to hide</b>',
                legend=dict(x=0.02, y=1, bgcolor="rgba(0,0,0,0)"),
                margin={"r": 10, "t": 30, "l": 30, "b": 70},
                dragmode=False,
                transition={'duration': 250, 'easing': 'linear-in-out'}
            ),
        },
        # animate=True,
        config=dict(displayModeBar=False),
        id=f'new-{metric.lower()}'
    )

    return (total_graph, new_graph), sig_fit


def generate_table(data):
    data[PROVINCE_NAME[COUNTRY]] = data.Province
    data = data.drop(columns='Province')
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
        sort_action="native",
        style_as_list_view=True,
        style_cell={'textAlign': 'center'}
    )
    return table


@lru_cache(1)
def generate_map(provinces, total_cases):
    df = pd.DataFrame({'Province': provinces, 'Total Cases': total_cases})
    df = df.loc[df.Province != COUNTRY]

    geojson = get_geojson()

    fig = px.choropleth(
        geojson=geojson, 
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
        lataxis_range=LAT_RANGES[COUNTRY],
        lonaxis_range=LON_RANGES[COUNTRY],
        projection_rotation=dict(lat=30),
        visible=False
    )
    
    fig.update_layout(
        title=dict(text=f'Total Cases By {PROVINCE_NAME[COUNTRY]}', y=0.95, x=0),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        dragmode=False,
        annotations=[
            dict(
                x=0,
                y=0.9,
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
        id='map-graph',
        config=dict(displayModeBar=False),
        style=dict(height='100%')
    )


def placeholder_graph(id):
    return dcc.Graph(id=id, style=dict(position='absolute', left='-100vw'))
