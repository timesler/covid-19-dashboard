import os
from functools import lru_cache
import time
import requests
from multiprocessing import Pool
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

COUNTRY = os.environ.get('COUNTRY', 'Canada')

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


def _gen_logistic(t, K, alpha, nu, t0):
    K = K * 6
    t0 = t0 * 100
    return 10 ** K / ((1 + np.exp(- alpha * nu * (t - t0))) ** (1 / nu))


def run_bootstrap(args):
    t, ecdf, sigma, block_len, bounds = args
    poptb = None
    try:
        num_samples = ecdf.sum()
        new_samples = np.random.choice(len(ecdf), num_samples, p=ecdf / num_samples)
        new_ecdf = np.bincount(new_samples)
        yb = np.cumsum(new_ecdf)

        poptb, _ = curve_fit(
            _gen_logistic, t, yb,
            bounds=list(zip(*bounds)),
            sigma=sigma,
            # ftol=0.001, xtol=0.001,
        )
    
    except:
        pass

    return poptb


class GeneralizedLogistic:

    def __init__(self, t, y, popt):
        t = tuple(t.tolist())
        y = tuple(y.tolist())

        self.popt, self.popt_bs = self._fit(t, y, popt)
    
    @staticmethod
    @lru_cache(64)
    def _fit(t, y, popt=None):
        # Don't fit unless we have enough data
        if not len(y) > 0:
            return None, None
        if max(y) < 50:
            return None, None

        # Define sensible parameter bounds
        bounds = [
            [max(np.log10(max(y)), 0.1) / 6, 6 / 6],
            [0.05, 1],
            [0.01, 1],
            [10 / 100, 100 / 100],
        ]

        # If previous bounds are passed, use them to constrain parameters
        if popt is not None:
            bounds = [
                [np.log10(10 ** popt[0] * 0.15), 6 / 6],
                [popt[1] * 0.95, popt[1] * 1.05],
                [popt[2] * 0.95, popt[2] * 1.05],
                [int(popt[3]) + 5 / 100, max(int(popt[3]) + 30 / 100, 100 / 100)]
            ]

        # Don't use most recent (incomplete) day
        t = np.array(t)[:-1]
        y = np.array(y)[:-1]

        # Apply greater weight to more recent time points (arbitrarily)
        sigma = np.ones(len(t)) * (1 - t / t.max()) * 10 + 1

        popt, _ = curve_fit(
            _gen_logistic, t, y,
            bounds=list(zip(*bounds)),
            sigma=sigma
        )

        ecdf = np.insert(y[1:] - y[:-1], 0, y[0]).astype(np.int)

        # bootstraps = 100
        bootstraps = 0
        block_len = 5
        with Pool(8) as p:
            popt_bs = p.map(run_bootstrap, ((t, ecdf, sigma, block_len, bounds) for _ in range(bootstraps)))
        
        popt_bs = [p for p in popt_bs if p is not None]
        # popt_bs = np.stack(popt_bs)
        popt = tuple(popt.tolist())

        return popt, popt_bs
        
    def __call__(self, t):
        return _gen_logistic(t, *self.popt)
    
    def _step(self, y, dt):
        K, alpha, nu, t0 = np.transpose(self.popt_bs)
        K = K * 6
        t0 = t0 * 100
        return y + y * alpha * (1 - (y / 10 ** K) ** nu) * dt
    
    def project(self, y0, dt, n):
        yi = [y0 for _ in self.popt_bs]
        y = [yi]
        for i in range(n):
            yi = self._step(yi, dt)
            y.append(yi)
        
        return np.stack(y)


def generate_plot(data, start, project=1, metric='Cases', sig_fit=None):
    # Parse start and end dates for chart
    start = datetime.strptime(start, '%Y-%m-%d')
    end = datetime.now() + timedelta(days=project)

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

    # Convert time to integer days starting from 0
    t = data.index.astype(np.int64) / 1e9 / 60 / 1440
    t_min = t.min()
    t = t - t_min

    # Get current metric
    y = data[f'Total {metric}']

    # Find best fit
    gen_log = GeneralizedLogistic(t, y, popt=sig_fit)

    if gen_log.popt is not None:

        proj_n = 21
        trend_dates = pd.date_range(data.index[0], data.index[-1] + timedelta(days=proj_n), closed='left')
        fit_t = trend_dates.astype(np.int64) / 1e9 / 60 / 1440
        fit_t = fit_t - t_min
        fit_y = gen_log(fit_t)
        fit_y_new = np.insert(fit_y[1:] - fit_y[:-1], 0, fit_y[0])

        # Generate projections using differential equation
        # proj_dt = 1
        # proj_dates = pd.date_range(
        #     data.index[-1] - timedelta(days=1),
        #     data.index[-1] + timedelta(days=proj_n),
        #     closed='left'
        # )
        # proj_y0 = fit_y[trend_dates <= proj_dates[0]][-1]
        # proj_y = gen_log.project(proj_y0, proj_dt, proj_n)
        # proj_lb = np.quantile(proj_y, 0.1, axis=1)
        # proj_ub = np.quantile(proj_y, 0.9, axis=1)

        # traces_total.append(dict(
        #     x=proj_dates,
        #     y=proj_lb,
        #     mode='lines',
        #     opacity=0.7,
        #     line=dict(width=1, color='lightgrey'),
        #     name='Confidence interval (80%)',
        # ))

        # traces_total.append(dict(
        #     x=proj_dates,
        #     y=proj_ub,
        #     fill='tonexty',
        #     mode='lines',
        #     opacity=0.7,
        #     line=dict(width=1, color='lightgrey'),
        #     name='Confidence interval (80%)',
        #     showlegend=False
        # ))
        # y_max_total = max(y_max_total, proj_ub[proj_dates <= end].max())

        traces_total.append(dict(
            x=trend_dates,
            y=fit_y,
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash', color='#ff7f0e'),
            name='Trendline (generalized logistic)'
        ))
        y_max_total = max(y_max_total, fit_y[trend_dates <= end].max())

        traces_new.append(dict(
            x=trend_dates,
            y=fit_y_new,
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash', color='#ff7f0e'),
            name='Trendline (generalized logistic)'
        ))
        y_max_new = max(y_max_new, fit_y_new[trend_dates <= end].max())
    
    total_graph = dcc.Graph(
        figure={
            'data': traces_total,
            'layout': dict(
                xaxis=dict(range=[start, end]),
                yaxis=dict(range=[- y_max_total * 0.02, y_max_total * 1.02]),
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
        config=dict(displayModeBar=False),
        id=f'total-{metric.lower()}'
    )

    new_graph = dcc.Graph(
        figure={
            'data': traces_new,
            'layout': dict(
                xaxis=dict(range=[start, end], showgrid=True),
                yaxis=dict(range=[- y_max_new * 0.02, y_max_new * 1.02]),
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
        config=dict(displayModeBar=False),
        id=f'new-{metric.lower()}'
    )

    return (total_graph, new_graph), gen_log.popt


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
