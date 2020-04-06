from functools import lru_cache
import time

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime, timedelta


@lru_cache(1)
def get_data(hour_str):
    data = pd.read_csv('https://health-infobase.canada.ca/src/data/covidLive/covid19.csv')
    data = data[['prname', 'date', 'numdeaths', 'numtotal', 'numtested']]
    data['date_index'] = pd.to_datetime(data.date, format='%d-%m-%Y')
    data.set_index('date_index', inplace=True)
    data.columns = ['Province', 'Date', 'Total Deaths', 'Total Cases', 'Total Tests']
    data.sort_index(inplace=True)

    provinces = data.Province.unique()

    return data, provinces


start_t = time.time()
data, provinces = get_data(datetime.now().strftime('%Y%m%d%H'))
print(time.time() - start_t)


def filter_province(data, filt='Canada'):
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
def fit_sigmoid(x, y):
    sses = []
    popts = []
    for time_shift in range(60, 301, 60):
        popt, _ = curve_fit(sig_fun, x, y, bounds=([0.1, 0.05, 10], [6, 0.9, time_shift]))
        sses.append(((y - sig_fun(x, *popt)) ** 2).sum())
        popts.append(popt)
    popt = popts[np.argmin(sses)]
    if 10 ** popt[0] > 10 * max(y):
        return None
    return lambda x: sig_fun(x, *popt)


def generate_plot(data, start, project=1, metric='Cases'):
    start = datetime.strptime(start, '%Y-%m-%d')

    x = data.index.astype(np.int64) / 10e13
    x_min = x.min().copy()
    x = x[:-1] - x_min
    y = data[f'Total {metric}'][:-1]

    trends = {
        # 'Trendline (exponential)': fit_exponential(x, y),
        'Trendline (logistic)': fit_sigmoid(tuple(x.tolist()), tuple(y.tolist()))
    }
    trends = {k: v for k, v in trends.items() if v is not None}

    trend_dates = pd.date_range(data.index.min(), data.index.max() + timedelta(days=project))
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
        # mode='lines+markers',
        opacity=0.7,
        # marker=dict(size=10),
        # line=dict(width=2),
        name=f'New {metric}'
    ))
    y_max_new = max(y_max_new, data[f'New {metric}'].max())

    for series, fn in trends.items():
        y = fn(trend_x)
        y_new = np.array([0] + [y[i] - y[i-1] for i in range(1, len(y))])

        traces_total.append(dict(
            x=trend_dates,
            y=y,
            ids=[str(id) for id in range(len(trend_dates))],
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash'),
            name=series
        ))
        y_max_total = max(y_max_total, y.max())

        traces_new.append(dict(
            x=trend_dates,
            y=y_new,
            ids=[str(id) for id in range(len(trend_dates))],
            mode='lines',
            opacity=0.7,
            line=dict(width=3, dash='dash'),
            name=series
        ))
        y_max_new = max(y_max_new, y_new.max())
    
    total_graph = dcc.Graph(
        figure={
            'data': traces_total,
            'layout': dict(
                xaxis={'title': 'Date', 'range': [start, trend_dates.max()]},
                yaxis={'title': f'{metric}', 'range': [- y_max_total * 0.05, y_max_total * 1.05]},
                hovermode='closest',
                height=500,
                title=f'Total {metric}',
                legend_title='<b>Click to hide</b>',
                legend=dict(x=0.02, y=1, bgcolor="rgba(0,0,0,0)")
            ),
        },
        animate=True,
    )

    new_graph = dcc.Graph(
        figure={
            'data': traces_new,
            'layout': dict(
                xaxis={'title': 'Date', 'range': [start, trend_dates.max()]},
                yaxis={'title': f'{metric}', 'range': [- y_max_new * 0.05, y_max_new * 1.05]},
                hovermode='closest',
                height=500,
                title=f'New {metric}',
                legend_title='<b>Click to hide</b>',
                legend=dict(x=0.02, y=1, bgcolor="rgba(0,0,0,0)")
            ),
        },
        animate=True,
    )

    return html.Div(
        [
            html.Div([total_graph], className="six columns", style={'margin': 0, 'padding': 0}),
            html.Div([new_graph], className="six columns", style={'margin': 0, 'padding': 0})
        ],
        className='row'
    )


def generate_table(data):
    table = html.Table(
        [
            html.Thead(
                html.Tr([html.Th(col) for col in data.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(data.iloc[i][col]) for col in data.columns
                ]) for i in range(len(data))
            ])
        ],
        style={'width': '80%', 'marginLeft': '10%', 'marginRight': '10%'}
    )
    return html.Details([
        html.Summary('Expand for tabular data'),
        table,
    ])


app = dash.Dash(__name__)

server = app.server

app.layout = html.Div(
    children=[
        html.Img(
            id="logo",
            src="assets/xtract-logo.png",
            style={'position': 'relative', 'top': '1.5%', 'left': '-6vw', 'width': '20%'}
        ),
        html.H2('COVID-19: Canada'),
        dcc.Dropdown(
            id='province-select',
            options=[{'label': p, 'value': p} for p in provinces],
            value='Canada',
            clearable=False
        ),
        html.Br(),
        html.Div(
            [
                'Show from:',
                dcc.DatePickerSingle(
                    id='time-select',
                    min_date_allowed=data.index.min(),
                    max_date_allowed=data.index.max() + timedelta(days=1),
                    date=(datetime.now() - timedelta(weeks=8)).strftime('%Y-%m-%d'),
                    initial_visible_month=datetime.now() - timedelta(weeks=4),
                    style={'marginLeft': 5, 'marginRight': 20}
                ),
                'Extrapolate growth:',
                dcc.RadioItems(
                    id='project-select',
                    options=[
                        {'label': '1 day', 'value': 1},
                        {'label': '1 week', 'value': 7},
                        {'label': '2 weeks', 'value': 14},
                        {'label': '1 month', 'value': 30},
                        {'label': '2 months', 'value': 60},
                    ],
                    value=1,
                    style={'display': 'inline-block', 'marginLeft': 5}
                ),
            ],
        ),
        html.Div(id='vis'),
        html.Div(
            'Copyright \u00A9 Xtract Technologies 2020',
            style={
                'position': 'relative', 'left': '-7vw', 'width': '95vw', 'marginTop': 20,
                'textAlign': 'right', 'fontSize': '12px',
            }
        ),
    ],
    style={'width': '86%', 'margin': 'auto'}
)


@app.callback(
    Output('vis', 'children'),
    [
        Input('province-select', 'value'),
        Input('time-select', 'date'),
        Input('project-select', 'value'),
    ]
)
@lru_cache(32)
def update_output_div(province, start_date, project):
    start_t = time.time()
    data, provinces = get_data(datetime.now().strftime('%Y%m%d%H'))
    print(time.time() - start_t)
    data_filt = filter_province(data, filt=province)
    table = generate_table(data_filt)
    case_plot = generate_plot(data_filt, start=start_date, project=project)
    death_plot = generate_plot(data_filt, start=start_date, project=project, metric='Deaths')
    return [case_plot, death_plot, table]


if __name__ == '__main__':
    app.run_server()
