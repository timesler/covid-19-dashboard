import os
from functools import lru_cache
import time
import requests
from datetime import datetime, timedelta

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from helpers import (
    filter_province,
    generate_table,
    generate_plot,
    generate_map,
    get_data,
    placeholder_graph,
    PROVINCE_NAME
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[dict(name="viewport", content="width=device-width, initial-scale=1")]
)
server = app.server

COUNTRY = os.environ.get('COUNTRY', 'Canada')

# Get data
data, province_totals = get_data(datetime.now().strftime('%H'))
initial_start = (datetime.now() - timedelta(weeks=8)).strftime('%Y-%m-%d')


def init_vis(province, start_date, project):

    timer_start = time.time()
    data_filt, province_totals = filter_province(datetime.now().strftime('%H'), filt=province)
    print(f'\nGet data: {time.time() - timer_start} seconds')
    
    timer_start = time.time()
    table = generate_table(data_filt)
    print(f'Generate table: {time.time() - timer_start} seconds')
    
    timer_start = time.time()
    case_graphs, sig_fit = generate_plot(data_filt, start=start_date, project=project)
    death_graphs, _ = generate_plot(data_filt, start=start_date, project=project, metric='Deaths', sig_fit=sig_fit)
    print(f'Generate charts: {time.time() - timer_start} seconds')

    return case_graphs, death_graphs, table, province_totals.to_dict('records')


# Layout
heading_elem = html.H2(f'COVID-19: {COUNTRY}', className='my-3')

map_elem = dbc.Col(
    [
        generate_map(
            tuple(province_totals.Province.tolist()),
            tuple(province_totals['Total Cases'].tolist())
        )
    ],
    id='map',
    md=dict(offset=1, size=10),
    lg=dict(offset=2, size=8),
)

dropdown_elem = dcc.Dropdown(
    id='province-select',
    options=[dict(label=p, value=p) for p in province_totals.Province.values],
    value=COUNTRY,
    clearable=False,
)

datepicker_elem = dcc.DatePickerSingle(
    id='time-select',
    min_date_allowed=data.index.min(),
    max_date_allowed=data.index.max() + timedelta(days=1),
    date=initial_start,
    initial_visible_month=datetime.now() - timedelta(weeks=4),
)

radio_elem = dbc.RadioItems(
    id='project-select',
    options=[
        dict(label='1 day', value=1),
        dict(label='1 week', value=7),
        dict(label='2 weeks', value=14),
        dict(label='3 weeks', value=21),
        dict(label='1 month', value=30),
    ],
    value=1,
    inline=True
)

input_elem = html.Div([
    dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label(f'Select {PROVINCE_NAME[COUNTRY].lower()}:'),
                    dbc.Col([dropdown_elem], width='True'),
                ],
                className='m-2',
            )
        ]
    ),
    dbc.Form(
        [
            dbc.FormGroup(
                [
                    dbc.Label('Show from:', className='mr-2'),
                    datepicker_elem,
                ],
                className='m-2',
            ),
            dbc.FormGroup(
                [
                    dbc.Label('Extrapolate growth:', className='mr-2'),
                    radio_elem,
                ],
                className='m-2',
            ),
        ],
        inline=True,
        className='mb-3'
    )
])

case_graphs, death_graphs, table, _ = init_vis(COUNTRY, initial_start, 1)

cases_elem = dbc.Row(
    [
        dbc.Col(case_graphs[0], id='total-cases-div', sm=dict(size=12), lg=dict(size=6)),
        dbc.Col(case_graphs[1], id='new-cases-div', sm=dict(size=12), lg=dict(size=6))
    ],
    className='p-2'
)

deaths_elem = dbc.Row(
    [
        dbc.Col(death_graphs[0], id='total-deaths-div', sm=dict(size=12), lg=dict(size=6)),
        dbc.Col(death_graphs[1], id='new-deaths-div', sm=dict(size=12), lg=dict(size=6))
    ],
    className='p-2'
)

table_elem = html.Details(
    [
        html.Summary('Expand for tabular data'),
        html.Div(table, id='table', className='p-2 mx-3'),
    ],
    className='py-4',
)

app.layout = html.Div(
    children=[
        heading_elem,
        map_elem,
        input_elem,
        cases_elem,
        deaths_elem,
        table_elem,
        dcc.Store('map-store'),
    ],
    className='container'
)


# Callbacks
@app.callback(
    [
        Output('total-cases-div', 'children'),
        Output('new-cases-div', 'children'),
        Output('total-deaths-div', 'children'),
        Output('new-deaths-div', 'children'),
        Output('table', 'children'),
        Output('map-store', 'data'),
    ],
    [
        Input('province-select', 'value'),
    ],
    [
        State('time-select', 'date'),
        State('project-select', 'value'),
    ]
)
def update_vis(province, start_date, project):
    case_graphs, death_graphs, table, map_data = init_vis(province, start_date, project)

    return (
        case_graphs[0],
        case_graphs[1],
        death_graphs[0],
        death_graphs[1],
        table,
        map_data
    )



app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='update_map'),
    Output('map-graph', 'figure'),
    [Input('map-store', 'data')],
    [State('map-graph', 'figure')]
)


app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='update_dropdown'),
    Output('province-select', 'value'),
    [Input('map-graph', 'clickData')],
    [State('province-select', 'value')]
)

for id_name in ['total-cases', 'new-cases', 'total-deaths', 'new-deaths']:
    app.clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='update_range'),
        Output(id_name, 'figure'),
        [Input('project-select', 'value'), Input('time-select', 'date')],
        [State(id_name, 'figure')]
    )


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
