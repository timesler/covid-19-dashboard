from functools import lru_cache
import time
import requests
from datetime import datetime, timedelta

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, ClientsideFunction
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
    placeholder_graph
)

app = dash.Dash(__name__)
server = app.server

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
    case_plot, sig_fit = generate_plot(data_filt, start=start_date, project=project)
    death_plot, _ = generate_plot(data_filt, start=start_date, project=project, metric='Deaths', sig_fit=sig_fit)
    print(f'Generate charts: {time.time() - timer_start} seconds')

    return [case_plot, death_plot, table], province_totals.to_dict('records')


# Layout
logo_elem = html.Img(
    id="logo",
    src="assets/xtract-logo.png",
    style=dict(position='relative', top='1.5%', left='-6vw', width=220)
)

heading_elem = html.H2('COVID-19: Canada')

dropdown_elem = dcc.Dropdown(
    id='province-select',
    options=[dict(label=p, value=p) for p in province_totals.Province.values],
    value='Canada',
    clearable=False,
    style=dict(marginBottom=20)
)

datepicker_elem = dcc.DatePickerSingle(
    id='time-select',
    min_date_allowed=data.index.min(),
    max_date_allowed=data.index.max() + timedelta(days=1),
    date=initial_start,
    initial_visible_month=datetime.now() - timedelta(weeks=4),
    style=dict(marginLeft=5, marginRight=20)
)

radio_elem = dcc.RadioItems(
    id='project-select',
    options=[
        dict(label='1 day', value=1),
        dict(label='1 week', value=7),
        dict(label='2 weeks', value=14),
        dict(label='1 month', value=30),
        dict(label='2 months', value=60),
    ],
    value=1,
    style=dict(display='inline-block', marginLeft=5)
)

input_elem = html.Div([
    'Show from:',
    datepicker_elem,
    'Extrapolate growth:',
    radio_elem,
])

copyright_elem = html.Div(
    'Copyright \u00A9 Xtract Technologies 2020',
    style=dict(
        position='relative',
        left='-7vw',
        width='95vw',
        marginTop=20,
        textAlign='right',
        fontSize='12px',
    )
)

vis_plots, _ = init_vis('Canada', initial_start, 1)

map_elem = html.Div(
    [
        generate_map(
            tuple(province_totals.Province.tolist()),
            tuple(province_totals['Total Cases'].tolist())
        )
    ],
    id='map'
)

vis_elem = html.Div(vis_plots, id='vis', style=dict(marginBottom=60))

app.layout = html.Div(
    children=[
        # logo_elem,
        heading_elem,
        map_elem,
        dropdown_elem,
        input_elem,
        vis_elem,
        # copyright_elem,
        html.Div(id='placeholder'),
        dcc.Store('map-store'),
    ],
    style={'width': '86%', 'margin': 'auto'}
)


# Callbacks
@app.callback(
    [
        Output('vis', 'children'),
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
    return init_vis(province, start_date, project)


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
    app.run_server(debug=True)
