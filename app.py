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

from helpers import (
    filter_province,
    generate_table,
    generate_plot,
    generate_map,
    get_data,
    placeholder_graph
)


data, provinces = get_data(datetime.now().strftime('%H'))

app = dash.Dash(__name__)

server = app.server

logo_elem = html.Img(
    id="logo",
    src="assets/xtract-logo.png",
    style=dict(position='relative', top='1.5%', left='-6vw', width=220)
)

heading_elem = html.H2('COVID-19: Canada')

dropdown_elem = dcc.Dropdown(
    id='province-select',
    options=[dict(label=p, value=p) for p in provinces],
    value='Canada',
    clearable=False,
    style=dict(marginBottom=20)
)

datepicker_elem = dcc.DatePickerSingle(
    id='time-select',
    min_date_allowed=data.index.min(),
    max_date_allowed=data.index.max() + timedelta(days=1),
    date=(datetime.now() - timedelta(weeks=8)).strftime('%Y-%m-%d'),
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

map_elem = html.Div([placeholder_graph('map-graph')], id='map')

cases_elem = html.Div(
    [
        html.Div(
            [placeholder_graph('total-cases')],
            className="six columns",
            style=dict(margin=0, padding=0)
        ),
        html.Div(
            [placeholder_graph('new-cases')],
            className="six columns",
            style=dict(margin=0, padding=0)
        )
    ],
    style=dict(marginTop=20),
    className='row'
)

deaths_elem = html.Div(
    [
        html.Div(
            [placeholder_graph('total-deaths')],
            className="six columns",
            style=dict(margin=0, padding=0)
        ),
        html.Div(
            [placeholder_graph('new-deaths')],
            className="six columns",
            style=dict(margin=0, padding=0)
        )
    ],
    style=dict(marginTop=20),
    className='row'
)

table_elem = html.Details(id='table')

vis_elem = html.Div([cases_elem, deaths_elem, table_elem], id='vis')

app.layout = html.Div(
    children=[
        # logo_elem,
        heading_elem,
        map_elem,
        dropdown_elem,
        input_elem,
        vis_elem,
        # copyright_elem,
        html.Div(id='placeholder')
    ],
    style={'width': '86%', 'margin': 'auto'}
)


@app.callback(
    [
        Output('vis', 'children'),
        Output('map', 'children'),
    ],
    [
        Input('province-select', 'value'),
        Input('time-select', 'date'),
        Input('project-select', 'value')
    ]
)
def update_vis(province, start_date, project):

    timer_start = time.time()
    data_filt = filter_province(datetime.now().strftime('%Y%m%d%H'), filt=province)
    print(f'\nGet data: {time.time() - timer_start} seconds')
    
    timer_start = time.time()
    table = generate_table(data_filt)
    print(f'Generate table: {time.time() - timer_start} seconds')
    
    timer_start = time.time()
    case_plot, sig_fit = generate_plot(data_filt, start=start_date, project=project)
    death_plot, _ = generate_plot(data_filt, start=start_date, project=project, metric='Deaths', sig_fit=sig_fit)
    print(f'Generate charts: {time.time() - timer_start} seconds')
    
    timer_start = time.time()
    map_plot = generate_map(tuple(data.Province.tolist()), tuple(data['Total Cases'].tolist()))
    print(f'Generate map: {time.time() - timer_start} seconds')


    return [case_plot, death_plot, table], map_plot


app.clientside_callback(
    """
    function(map_click, old_value) {
        var value = old_value;
        if (map_click !== undefined) {
            if (map_click['points'] !== undefined)
                value = map_click['points'][0]['location'];
        }
        return value
    }
    """,
    Output('province-select', 'value'),
    [Input('map-graph', 'clickData')],
    [State('province-select', 'value')]
)


# range_callback = """
# function(value, fig) {
#     if (fig === undefined) {
#         return fig
#     }

#     // Get new upper bound for x
#     var max_x = new Date();
#     max_x.setDate(max_x.getDate() + value);

#     // Get max y for which x is <= max_x
#     var x = fig['data'][1]['x'];
#     var y = fig['data'][1]['y'];
#     var max_y = -1000000;
#     for (var i = 0; i < x.length; i++) {
#         var x_date = new Date(x[i]);
#         if (x_date < max_x) {
#             if (y[i] > max_y) max_y = y[i];
#         }
#     }

#     max_y = Math.max(max_y, Math.max(...fig['data'][0]['y'])) * 1.05

#     max_x = max_x
#         .toLocaleString('en-us', {year: 'numeric', month: '2-digit', day: '2-digit'})
#         .replace(/(\d+)\/(\d+)\/(\d+)/, '$3-$1-$2T00:00:00');
    
#     fig['layout']['xaxis']['range'] = [
#         fig['layout']['xaxis']['range'][0], max_x
#     ];
#     fig['layout']['yaxis']['range'] = [
#         fig['layout']['yaxis']['range'][0], max_y
#     ];

#     return fig
# }
# """

# app.clientside_callback(
#     range_callback,
#     Output('total-cases', 'figure'),
#     [Input('project-select', 'value')],
#     [State('total-cases', 'figure')]
# )
# app.clientside_callback(
#     range_callback,
#     Output('new-cases', 'figure'),
#     [Input('project-select', 'value')],
#     [State('new-cases', 'figure')]
# )
# app.clientside_callback(
#     range_callback,
#     Output('total-deaths', 'figure'),
#     [Input('project-select', 'value')],
#     [State('total-deaths', 'figure')]
# )
# app.clientside_callback(
#     range_callback,
#     Output('new-deaths', 'figure'),
#     [Input('project-select', 'value')],
#     [State('new-deaths', 'figure')]
# )


if __name__ == '__main__':
    app.run_server(debug=True)
