import collections
import datetime
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import ha as ha
import plotly
import psutil
from _plotly_utils.colors.qualitative import Plotly
from dash.dependencies import Input, Output
import numpy as np
import halcon as ha
import os

iterationList = []
lossList = []
epochList = []
TrainSet_top1_error_valueList =[]
ValidationSet_top1_error_valueList = []

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Chadle Graph Plotter'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        html.Br(),
        html.Br(),
        html.Br(),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):

    style = {'padding': '5px', 'fontSize': '16px'}
    if os.path.isfile('C:/Users/930415/Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict'):
        TrainInfo = ha.read_dict('C:/Users/930415/Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict', (), ())
        time_elapsed = ha.get_dict_tuple(TrainInfo, 'time_elapsed')
        time_remaining = ha.get_dict_tuple(TrainInfo, 'time_remaining')
        epoch = ha.get_dict_tuple(TrainInfo, 'epoch')

    else:
        time_elapsed = [0]
        time_remaining = [0]
        epoch = [0]
    return [
        html.Span('Time Elapsed: {}s'.format(int(time_elapsed[0])), style=style),
        html.Span('Time Remaining: {}'.format(time_remaining[0]), style=style),
        html.Span('Current Epoch: {}'.format(epoch[0]), style=style)
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n, iterationList=iterationList, epochList=epochList, lossList=lossList):
    #Loss Graph
    fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=1)
    fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 80, 't': 80, 'autoexpand': False,
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    if not os.path.isfile('C:/Users/930415/Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict'):

        fig.layout = {}
        iterationList.clear()
        epochList.clear()
        lossList.clear()
    else:

        TrainInfo = ha.read_dict('C:/Users/930415\Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict', (), ())
        epoch_tuple = ha.get_dict_tuple(TrainInfo, 'epoch')
        loss_tuple = ha.get_dict_tuple(TrainInfo, 'mean_loss')
        num_iterations_per_epoch = ha.get_dict_tuple(TrainInfo, 'num_iterations_per_epoch')
        iteration = num_iterations_per_epoch[0] * epoch_tuple[0]

        if iteration not in iterationList:
            iterationList.append(iteration)
            epochList.append(epoch_tuple[0])
        if loss_tuple[0] not in lossList:
            lossList.append(loss_tuple[0])

        fig.append_trace({

            'x': iterationList,
            'y': lossList,
            'text': epochList,
            'name': 'iteration vs loss',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)

    # Top1 Error Graph
    if not os.path.isfile('C:/Users/930415/Desktop/Chadle_Halcon_Scripts/EvaluationInfo.hdict'):
        fig.data = []
        fig.layout = {}
    else:
        Evaluation_Info = ha.read_dict('C:/Users/930415\Desktop/Chadle_Halcon_Scripts/EvaluationInfo.hdict', (), ())

        TrainSet_result = ha.get_dict_tuple(Evaluation_Info, 'result_train')
        TrainSet_result_global = ha.get_dict_tuple(TrainSet_result, 'global')
        TrainSet_top1_error = ha.get_dict_tuple(TrainSet_result_global, 'top1_error')

        ValidationSet_result = ha.get_dict_tuple(Evaluation_Info, 'result')
        ValidationSet_result_global = ha.get_dict_tuple(ValidationSet_result, 'global')
        ValidationSet_top1_error = ha.get_dict_tuple(ValidationSet_result_global, 'top1_error')

        TrainSet_top1_error_value = TrainSet_top1_error[0]
        ValidationSet_top1_error_value = ValidationSet_top1_error[0]
        tempdf={
            'TrainSet_top1_error_value': TrainSet_top1_error_value,
            'ValidationSet_top1_error_value':ValidationSet_top1_error_value
        }

        if TrainSet_top1_error_value not in TrainSet_top1_error_valueList:
            TrainSet_top1_error_valueList.append(TrainSet_top1_error_value)
        if ValidationSet_top1_error_value not in ValidationSet_top1_error_valueList:
            ValidationSet_top1_error_valueList.append(ValidationSet_top1_error_value)

        fig.append_trace({
            'x': TrainSet_top1_error_valueList,
            'y': ValidationSet_top1_error_valueList,

            'name': 'Longitude vs Latitude',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 2, 1)
    return fig






if __name__ == '__main__':
    app.run_server(debug=True)
