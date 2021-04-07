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

# Initialise public lists for storing data extracted from Hdict.
iterationList = []
lossList = []
epochOfLossList = []
epochOfTop1ErrorList = []
TrainSet_top1_error_valueList = []
ValidationSet_top1_error_valueList = []

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Chadle Graph Plotter'),
        html.Div(id='indication-text'),
        dcc.Graph(id='iteration-loss-graph'),
        dcc.Graph(id='top1-error-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('indication-text', 'children'),
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
@app.callback(Output('iteration-loss-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def iteration_loss_graph(n):
    # Loss Graph configuration
    # Using plotly subplots. May consider changing to others.
    iteration_loss_graph_fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=1)
    iteration_loss_graph_fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 50, 't': 80, 'autoexpand': False,
    }
    iteration_loss_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left','title':'Loss-Iteration Graph'}
    iteration_loss_graph_fig.update_layout(legend_title_text=123)
    iteration_loss_graph_fig.update_xaxes(title_text="Iteration", row=1, col=1)
    iteration_loss_graph_fig.update_yaxes(title_text="Loss", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    if not os.path.isfile('C:/Users/930415/Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict'):

        iteration_loss_graph_fig.layout = {}
        iterationList.clear()
        epochOfLossList.clear()
        lossList.clear()
    else:
        # Extract data from the Hdict file.
        TrainInfo = ha.read_dict('C:/Users/930415\Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict', (), ())
        epoch_tuple = ha.get_dict_tuple(TrainInfo, 'epoch')
        loss_tuple = ha.get_dict_tuple(TrainInfo, 'mean_loss')
        num_iterations_per_epoch = ha.get_dict_tuple(TrainInfo, 'num_iterations_per_epoch')
        iteration = num_iterations_per_epoch[0] * epoch_tuple[0]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        if iteration != iterationList[-1]:
            iterationList.append(iteration)
            epochOfLossList.append(epoch_tuple[0])
            lossList.append(loss_tuple[0])

        # Add the values to graph and start plotting.
        iteration_loss_graph_fig.append_trace({

            'x': iterationList,
            'y': lossList,
            'text': epochOfLossList,
            'name': 'iteration vs loss',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)

    return iteration_loss_graph_fig


@app.callback(Output('top1-error-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def top1_error_graph(n):
    # Top1 Error Graph configuration.
    # Using plotly subplots. May consider changing to others.
    top1_error_graph_fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=1,)
    top1_error_graph_fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 100, 't': 80, 'autoexpand': False,
    }
    top1_error_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    top1_error_graph_fig.update_xaxes(title_text="Epoch",row=1, col=1)
    top1_error_graph_fig.update_yaxes(title_text="Top1 Error", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    if not os.path.isfile('C:/Users/930415/Desktop/Chadle_Halcon_Scripts/EvaluationInfo.hdict'):
        top1_error_graph_fig.layout = {}
        TrainSet_top1_error_valueList.clear()
        ValidationSet_top1_error_valueList.clear()
        epochOfTop1ErrorList.clear()
    else:
        # Extract data from the Hdict file.
        Evaluation_Info = ha.read_dict('C:/Users/930415/Desktop/Chadle_Halcon_Scripts/EvaluationInfo.hdict', (), ())

        Epoch = ha.get_dict_tuple(Evaluation_Info, 'epoch')

        TrainSet_result = ha.get_dict_tuple(Evaluation_Info, 'result_train')
        TrainSet_result_global = ha.get_dict_tuple(TrainSet_result, 'global')
        TrainSet_top1_error = ha.get_dict_tuple(TrainSet_result_global, 'top1_error')

        ValidationSet_result = ha.get_dict_tuple(Evaluation_Info, 'result')
        ValidationSet_result_global = ha.get_dict_tuple(ValidationSet_result, 'global')
        ValidationSet_top1_error = ha.get_dict_tuple(ValidationSet_result_global, 'top1_error')

        TrainSet_top1_error_value = TrainSet_top1_error[0]
        ValidationSet_top1_error_value = ValidationSet_top1_error[0]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        if TrainSet_top1_error_value != TrainSet_top1_error_valueList[-1]:
            epochOfTop1ErrorList.append(Epoch[0])
            TrainSet_top1_error_valueList.append(TrainSet_top1_error_value)
            ValidationSet_top1_error_valueList.append(ValidationSet_top1_error_value)

        # Add the values to graph and start plotting.
        # Two plots on the same graph.
        top1_error_graph_fig.append_trace({
            'x': epochOfTop1ErrorList,
            'y': TrainSet_top1_error_valueList,


            'name': 'Train Set Top1_error',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)

        top1_error_graph_fig.append_trace({
            'x': epochOfTop1ErrorList,
            'y': ValidationSet_top1_error_valueList,


            'name': 'Validation Set Top1_error',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)
    return top1_error_graph_fig


if __name__ == '__main__':
    app.run_server(debug=True)
