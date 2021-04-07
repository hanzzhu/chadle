import collections
import datetime
import json
import run
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
    # Indication Text configuration
    # Extract data from Hdict and show as texts.
    style = {'padding': '5px', 'fontSize': '16px'}

    getmetrics = run.get_TrainInfo()
    if getmetrics:
        time_elapsed = getmetrics[0]
        time_remaining = getmetrics[1]
        epoch_metrics = getmetrics[2]
    else:
        time_elapsed = 0
        time_remaining = 0
        epoch_metrics = 0

    return [
        html.Span('Time Elapsed: {}s'.format(int(time_elapsed)), style=style),
        html.Span('Time Remaining: {}'.format(time_remaining), style=style),
        html.Span('Current Epoch: {}'.format(epoch_metrics), style=style)
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
    iteration_loss_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left', 'title': 'Loss-Iteration Graph'}
    iteration_loss_graph_fig.update_layout(legend_title_text=123)
    iteration_loss_graph_fig.update_xaxes(title_text="Iteration", row=1, col=1)
    iteration_loss_graph_fig.update_yaxes(title_text="Loss", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    getTrainInfo = run.get_TrainInfo()
    if not getTrainInfo:

        iterationList.clear()
        epochOfLossList.clear()
        lossList.clear()
    else:
        epoch_TrainInfo = getTrainInfo[2]
        loss = getTrainInfo[3]
        iteration = getTrainInfo[4]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        # Using <in> with list, average time complexity: O(n)
        if iteration not in iterationList:
            epochOfLossList.append(epoch_TrainInfo)
            lossList.append(loss)
            iterationList.append(iteration)

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
    top1_error_graph_fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=1, )
    top1_error_graph_fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 100, 't': 80, 'autoexpand': False,
    }
    top1_error_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    top1_error_graph_fig.update_xaxes(title_text="Epoch", row=1, col=1)
    top1_error_graph_fig.update_yaxes(title_text="Top1 Error", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    getEvaluationInfo = run.get_EvaluationInfo()
    if not getEvaluationInfo:
        TrainSet_top1_error_valueList.clear()
        ValidationSet_top1_error_valueList.clear()
        epochOfTop1ErrorList.clear()
    else:
        epoch_EvaluationInfo = getEvaluationInfo[0]
        TrainSet_top1_error_value = getEvaluationInfo[1]
        ValidationSet_top1_error_value = getEvaluationInfo[2]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        # Using <in> with list, average time complexity: O(n)
        if epoch_EvaluationInfo not in epochOfTop1ErrorList:
            epochOfTop1ErrorList.append(epoch_EvaluationInfo)
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
