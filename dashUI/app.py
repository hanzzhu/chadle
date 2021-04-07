import ha as ha
import plotly
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
import run

iterationList = []
lossList = []
epochOfLossList = []
epochOfTop1ErrorList = []
TrainSet_top1_error_valueList = []
ValidationSet_top1_error_valueList = []

templist = []
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Chadle ', style={
        'textAlign': 'left',
        'color': 'Black'
    }),
    html.Div(["Project Name:", dcc.Dropdown(
        id='ProjectName',
        options=[{'label': i, 'value': i} for i in ['Animals', 'NTBW Image Analytics']],
        value='Animals'
    ),
              "Training Device:", dcc.RadioItems(
            id='Runtime',
            options=[{'label': i, 'value': i} for i in ['CPU', 'GPU']],
            value='CPU',
            labelStyle={'display': 'inline-block'}
        ),
              "Pretrained Model:", dcc.Dropdown(
            id='PretrainedModel',
            options=[{'label': i, 'value': i} for i in ["classifier_enhanced", "classifier_compact"]],
            value='classifier_compact'
        ),
              ],
             style={'width': '48%', 'display': 'inline-block'}),
    html.Br(),
    html.Br(),

    html.Div([
        html.Div([
            html.Label('Image Width'),
            dcc.Input(id='ImWidth', value='100 ', type='number', min=0, step=1, ),
            html.Label('Image Height'),
            dcc.Input(id='ImHeight', value='100', type='number', min=0, step=1, ),
            html.Label('Image Channel'),
            dcc.Input(id='ImChannel', value='3', type='number', min=0, step=1, ),
            html.Label('Batch Size'),
            dcc.Input(id='BatchSize', value='1', type='number', min=0, step=1, ),
            html.Label('Initial Learning Rate'),
            dcc.Input(id='InitialLearningRate', value='0.01', type='number', min=0, step=0.01, ),
            html.Label('Momentum'),
            dcc.Input(id='Momentum', value='0.9', type='number', min=0, step=0.01, ),
            html.Label('Number of Epochs'),
            dcc.Input(id='NumEpochs', value='2', type='number', min=0, step=1, ),
            html.Label('Change Learning Rate @ Epochs'),
            dcc.Input(id='ChangeLearningRateEpochs', value='5,100', type='text'),
            html.Label('Learning Rate Schedule'),
            dcc.Input(id='lr_change', value='0.01,0.05', type='text'),
            html.Label('Regularisation Constant'),
            dcc.Input(id='WeightPrior', value='0.9', type='number', min=0, step=0.01, ),
            html.Label('Class Penalty'),
            dcc.Input(id='class_penalty', value='0,0', type='text'),
        ],
            style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Augmentation Percentage'),
            dcc.Input(id='AugmentationPercentage', value='100', type='number', min=0, max=100, step=1, ),
            html.Label('Rotation'),
            dcc.Input(id='Rotation', value='90', type='number', min=-180, max=180, step=90, ),
            html.Label('Mirror (off,c,r,rc)'),
            dcc.Input(id='mirror', value='off', type='text', ),
            html.Label('Brightness Variation'),
            dcc.Input(id='BrightnessVariation', value='1', type='number', min=-100, max=100, step=1, ),
            html.Label('Brightness Variation Spot'),
            dcc.Input(id='BrightnessVariationSpot', value='1', type='number', min=-100, max=100, step=1, ),
            html.Label('Crop Percentage'),
            dcc.Input(id='CropPercentage', value='50', type='number', min=1, max=100, step=1, ),
            html.Label('Crop Pixel'),
            dcc.Input(id='CropPixel', value='500', type='number', min=1, step=1, ),
            html.Label('Rotation Range'),
            dcc.Input(id='RotationRange', value='1', type='number', min=1, step=1, ),
            html.Label('Ignore Direction'),
            dcc.Input(id='IgnoreDirection', value='false', type='text'),
            html.Label('Class IDs No Orientation Exist'),
            dcc.Input(id='ClassIDsNoOrientationExist', value='false', type='text'),
            html.Label('Class Penalty'),
            dcc.Input(id='ClassIDsNoOrientation', value='[]', type='text'),
        ],
            style={'width': '20%', 'float': 'left', 'display': 'inline-block'}),

    ]),

    html.Br(),
    html.Br(),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Button(id='preprocess_button', n_clicks=0, children='Pre-Process'),
    html.Button(id='train_button', n_clicks=0, children='Train'),
    html.Button(id='parameters_out_button', n_clicks=0, children='Output Parameters'),
    html.Div(id='output-state'),
    html.Div(id='Result'),
    html.Div(id='Train'),
    html.Div(id='parametersOut'),
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
])


@app.callback(Output('output-state', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('ProjectName', 'value'),
              State('Runtime', 'value'),
              State('PretrainedModel', 'value'),
              State('ImWidth', 'value'),
              State('ImHeight', 'value'),
              State('ImChannel', 'value'),
              State('BatchSize', 'value'),
              State('InitialLearningRate', 'value'),
              State('Momentum', 'value'),
              )
def update_output(n_clicks, ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel, BatchSize,
                  InitialLearningRate, Momentum, ):

    if n_clicks == 0:
        raise PreventUpdate
    else:
        return run.setup_hdev_engine(), u'''
                The Button has been pressed {} times,\n
                Project Name is "{}",\n
                Training Device is "{}",\n
                Pretrained model is "{}",\n
                Pretrained model is "{}",\n
                Pretrained model is "{}",\n
                Pretrained model is "{}",\n
                Pretrained model is "{}",\n
                Pretrained model is "{}",\n
                Pretrained model is "{}",\n
            '''.format(n_clicks, ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel, BatchSize,
                       InitialLearningRate, Momentum, )


@app.callback(Output('Result', 'children'),
              Input('preprocess_button', 'n_clicks'),
              Input('train_button', 'n_clicks'),
              Input('ProjectName', 'value'),
              State('Runtime', 'value'),
              State('PretrainedModel', 'value'),

              State('ImWidth', 'value'),
              State('ImHeight', 'value'),
              State('ImChannel', 'value'),
              State('BatchSize', 'value'),
              State('InitialLearningRate', 'value'),
              State('Momentum', 'value'),
              State('NumEpochs', 'value'),
              State('ChangeLearningRateEpochs', 'value'),
              State('lr_change', 'value'),
              State('WeightPrior', 'value'),
              State('class_penalty', 'value'),
              State('AugmentationPercentage', 'value'),
              State('Rotation', 'value'),
              State('mirror', 'value'),
              State('BrightnessVariation', 'value'),
              State('BrightnessVariationSpot', 'value'),
              State('CropPercentage', 'value'),
              State('CropPixel', 'value'),
              State('RotationRange', 'value'),
              State('IgnoreDirection', 'value'),
              State('ClassIDsNoOrientationExist', 'value'),
              State('ClassIDsNoOrientation', 'value'),
              )
def operation(preprocess_button, train_button, ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
              BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs, lr_change, WeightPrior,
              class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation, BrightnessVariationSpot,
              CropPercentage, CropPixel, RotationRange, IgnoreDirection, ClassIDsNoOrientationExist,
              ClassIDsNoOrientation):

    ctx = dash.callback_context
    run.setup_hdev_engine()
    if not ctx.triggered:
        button_id = 'Null'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    print(button_id)
    if button_id == 'Null':
        raise PreventUpdate
    else:
        if button_id == 'preprocess_button':

            pre_process_param = run.pre_process(ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
                                                BatchSize, InitialLearningRate, Momentum, NumEpochs,
                                                ChangeLearningRateEpochs, lr_change, WeightPrior,
                                                class_penalty, AugmentationPercentage, Rotation, mirror,
                                                BrightnessVariation, BrightnessVariationSpot,
                                                CropPercentage, CropPixel, RotationRange, IgnoreDirection,
                                                ClassIDsNoOrientationExist,
                                                ClassIDsNoOrientation)

            DLModelHandle = pre_process_param[0][0]
            DLDataset = pre_process_param[1][0]
            TrainParam = pre_process_param[2][0]
            templist.append(DLModelHandle)
            templist.append(DLDataset)
            templist.append(TrainParam)

            return 'Pre-process is done'
            # run.training(templist[0], templist[1], templist[2])

        if button_id == 'train_button':
            run.training(templist[0], templist[1], templist[2])


@app.callback(Output('parametersOut', 'children'),
              Input('parameters_out_button', 'n_clicks'),
              Input('ProjectName', 'value'),
              State('Runtime', 'value'),
              State('PretrainedModel', 'value'),
              State('ImWidth', 'value'),
              State('ImHeight', 'value'),
              State('ImChannel', 'value'),
              State('BatchSize', 'value'),
              State('InitialLearningRate', 'value'),
              State('Momentum', 'value'),
              State('NumEpochs', 'value'),
              State('ChangeLearningRateEpochs', 'value'),
              State('lr_change', 'value'),
              State('WeightPrior', 'value'),
              State('class_penalty', 'value'),
              State('AugmentationPercentage', 'value'),
              State('Rotation', 'value'),
              State('mirror', 'value'),
              State('BrightnessVariation', 'value'),
              State('BrightnessVariationSpot', 'value'),
              State('CropPercentage', 'value'),
              State('CropPixel', 'value'),
              State('RotationRange', 'value'),
              State('IgnoreDirection', 'value'),
              State('ClassIDsNoOrientationExist', 'value'),
              State('ClassIDsNoOrientation', 'value'),
              )
def parametersOut(parameters_out_button, ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
                  BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs, lr_change, WeightPrior,
                  class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation, BrightnessVariationSpot,
                  CropPercentage, CropPixel, RotationRange, IgnoreDirection, ClassIDsNoOrientationExist,
                  ClassIDsNoOrientation):
    ParameterDict = {'ProjectName': ProjectName,
                     'Runtime': Runtime, 'PretrainedModel': PretrainedModel, 'ImWidth': ImWidth, 'ImHeight': ImHeight,
                     'ImChannel': ImChannel,
                     'BatchSize': BatchSize, 'InitialLearningRate': InitialLearningRate, 'Momentum': Momentum,
                     'NumEpochs': NumEpochs,
                     'ChangeLearningRateEpochs': ChangeLearningRateEpochs, 'lr_change': lr_change,
                     'WeightPrior': WeightPrior,
                     'class_penalty': class_penalty, 'AugmentationPercentage': AugmentationPercentage,
                     'Rotation': Rotation, 'mirror': mirror,
                     'BrightnessVariation': BrightnessVariation, 'BrightnessVariationSpot': BrightnessVariationSpot,
                     'CropPercentage': CropPercentage, 'CropPixel': CropPixel, 'RotationRange': RotationRange,
                     'IgnoreDirection': IgnoreDirection,
                     'ClassIDsNoOrientationExist': ClassIDsNoOrientationExist,
                     'ClassIDsNoOrientation': ClassIDsNoOrientation}
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'Null'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'parameters_out_button':
        with open('parameters_json.txt', 'w') as outfile:
            json.dump(ParameterDict, outfile)
        return 'To json done!'


@app.server.route('/downloads')
def createFile():
    return 0



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
        if TrainSet_top1_error_value not in TrainSet_top1_error_valueList:
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
