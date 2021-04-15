import base64
import datetime

import plotly
import plotly.figure_factory as ff
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
image_filename = 'icon.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

ProjectNames = ','.join(run.ProjectList)

app.layout = html.Div([
    html.Center(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width='80', height='70')),
    html.H1('CHaDLE ',
            style={
                "font": 'verdana',
                'textAlign': 'center',
                'color': 'Black'
            }
            ),

    html.Div([
        "Project Name:",
        dcc.Input(
            id='ProjectName', value='Animals', type='text'),

        html.Br(),
        html.Label(children='Available Projects: '+ProjectNames),
        html.Br(),
        "Training Device:", dcc.RadioItems(
            id='Runtime',
            options=[{'label': i, 'value': i} for i in ['cpu', 'gpu']],
            value='cpu',
            labelStyle={'display': 'inline-block'}
        ),
        "Pretrained Model:", dcc.Dropdown(
            id='PretrainedModel',
            options=[{'label': i, 'value': i} for i in ["classifier_enhanced", "classifier_compact"]],
            value='classifier_compact'
        ),
    ],
        style={'width': '25%', 'display': 'inline-block'}),
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
            dcc.Input(id='InitialLearningRate', value='0.001', type='number', min=0, step=0.001, ),
            html.Label('Momentum'),
            dcc.Input(id='Momentum', value='0.09', type='number', min=0, step=0.001, ),
            html.Label('Number of Epochs'),
            dcc.Input(id='NumEpochs', value='2', type='number', min=0, step=1, ),
            html.Label('Change Learning Rate @ Epochs'),
            dcc.Input(id='ChangeLearningRateEpochs', value='5,100', type='text'),
            html.Label('Learning Rate Schedule'),
            dcc.Input(id='lr_change', value='0.01,0.05', type='text'),
            html.Label('Regularisation Constant'),
            dcc.Input(id='WeightPrior', value='0.001', type='number', min=0, step=0.001, ),
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

        html.Div([html.H4('Evaluation'),
                  html.Div(id='evaluation_text'),
                  dcc.Graph(id='evaluation_graph'),
                  ],
                 style={'width': '50%', 'float': 'right', }),
        dcc.Interval(
            id='interval-evaluation',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        )

    ]),

    html.Br(),
    html.Br(),
    # html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Button(id='operation_button', n_clicks=0, children='Start Training'),
    # html.Button(id='train_button', n_clicks=0, children='Train'),
    # html.Button(id='parameters_out_button', n_clicks=0, children='Output Parameters'),
    html.Button(id='evaluation_button', n_clicks=0, children='Evaluation'),
    html.Div(id='output-state'),
    html.Div(id='Operation Result'),
    html.Div(id='Train Result'),
    html.Div(id='parametersOut'),
    html.Div(id='Evaluation Result'),
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


@app.callback(Output('Operation Result', 'children'),
              Input('operation_button', 'n_clicks'),
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
def operation(operation_button, ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
              BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs, lr_change, WeightPrior,
              class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation, BrightnessVariationSpot,
              CropPercentage, CropPixel, RotationRange, IgnoreDirection, ClassIDsNoOrientationExist,
              ClassIDsNoOrientation):
    ctx_operation = dash.callback_context
    run.setup_hdev_engine()
    if not ctx_operation.triggered:
        button_id = 'Null'
    else:
        button_id = ctx_operation.triggered[0]['prop_id'].split('.')[0]

    print(button_id)
    if button_id == 'Null':
        raise PreventUpdate
    else:
        if button_id == 'operation_button':
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

            # run.training(templist[-3], templist[-2], templist[-1])
            run.training(templist[0], templist[1], templist[2])
        else:
            i = 1
            # run.training(templist[-3], templist[-2], templist[-1])

    return "Training is done!"


"""
@app.callback(Output('Train Result', 'children'),
              Input('train_button', 'n_clicks'),
              )
def training(train_button):
    ctx_training = dash.callback_context
    if not ctx_training.triggered:
        button_id = 'Null'
    else:
        button_id = ctx_training.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'train_button':
        print(templist)
        print(templist[-3], templist[-2], templist[-1])
        # run.training(templist[-3], templist[-2], templist[-1])
        return "Training Done"
"""


@app.callback(Output('evaluation_graph', 'figure'),
              Input('evaluation_button', 'n_clicks'),
              State('ProjectName', 'value'),
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

              )
def evaluation(evaluation_button, ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
               BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs, lr_change, WeightPrior,
               class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation, BrightnessVariationSpot,
               CropPercentage, CropPixel, RotationRange, IgnoreDirection,
               ):
    z = [[0, 0], [0, 0]]

    x = ['Confusion Matrix', 'Confusion Matrix']
    y = ['Confusion Matrix', 'Confusion Matrix']

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    fig = ff.create_annotated_heatmap([[0, 0], [0, 0]], x=x, y=y, annotation_text=z_text, colorscale='Blues')

    ctx_evaluation = dash.callback_context
    if not ctx_evaluation.triggered:
        button_id = 'Null'
    else:
        button_id = ctx_evaluation.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'evaluation_button':
        print('Evaluation Started')
        evaluationList = run.evaluation(ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
                                        BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs,
                                        lr_change, WeightPrior,
                                        class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation,
                                        BrightnessVariationSpot,
                                        CropPercentage, CropPixel, RotationRange, IgnoreDirection, )

        z.clear()
        x.clear()
        y.clear()
        z_text.clear()
        confusion_matrix_List = evaluationList[0]
        mean_precision = evaluationList[1][0]
        mean_recall = evaluationList[2][0]
        mean_f_score = evaluationList[3][0]

        mean_precision = format(mean_precision, '.3f')
        mean_recall = format(mean_recall, '.3f')
        mean_f_score = format(mean_f_score, '.3f')

        categories = run.getImageCategories(ProjectName)[0]
        labels = run.getImageCategories(ProjectName)[1]
        # threading.Thread(target=evaluation).start()

        length = len(categories)

        sublist = [confusion_matrix_List[i:i + length] for i in range(0, len(confusion_matrix_List), length)]
        for i in sublist:
            z.append(i)
        for i in categories:
            x.append(i)
            y.append(i)

        # change each element of z to type string for annotations
        # z_text = [[str(y) for y in x] for x in z]


        # set up figure
        z_text = [[str(y) for y in x] for x in z]
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues')
        # change each element of z to type string for annotations
        # add title
        fig.update_layout(
            title_text='Mean Precision: ' + str(mean_precision) + '\n Mean Recall: ' + str(
                mean_recall) + '\n Mean F Score: ' + str(mean_f_score),
        )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Ground Truth",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Prediction",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig['data'][0]['showscale'] = True

    return fig


@app.callback(Output('parametersOut', 'children'),
              #       Input('parameters_out_button', 'n_clicks'),
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
def parametersOut(ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
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
        html.Span('Time Elapsed: {}'.format(str(datetime.timedelta(seconds=int(time_elapsed)))), style=style),
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
        'mode': 'lines',
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
