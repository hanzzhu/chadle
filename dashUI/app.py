import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import run

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
            value='GPU',
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
            dcc.Input(id='ImWidth', value='500', type='number', min=0, step=1, ),
            html.Label('Image Height'),
            dcc.Input(id='ImHeight', value='300', type='number', min=0, step=1, ),
            html.Label('Image Channel'),
            dcc.Input(id='ImChannel', value='3', type='number', min=0, step=1, ),
            html.Label('Batch Size'),
            dcc.Input(id='BatchSize', value='16', type='number', min=0, step=1, ),
            html.Label('Initial Learning Rate'),
            dcc.Input(id='InitialLearningRate', value='0.01', type='number', min=0, step=0.01, ),
            html.Label('Momentum'),
            dcc.Input(id='Momentum', value='0.9', type='number', min=0, step=0.01, ),
            html.Label('Number of Epochs'),
            dcc.Input(id='NumEpochs', value='500', type='number', min=0, step=1, ),
            html.Label('Change Learning Rate @ Epochs'),
            dcc.Input(id='ChangeLearningRateEpochs', value='5,10,100', type='text'),
            html.Label('Learning Rate Schedule'),
            dcc.Input(id='lr_change', value='0.01,0.1,0.5', type='text'),
            html.Label('Regularisation Constant'),
            dcc.Input(id='WeightPrior', value='0.9', type='number', min=0, step=0.01, ),
            html.Label('Class Penalty'),
            dcc.Input(id='class_penalty', value='0.01,0.1,0.5', type='text'),
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
    html.Div(id='output-state'),
    html.Div(id='Result'),
    html.Div(id='Train'),

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
              Input('ProjectName', 'value'))
def preprocess(preprocess_button, train_button, ProjectName):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'Null'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(button_id)
    if button_id == 'Null':
        raise PreventUpdate
    else:
        pre_process_param = run.pre_process(ProjectName),
        print(pre_process_param)
        DLModelHandle = pre_process_param[0][0][0]
        DLDataset = pre_process_param[0][1][0]
        TrainParam = pre_process_param[0][2][0]

        if button_id == 'train_button':
            run.training(DLDataset, DLModelHandle, TrainParam)
            return 'training is done'


@app.server.route('/downloads')
def createFile():
    return 0


if __name__ == '__main__':
    app.run_server(debug=True)
