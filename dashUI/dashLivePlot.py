import collections

import dash
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
import psutil

from dash.dependencies import Input, Output

# Example data (a circle).
resolution = 30
t = np.linspace(0, np.pi * 2, resolution)
#x, y = np.cos(t), np.sin(t)

cpu = collections.deque(np.zeros(10))
ram = collections.deque(np.zeros(10))



x = cpu
y = ram
# Example app.
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
figure = dict(data=[{'x': [], 'y': []}], layout=dict(xaxis=dict(range=[0, 10]), yaxis=dict(range=[0, 100])))
app = dash.Dash(__name__, update_title=None,external_stylesheets=external_stylesheets)  # remove "Updating..." from title
app.layout = html.Div([dcc.Graph(id='graph', figure=figure), dcc.Interval(id="interval")])


@app.callback(Output('graph', 'extendData'), [Input('interval', 'n_intervals')])
def update_data(n_intervals):
    index = n_intervals % resolution
    cpu.popleft()
    cpu.append(psutil.cpu_percent())

    ram.popleft()
    ram.append(psutil.virtual_memory().percent)

    return dict(x=[[cpu[index]]], y=[[ram[index]]]), [0], 10


if __name__ == '__main__':
    app.run_server(debug=True)