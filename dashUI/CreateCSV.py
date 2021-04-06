import json
import time

x =[]
y = []
i = 0

data = {'x':x,
            'y':y}
while True:
    with open('PlotData.txt', 'w') as outfile:
        json.dump(data, outfile)

    time.sleep(0.5)
    x.append(i)
    y.append(time.thread_time())
    i += 1
