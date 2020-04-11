from __future__ import print_function
import math
import numpy
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def display_graph_plotly(lines, title, xaxis_title, yaxis_title):
  fig = go.Figure()
  for line in lines:
    fig.add_trace(go.Scatter(x=line['x'], y=line['y'], mode='lines', name=line['name']))
  # Edit the layout
  fig.update_layout(title=title,
                   xaxis_title=xaxis_title,
                   yaxis_title=yaxis_title)
  fig.show()
  #fig.write_image("drive/EVA4/Session11/visualization/clr_graph.png")

def visualize_save_graph_matplotlib(lines, title, xaxis_title, yaxis_title, path, name):
  plt.figure(figsize=(20,10))
  for line in lines:
    plt.plot(line['x'], line['y'], label=line['name'])
  
  plt.title(title)
  plt.xlabel(xaxis_title)
  plt.ylabel(yaxis_title)
  plt.legend()
  plt.savefig(path+"/"+name+".png")

def cyclic_learning_rate(lr_min, lr_max, step_size, max_iteration):
  lr = []
  #max_iteration = 50
  delta_lr = lr_max - lr_min
  x_axis = [i for i in range(max_iteration + 1)]
  for iteration_num in range(max_iteration + 1):
    cycle = math.floor(1 + (iteration_num/(2*step_size)))
    x = abs((iteration_num/step_size) - (2*cycle) + 1)
    lr.append(lr_min + delta_lr*(1-x))
  return x_axis, lr

def generate_cyclic_learning_rate(lr_min, lr_max, step_size, max_iteration, path, name):
  x_axis, lr = cyclic_learning_rate(lr_min, lr_max, step_size, max_iteration)

  # display graph
  lines = [{'x': x_axis, 'y': [lr_max]*len(x_axis), 'name': 'max_lr'}, {'x': x_axis, 'y': lr, 'name': 'lr'}, {'x': x_axis, 'y': [lr_min]*len(x_axis), 'name': 'min_lr'}]
  title, xaxis_title, yaxis_title = "Cyclic Learning Rate", "Iteration", "Learning Rate"
  #display_graph(lines, title, xaxis_title, yaxis_title)
  visualize_save_graph_matplotlib(lines, title, xaxis_title, yaxis_title, path, name)