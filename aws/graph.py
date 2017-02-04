import pandas as pd
import pygal, sys
from decimal import *
from pygal.style import DefaultStyle

name = sys.argv[1]
output_path = "output/" + name + ".csv"
graph = pd.read_csv(output_path)

bar_chart = pygal.Bar(style=DefaultStyle, x_title='Datasets', y_title='Accuracy (%)', width=1280, height=600, show_legend=True, human_readable=True, title="kNN - NB Comparison")

knn_data, nb_data = [], []

# bar_chart.x_labels = map(str, range(1, len(graph)+1, 1))
bar_chart.x_labels = ("ecoli.csv", "fertility.csv", "glass.csv", "ilpd.csv", "ionosphere.csv", "libras.csv", "parkinsons.csv", "popfailure.csv", "sonar.csv", "wine.csv")
# bar_chart.x_labels = ("airfoil.csv", "ERA.csv", "ESL.csv", "LEV.csv", "slump.csv", "student.csv", "wdbc.csv", "yacht.csv", "glass.csv", "ilpd.csv")

for index, row in graph.iterrows():
	knn_data.append(Decimal(row["kNN"]))
	nb_data.append(Decimal(row["NB"]))

bar_chart.add("kNN", knn_data)
bar_chart.add("NB", nb_data)
chart_path = name + ".png"
bar_chart.render_to_file(chart_path)