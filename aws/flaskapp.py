from flask import Flask, render_template, request
import csv, random, math, statistics, os, pygal, subprocess, time
from operator import itemgetter

app = Flask(__name__)

def generate_random_numbers(max):
	random_numbers = random.sample(range(0, max), max)
	return random_numbers

def normalize(rows, numcolumns, numrows):
	if rows is not []:
		for i in range(numcolumns):
			column = []
			for j in range(numrows):
				column.append(rows[j][i])
			
			column_min = min(column)
			column_max = max(column)
			
			for j in range(numrows):
				rows[j][i] = (rows[j][i] - column_min)/(column_max - column_min)

		return rows
	return None
	
def generate_sets(rows, kfcv=10):
	if rows is not None:
		sets, numcolumns, numrows = [], len(rows[0])-1, len(rows)

		for i in range(numrows):
			for j in range(numcolumns):
				rows[i][j] = float(rows[i][j])

		rows = normalize(rows, numcolumns, numrows)
		random_numbers = generate_random_numbers(numrows)
		
		for i in range(numrows):
			sets.append(rows[random_numbers[i]])
		
		numrows = len(rows)
		records = int(numrows/kfcv)+1
		
		return sets, numrows, records
	return None

def calculate_euclidean_distance(x, y, length):
	euclidean_distance = 0
	for i in range(length):
		euclidean_distance += (x[i] - y[i])**2
	return euclidean_distance**0.5

def get_distances(train, test_row, k):
	distances = []
	for i in range(len(train)):
		distances.append((train[i], calculate_euclidean_distance(test_row, train[i], len(test_row)-1)))

	distances.sort(key=itemgetter(1))
	return distances
		
def get_nearest_neighbors(train, test_row, k):
	distances = get_distances(train, test_row, k)
	nearest_neighbors = []

	for i in range(k):
		nearest_neighbors.append(distances[i][0])

	return nearest_neighbors

def get_regression_value(train, test_row, k):
	distances = get_distances(train, test_row, k)
	sum = 0
	for i in range(k):
		sum += float(distances[i][0][-1])

	return sum/k	

def get_knn_prediction(nearest_neighbors):
	predicted_classes = {}

	for i in range(len(nearest_neighbors)):
		predicted_class = nearest_neighbors[i][-1]

		if predicted_class in predicted_classes:
			predicted_classes[predicted_class] += 1
		else:
			predicted_classes[predicted_class] = 1

	sorted_predicted_classes = sorted(predicted_classes.items(), key=itemgetter(1), reverse=True)
	return sorted_predicted_classes[0][0]

def calculate_standard_deviation(value):
	return (sum([(each_value - statistics.mean(value))**2 for each_value in value]) / float(len(value)))**0.5

def calculate_class_metrics(train):
	class_dictionaries = {}
	class_metrics = {}
	
	for i in range(len(train)):
		if (train[i][-1] not in class_dictionaries):
			class_dictionaries[train[i][-1]] = []
		class_dictionaries[train[i][-1]].append(train[i][:-1])

	for key, value in class_dictionaries.items():
		class_metrics[key] = [(statistics.mean(attribute), calculate_standard_deviation(attribute)) for attribute in zip(*value)]
		
	return class_metrics

def calculate_gaussian_probability(value, mean, standard_deviation):
    return math.exp(-(float(value)-float(mean))**2 / (2 * (float(standard_deviation)**2))) / (2 * math.pi * (float(standard_deviation)**2))**0.5

def get_nb_prediction(class_metrics, test_row, k):
	probabilities = {}
	
	for key, value in class_metrics.items():
		probabilities[key] = 1.0
		for i in range(len(value)):
			if value[i][1] != 0:
				probabilities[key] *= calculate_gaussian_probability(test_row[i], value[i][0], value[i][1])
	
	return max(probabilities, key=probabilities.get)
			
def get_nb_regression_prediction(class_metrics, test_row, k):
	probabilities = {}
	
	for key, value in class_metrics.items():
		probabilities[key] = 1.0
		for i in range(len(value)):
			if value[i][1] != 0:
				probabilities[key] *= calculate_gaussian_probability(test_row[i], value[i][0], value[i][1])
	
	sorted_probabilities = sorted(probabilities.items(), key=itemgetter(1), reverse=True)
		
	prediction, count = 0.0, 0
	for tup in sorted_probabilities:
		if(tup[1] > 1.0):
			prediction += float(tup[0])
			count += 1
	
	if prediction == 0.0:
		top = sorted_probabilities[:k]
		sum = 0
		for i in range(len(top)):
			sum += float(top[i][0])
		return sum/len(top)
	else:
		return prediction/count
	
def calculate_accuracy(test, predictions):
	actual = 0
	for i in range(len(test)):
		if test[i][-1] == predictions[i]:
			actual += 1
	accuracy = (actual / float(len(test))) * 100.0
	return accuracy
	
def calculate_error(test, predictions):	
	difference = 0
	for i in range(len(test)):
		difference += (float(test[i][-1]) - float(predictions[i]))**2
	
	return (difference/len(test))**0.5
	

def perform_knn_tasks(train, test, k):
	predictions = []
	for i in range(len(test)):
		nearest_neighbors = get_nearest_neighbors(train, test[i], k)
		prediction = get_knn_prediction(nearest_neighbors)
		predictions.append(prediction)
	
	return calculate_accuracy(test, predictions)

def perform_knn_regression_tasks(train, test, k):
	regressions = []
	for i in range(len(test)):
		regression_value = get_regression_value(train, test[i], k)
		regressions.append(regression_value)
	
	return calculate_error(test, regressions)
	
def perform_nb_tasks(train, test, class_metrics):	
	predictions, probabilities = [], {}
	for i in range(len(test)):
		prediction = get_nb_prediction(class_metrics, test[i], 0)
		predictions.append(prediction)
	
	return calculate_accuracy(test, predictions)
	
def perform_nb_regression_tasks(train, test, k, class_metrics):
	predictions, probabilities = [], {}
	for i in range(len(test)):
		prediction = get_nb_regression_prediction(class_metrics, test[i], k)
		predictions.append(prediction)
	
	return calculate_error(test, predictions)
	
def perform_validation(train, validation):
	accuracies = {}
	for i in range(1, 11, 1):
		accuracies[i] = perform_knn_tasks(train, validation, i)

	sorted_accuracies = sorted(accuracies.items(), key=itemgetter(1), reverse=True)
	
	return sorted_accuracies[0][0]

@app.route("/", methods=["POST","GET"])
def home():
	if request.method == "POST":
		task = request.form["task"]
		kfcv = int(request.form["kfcv"])
		uploaded_files = request.files.getlist("file")
		render_chart = ""
		
		for file in uploaded_files:
			file_name, extension = file.filename.split(".")
			
			knn_total, nb_total, knn_time, nb_time, return_string = 0.0, 0.0, 0.0, 0.0, ""
			rows = list(csv.reader(file))
			sets, numrows, records = generate_sets(rows, kfcv)

			start_time = time.clock()
			k = perform_validation(sets[records+1 : numrows], sets[0 : records])
			knn_validation_time = time.clock()-start_time
			
			contents = "kNN,NB\n"
			
			for i in range(0, numrows, records):
				test = sets[i:i+records]
				train = sets[0:i] + sets[i+records+1:numrows]
				
				start_time = time.clock()
				class_metrics = calculate_class_metrics(train)
				nb_pre_process_time = time.clock()-start_time				
				
				if task == "classification":					
					start_time = time.clock()
					knn_accuracy = round(perform_knn_tasks(train, test, k), 2)
					knn_time += time.clock()-start_time
					
					start_time = time.clock()
					nb_accuracy = round(perform_nb_tasks(train, test, class_metrics), 2)
					nb_time += time.clock()-start_time
					
					knn_total += knn_accuracy
					nb_total += nb_accuracy
					
					contents += str(knn_accuracy) + "," + str(nb_accuracy) + "\n"
				
				elif task == "regression":
					
					start_time = time.clock()
					knn_error = round(perform_knn_regression_tasks(train, test, k), 2)
					knn_time += time.clock()-start_time
					
					start_time = time.clock()
					nb_error = round(perform_nb_regression_tasks(train, test, k, class_metrics), 2)
					nb_time += time.clock()-start_time
					
					knn_total += knn_error
					nb_total += nb_error
					
					contents += str(knn_error) + "," + str(nb_error) + "\n"
			
			if task == "classification":
				knn_total = round(knn_total/kfcv, 2)
				nb_total = round(nb_total/kfcv, 2)
			
			elif task == "regression":
				knn_total = round(knn_total/kfcv, 2)
				nb_total = round(nb_total/kfcv, 2)
				
			output_path = "/home/ubuntu/flaskapp/output/" + file_name + ".csv"
			if os.path.exists(output_path):
				os.remove(output_path)
				
			with open(output_path,"w") as f:
				f.write(contents)
			f.close()
			
			chart_path = "/home/ubuntu/flaskapp/static/images/" + file_name + ".svg"
			if os.path.exists(chart_path):
				os.remove(chart_path)
			
			cmd = "python /home/ubuntu/flaskapp/visual.py " + file_name
			result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		
			while not os.path.exists(chart_path):
				time.sleep(0.1)
			
			render_chart += '<div><center><font size = "5">' + file_name + '.csv</font><br>k-NN average: ' + str(knn_total) + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NB average: ' + str(nb_total) + '<br>k-NN time: ' + str(round(knn_time, 2)) + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NB time: ' + str(round(nb_time, 2)) + '<br><iframe id="' + file_name + '" src="/static/images/' + file_name + '.svg" width="100%" height="100%"></iframe></div><br><br>'
		
		return '''<html><head><title>ML Project</title><meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="static/stylesheets/style.css"></head><body><div class="container"><div class="header"><br><h3 class="text-muted"><center>Comparison: K-NN v/s Naive Bayes</center></h3></div><hr>''' + render_chart + '''</div></div></body></html>'''
		
	return render_template("index.html")

if __name__ == '__main__':
	app.run()