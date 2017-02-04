import csv, random, math, statistics, os, time
from operator import itemgetter

def get_csv(path):
	if path is not None:
		return list(csv.reader(open(path, "r")))

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
	
def main():
	type = ["class", "reg"]
	count = 0
	exit = input("1-Classification, 2-Regression, 0-EXIT: ")
	
	while(exit != "0"):
		for file in os.listdir(type[int(exit)-1]):
			if file.endswith(".csv"):
				count += 1
				print("\nWorking on file #" + str(count) + " : " + file)
				
				kfcv, knn_total, nb_total, knn_time, nb_time = 10, 0.0, 0.0, 0.0, 0.0
				rows = get_csv(type[int(exit)-1]+"/"+file)
				sets, numrows, records = generate_sets(rows, kfcv)
				
				start_time = time.clock()
				k = perform_validation(sets[records+1 : numrows], sets[0 : records])
				knn_validation_time = time.clock()-start_time
				# print("Value of k " + str(k))
				
				for i in range(0, numrows, records):
					test = sets[i:i+records]
					train = sets[0:i] + sets[i+records+1:numrows]
					
					start_time = time.clock()
					class_metrics = calculate_class_metrics(train)
					nb_pre_process_time = time.clock()-start_time
					
					if(exit == "1"):
						start_time = time.clock()
						knn_accuracy = perform_knn_tasks(train, test, k)
						knn_time += time.clock()-start_time
						
						start_time = time.clock()
						nb_accuracy = perform_nb_tasks(train, test, class_metrics)
						nb_time += time.clock()-start_time
						
						nb_total += nb_accuracy
						knn_total += knn_accuracy
						
						print("KNN: " + str(knn_accuracy) + "\tNB: " + str(nb_accuracy))
					
					elif(exit == "2"):
						start_time = time.clock()
						knn_error = perform_knn_regression_tasks(train, test, k)
						knn_time += time.clock()-start_time
						
						start_time = time.clock()
						nb_error = perform_nb_regression_tasks(train, test, k, class_metrics)
						nb_time += time.clock()-start_time
						
						knn_total += knn_error
						nb_total += nb_error
						
						print("KNN: " + str(knn_error) + "\tNB: " + str(nb_error))
				
				if(exit == "1"):
					print("Avg accuracy: " + str(round(knn_total/kfcv, 2)) + " %\tAvg accuracy: " + str(round(nb_total/kfcv, 2)) + " %\n")
				
				elif(exit == "2"):
					print("Avg error: " + str(round(knn_total/kfcv, 2)) + " %\tAvg error: " + str(round(nb_total/kfcv, 2)) + " %\n")
				
				print("Time: " + str(round(knn_time, 2)) + " s\tTime: " + str(round(nb_time, 2)) + " s\n")
				
		exit = input("1-Classification, 2-Regression, 0-EXIT: ")
		
main()