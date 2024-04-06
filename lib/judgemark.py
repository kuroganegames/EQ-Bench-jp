import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from lib.scoring import calculate_creative_writing_score_judgemark

# model, arena elo, eq-bench, magi
# Midnight-Miqu-70B-v1.5,,75.9,40.74
other_benchmarks_str = """
claude-3-opus-20240229,1255,82.19,76.55
claude-3-sonnet-20240229,1200,80.45,61.01
gpt-4-0125-preview,1249,83.87,76.83
claude-3-haiku-20240307,1177,63.35,47.71
mistral-large-2402,1157,85.17,67.69
mistral-medium,1146,82.57,62.15
goliath-120b,,76.09,50.36
Yi-34B-Chat,1100,71.62,57.1
Qwen1.5-14B-Chat,,74.99,49.27
Mixtral-8x7B-Instruct-v0.1,1114,72.37,45.74
mistral-small,,80.36,51.9
Llama-2-13b-chat-hf,1043,49.12,28.2
Platypus2-70B-instruct,,,
openchat-3.5-1210,1071,72.52,38.81
gpt-3.5-turbo-0301,1101,70.67,46.66
Llama-2-7b-chat-hf,1027,36.32,27.5
gemma-7b-it,1029,61.72,24.85
gemma-2b-it,985,23.26,24.16
Qwen1.5-4B-Chat,974,28.75,32.66
"""

ignore_for_self_bias_calc = ["gpt-3.5-turbo-0125"]
judgemark_results_for_self_bias_str = """
test-model	gpt-4-0125-preview	gpt-3.5-turbo-0125	claude-3-haiku-20240307	claude-3-sonnet-20240229	claude-3-opus-20240229	mistral-large-2402	mistral-small	mistral-medium
gpt-4-0125-preview	71.49	63.98	83.67	80.09	74.97	76.14	76.38	76.88
claude-3-opus-20240229	69.69	65.57	82.96	77.1	76.81	77.87	73.3	74.3
claude-3-sonnet-20240229	68.5	63.56	82.69	77.21	76.23	77.71	76.4	72.77
claude-3-haiku-20240307	67.13	64.65	82.86	75.18	73.91	79.23	73.67	73.25
mistral-small	62.79	62.6	81.32	76.21	63.99	77.71	67.89	72.18
mistral-medium	68.29	63.39	81.08	74.69	69.87	77.75	73.46	75.33
mistral-large-2402	69.12	63.47	82.6	76.46	70.48	78.95	72.85	76.32
gpt-3.5-turbo-0301	53.08	59.53	77.27	63.97	50.97	70.69	61.32	63.77
01-ai/Yi-34B-Chat	66.88	66.71	83.9	77.37	67.14	74.64	79.96	72.99
openchat/openchat-3.5-1210	63.66	63.18	81.22	71.34	56.08	73.32	66.56	68.51
garage-bAInd/Platypus2-70B-instruct	55.64	59.15	78.83	71.29	51.19	69.71	64.66	65.84
mistralai/Mixtral-8x7B-Instruct-v0.1	65.89	63.87	81.17	75.34	68.21	76.99	71.99	72.46
Qwen/Qwen1.5-14B-Chat	65.5	65.6	81.97	74.33	67.13	75.83	71.48	75.9
Qwen/Qwen1.5-4B-Chat	36.49	55.33	63.34	48.78	35.33	48.32	47.27	39.93
google/gemma-2b-it	51.98	61.79	79.03	66.84	37.78	61.15	61.26	62.68
google/gemma-7b-it	55.01	60.45	79.28	70.78	50.07	71.06	62.2	61.88
meta-llama/Llama-2-7b-chat-hf	53.79	61.47	78.48	68.4	48.27	65.5	58.09	60.78
meta-llama/Llama-2-13b-chat-hf	56.52	60.64	78.1	68	55.47	70.49	65.53	66.55
sophosympatheia/Midnight-Miqu-70B-v1.5	68.55	66.01	83.63	77.25	76.41	79.55	75.6	77.03
"""

model_families = {	
	'mistralai/Mixtral-8x7B-Instruct-v0.1': 'Mistral',
	'mistral-small':'Mistral',
	'mistral-medium':'Mistral',
	'mistral-large-2402':'Mistral',
	'gpt-4-0125-preview':'OpenAI',
	'gpt-3.5-turbo-0301':'OpenAI',
	'claude-3-opus-20240229':'Anthropic',
	'claude-3-sonnet-20240229':'Anthropic',
	'claude-3-haiku-20240307':'Anthropic'
}

def parse_self_bias_judgemark_results(judgemark_str):
	data = {}
	lines = judgemark_str.strip().split('\n')
	judges = lines[0].split()[1:]
	for judge in judges:
		data[judge] = {}
	for line in lines[1:]:
		values = line.split()
		test_model = values[0]
		for i, score in enumerate(values[1:]):
			data[judges[i]][test_model] = float(score)
	return data


def create_and_save_chart(X, y, judge_mean_score, actual_judge_score, judge_name, included_models, judge_family, degree=2):
	"""
	Create and save a scatter plot with polynomial regression, highlighting in-family models and the judge model data point.
	"""
	polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
	polyreg.fit(X, y)
	X_pred = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
	y_pred = polyreg.predict(X_pred)

	plt.figure(figsize=(10, 6))
	# Ensure each model family is added to the legend only once
	plotted_families = set()

	for i, model in enumerate(included_models):
		family = model_families.get(model, None)
		color = 'green' if family == judge_family else 'blue'
		label = None
		if family == judge_family and family not in plotted_families:
			label = 'In-Family Models'
			plotted_families.add(family)
		elif family != judge_family and 'Other Models' not in plotted_families:
			label = 'Other Models'
			plotted_families.add('Other Models')
		plt.scatter(X[i], y[i], color=color, label=label, alpha=0.7)

	plt.scatter(judge_mean_score, actual_judge_score, color='orange', label='Judge Model', zorder=5)
	plt.plot(X_pred, y_pred, color='red', label='Polynomial Regression Fit')
	plt.title(f'Judge Model: {judge_name}')
	plt.xlabel('Test Models Scored by Other Judges (Avgs)')
	plt.ylabel('Test Models Scored by This Judge')
	plt.legend()
	plt.grid(True)
	# Extend the X and y arrays to include the judge's data point for calculating limits
	extended_X = np.append(X, judge_mean_score)
	extended_y = np.append(y, actual_judge_score)
	plt.xlim(extended_X.min()-1, extended_X.max()+1)
	plt.ylim(extended_y.min()-1, extended_y.max()+1)

	judge_name_safe = judge_name.replace("/", "__").replace(" ", "_")
	plt.savefig(f'judgemark_scatter_{judge_name_safe}.png')
	plt.close()



def calculate_self_bias_polynomial(data, ignore_list, degree=2):
	judges = list(data.keys())
	test_models = list(data[judges[0]].keys())
	#ignore_list += ["Qwen/Qwen1.5-4B-Chat", "google/gemma-2b-it"]

	self_bias_polynomial = {}
	family_bias = {}

	for judge in judges:
		if judge in ignore_list:
			continue

		judge_scores = {}
		mean_scores = {}
		included_models = []
		for model in test_models:
			if model in ignore_list or model == judge:
					continue
			included_models.append(model)
			judge_scores[model] = data[judge][model]
			mean_scores[model] = np.mean([data[j][model] for j in judges if j != judge and j not in ignore_list])

		X = np.array(list(mean_scores.values())).reshape(-1, 1)
		y = np.array(list(judge_scores.values()))
		judge_family = model_families.get(judge, None)

		polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
		polyreg.fit(X, y)

		if judge in test_models:
			judge_mean_score = np.mean([data[j][judge] for j in judges if j != judge and j not in ignore_list])
			actual_judge_score = data[judge][judge]
			predicted_score = polyreg.predict(np.array([[judge_mean_score]]))[0]
			self_bias_polynomial[judge] = actual_judge_score - predicted_score

			create_and_save_chart(X, y, judge_mean_score, actual_judge_score, judge, included_models, judge_family, degree)

		in_family_biases = []
		for model in included_models:
			if model in model_families and model_families[model] == judge_family:
					model_mean_score = mean_scores[model]
					model_actual_score = judge_scores[model]
					predicted_score = polyreg.predict(np.array([[model_mean_score]]))[0]
					in_family_biases.append(model_actual_score - predicted_score)

		if in_family_biases:
			family_bias[judge] = np.mean(in_family_biases)
		else:
			family_bias[judge] = None

	return self_bias_polynomial, family_bias

def parse_benchmarks(benchmark_str):
	from io import StringIO
	df = pd.read_csv(StringIO(benchmark_str), header=None, names=["model", "arena_elo", "eq_bench", "magi"])
	return df

def merge_benchmarks(judgemark_results, benchmark_str):
	df_judgemark = pd.DataFrame(list(judgemark_results['model_scores'].items()), columns=['model', 'judgemark'])
	df_benchmarks = parse_benchmarks(benchmark_str)
	
	# Concatenate model name after "/"
	df_judgemark['model'] = df_judgemark['model'].apply(lambda x: x.split('/')[-1])
	
	df_combined = pd.merge(df_judgemark, df_benchmarks, on='model', how='left')
	return df_combined

def calculate_correlations(data):
	correlations = {}
	#for benchmark in ["arena_elo", "eq_bench", "magi"]:
	for benchmark in ["arena_elo", "eq_bench"]:
		valid_data = data.dropna(subset=['judgemark', benchmark])
		if len(valid_data) > 1:  # Need at least 2 valid points to calculate correlation
			pearson_corr, _ = pearsonr(valid_data['judgemark'], valid_data[benchmark])
			kendall_corr, _ = kendalltau(valid_data['judgemark'], valid_data[benchmark])
			correlations[f'pearson_{benchmark}'] = pearson_corr
			correlations[f'kendall_{benchmark}'] = kendall_corr
		else:
			correlations[f'pearson_{benchmark}'] = None
			correlations[f'kendall_{benchmark}'] = None
	return correlations

def calculate_top_n_correlations(data):
	correlations = {}
	top_n_models = data.nlargest(8, 'judgemark')
	for benchmark in ["arena_elo", "eq_bench"]:
		valid_data = top_n_models.dropna(subset=['judgemark', benchmark])
		if len(valid_data) > 1:  # Need at least 2 valid points to calculate correlation
			pearson_corr, _ = pearsonr(valid_data['judgemark'], valid_data[benchmark])
			kendall_corr, _ = kendalltau(valid_data['judgemark'], valid_data[benchmark])
			correlations[f'pearson_top_8_{benchmark}'] = pearson_corr
			correlations[f'kendall_top_8_{benchmark}'] = kendall_corr
		else:
			correlations[f'pearson_top_8_{benchmark}'] = None
			correlations[f'kendall_top_8_{benchmark}'] = None
	return correlations


def normalize_score(score, min_score, max_score):
	if score >= max_score:
		return 1.0
	elif score <= min_score:
		return 0.0
	else:
		return (score - min_score) / (max_score - min_score)

def calculate_metrics(data):
	metrics = {
		'mean_score': data['judgemark'].mean(),
		'range': data['judgemark'].max() - data['judgemark'].min(),
		'std_dev': data['judgemark'].std(),
		'CV': data['judgemark'].std() / data['judgemark'].mean(),
	}

	# Calculate std_dev of top 5 models
	top_5_models = data.nlargest(5, 'judgemark')
	metrics['std_dev_top_5'] = top_5_models['judgemark'].std()
	
	# Calculate correlations
	correlations = calculate_correlations(data)
	metrics.update(correlations)
	
	# Calculate top 6 correlations
	top_n_correlations = calculate_top_n_correlations(data)
	metrics.update(top_n_correlations)
	
	# Normalize metrics to 0-1 range
	normalized_metrics = {}
	for metric, value in metrics.items():
		if metric == 'mean_score':
			continue  # Skip mean, as we're leaving it out of the aggregate score
		elif metric == 'range' and False:
			normalized_metrics[metric] = normalize_score(value, 0, 60)
		elif metric == 'std_dev':
			normalized_metrics[metric] = normalize_score(value, 0, 15)
		elif metric == 'std_dev_top_5' and False:
			normalized_metrics[metric] = normalize_score(value, 0, 2)
		elif metric == 'CV' and False:
			normalized_metrics[metric] = normalize_score(value, 0, 0.4)
				
	kendalls_correlations = [value for key, value in metrics.items() if key.startswith('kendall_')]
	pearsons_correlations = [value for key, value in metrics.items() if key.startswith('pearson_')]
	
	avg_kendalls = sum(kendalls_correlations) / len(kendalls_correlations) if kendalls_correlations else 0
	avg_pearsons = sum(pearsons_correlations) / len(pearsons_correlations) if pearsons_correlations else 0
	
	normalized_metrics['avg_kendalls'] = normalize_score(avg_kendalls, 0, 1)
	normalized_metrics['avg_pearsons'] = normalize_score(avg_pearsons, 0, 1)

	# Calculate aggregate score
	aggregate_score = sum(normalized_metrics.values()) / len(normalized_metrics) * 100
	metrics['Judgemark'] = aggregate_score
	
	return metrics

def compute_judgemark_results(results, run_index, test_model_outputs, verbose):
	judge_model = results[run_index]['run_metadata']['judge_model']
	print('\n#',judge_model)
	results[run_index]['judgemark_results'] = {}
	model_scores = {}
	print('\nTest model scores:')
	for model_name, _ in test_model_outputs.items():
		# This is a placeholder for wherever you calculate the creative writing score
		creative_writing_score = calculate_creative_writing_score_judgemark(run_index, model_name, results)
		if creative_writing_score is not None:
			model_scores[model_name] = creative_writing_score			
			print(round(creative_writing_score, 2), model_name)
	
	mean_score = np.mean(list(model_scores.values()))
	std_dev = np.std(list(model_scores.values()), ddof=1)  # Using sample standard deviation
	
	results[run_index]['judgemark_results'] = {
		'mean_score': mean_score,
		'std_dev': std_dev,
		'model_scores': model_scores
	}

	# Merge Judgemark results with other benchmarks into a DataFrame
	df_combined = merge_benchmarks(results[run_index]['judgemark_results'], other_benchmarks_str)
	
	# Calculate extended metrics
	extended_metrics = calculate_metrics(df_combined)

	

	judgemark_results_for_self_bias = parse_self_bias_judgemark_results(judgemark_results_for_self_bias_str)

	judgemark_results_for_self_bias[judge_model] = model_scores

	self_bias_polynomial, family_bias = calculate_self_bias_polynomial(judgemark_results_for_self_bias, ignore_for_self_bias_calc)

	print('\nStats:')
	if judge_model in self_bias_polynomial:
		print('Self bias:', round(self_bias_polynomial[judge_model], 2))
	else:
		print('Self bias:', 'N/A')
	if judge_model in family_bias and family_bias[judge_model] != None:
		print('Family bias:', round(family_bias[judge_model], 2))
	else:
		print('Self bias:', 'N/A')
	

	results[run_index]['judgemark_results']['extended_metrics'] = extended_metrics

	for k,v in results[run_index]['judgemark_results']['extended_metrics'].items():
		print(k, round(v, 2))
	print()