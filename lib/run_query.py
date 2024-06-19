from transformers import pipeline
import time
import yaml
import requests
import json
import anthropic
import google.generativeai as genai
from transformers import StoppingCriteria, StoppingCriteriaList
anthropic_client = None
gemini_model = None

class MyStoppingCriteria(StoppingCriteria):
	def __init__(self, stopping_sequences, tokenizer):
		self.stopping_sequences = stopping_sequences
		self.tokenizer = tokenizer

	def __call__(self, input_ids, scores, **kwargs):
		# Get the generated text as a string
		generated_text = self.tokenizer.decode(input_ids[0])		
		# Check if the target sequence appears in the generated text
		for seq in self.stopping_sequences:
			if seq in generated_text:
				return True  # Stop generation
		return False  # Continue generation

	def __len__(self):
		return 1

	def __iter__(self):
		yield self

# Add custom stopping criteria here if required. Only works with run_pipeline_query (which is the default).
# e.g.
#STOPPING_CRITERIA = ['assistant\n']
STOPPING_CRITERIA = []

def run_chat_template_query(prompt, completion_tokens, model, tokenizer, temp):
	chat = [
    { "role": "user", "content": prompt },
	]
	prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
	outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=completion_tokens, temperature=temp, do_sample=True)
	output = tokenizer.decode(outputs[0], skip_special_tokens=True)
	# Trim off the prompt
	trimmed_output = output[len(prompt):].strip()	
	return trimmed_output

def run_chat_query(prompt, completion_tokens, model, tokenizer, temp):
	response, history = model.chat(tokenizer, prompt, history=None, max_new_tokens=completion_tokens, do_sample=True)
	return response

def run_pipeline_query(prompt, completion_tokens, model, tokenizer, temp):
	if STOPPING_CRITERIA:
		print('Using custom stopping criteria:', STOPPING_CRITERIA)
		my_stopping_criteria = MyStoppingCriteria(STOPPING_CRITERIA, tokenizer)
		text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, do_sample=True, temperature=temp, max_new_tokens=completion_tokens, generation_kwargs = {"stopping_criteria": StoppingCriteriaList([my_stopping_criteria])})
	else:
		text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, do_sample=True, temperature=temp, max_new_tokens=completion_tokens)
	output = text_gen(prompt)
	out_str = output[0]['generated_text']
	# Trim off the prompt
	if type(out_str) == str:
		trimmed_output = out_str[len(prompt):].strip()
	else:
		trimmed_output = out_str[-1]['content'].strip()
	
	if STOPPING_CRITERIA:
		for stop_str in STOPPING_CRITERIA:
			if stop_str in trimmed_output:
				trimmed_output = trimmed_output.split(stop_str)[0]
	return trimmed_output

def run_llama3_query(prompt, completion_tokens, model, tokenizer, temp):
	text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, do_sample=True, temperature=temp, max_new_tokens=completion_tokens)
	messages = [
		{"role": "system", "content": ""},
		{"role": "user", "content": prompt},
	]
	prompt = text_gen.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True
	)

	terminators = [
		tokenizer.eos_token_id,
		tokenizer.convert_tokens_to_ids("<|eot_id|>")
	]

	outputs =text_gen(
		prompt,
		max_new_tokens=completion_tokens,
		eos_token_id=terminators,
		do_sample=True,
		temperature=temp,
	)

	trimmed_output = outputs[0]["generated_text"][len(prompt):].strip()

	return trimmed_output

def run_generate_query(prompt, completion_tokens, model, tokenizer, temp):
	inputs = tokenizer(prompt, return_tensors="pt")
	outputs = model.generate(inputs.input_ids, max_new_tokens=completion_tokens, do_sample=True, temperature=temp)
	output = tokenizer.decode(outputs[0], skip_special_tokens=True)
	# Trim off the prompt
	trimmed_output = output[len(prompt):].strip()
	return trimmed_output

# IF you are using transformers as your inferencing engine
# AND your model requires an inferencing method other than the default of transformers pipeline
# THEN specify your model & inferencing function here:
OPENSOURCE_MODELS_INFERENCE_METHODS = {
	'mistralai/Mistral-7B-Instruct-v0.1': run_generate_query,
	'Qwen/Qwen-14B-Chat': run_chat_query,
	'google/gemma-7b-it': run_chat_template_query,
	'google/gemma/2b-it': run_chat_template_query,
	'google/gemma-1.1-7b-it': run_chat_template_query,
	'meta-llama/Meta-Llama-3-70B-Instruct': run_llama3_query,
	'meta-llama/Meta-Llama-3-8B-Instruct': run_llama3_query,
}

def run_llamacpp_query(prompt, prompt_format, completion_tokens, temp):
	# Generate the prompt from the template
	formatted_prompt = generate_prompt_from_template(prompt, prompt_format)		

	# Endpoint URL for the llama.cpp server, default is localhost and port 8080
	url = "http://localhost:8080/completion"
	
	data = {
		'prompt': formatted_prompt,
		'n_predict': completion_tokens,
		'temperature': temp
	}
	
	json_data = json.dumps(data)
	
	headers = {
		'Content-Type': 'application/json',
	}
	
	response = requests.post(url, headers=headers, data=json_data)

	if response.status_code == 200:
		completion = response.json()
		content = completion['content']
		if content:
			return content.strip()
		else:
			print('Error: message is empty')
	else:
		print(f"Error: {response.status_code}")

	return None



def run_anthropic_query(prompt, history, completion_tokens, temp, model, api_key):	
	global anthropic_client
	if not anthropic_client:
		anthropic_client = anthropic.Anthropic(
			# defaults to os.environ.get("ANTHROPIC_API_KEY")
			api_key=api_key,
		)
	try:		
		
		messages = history + [{"role": "user", "content": prompt}]

		message = anthropic_client.messages.create(
			model=model,
			max_tokens=completion_tokens,
			temperature=temp,
			system="You are an expert in emotional analysis.",
			messages=messages,
			stream=False
		)

		content = message.content[0].text

		if content:
			return content.strip()
		else:
			print('Error: message is empty')
			time.sleep(5)

	except Exception as e:
		print("Request failed.")
		print(e)
		time.sleep(5)

	return None

def run_gemini_query(prompt, history, completion_tokens, temp, model, api_key):
	global gemini_model
	try:
		if not gemini_model:
			genai.configure(api_key=api_key)
			gemini_model = genai.GenerativeModel(model)

		safety_settings = [
			{
					"category": "HARM_CATEGORY_HARASSMENT",
					"threshold": "BLOCK_NONE",
			},
			{
					"category": "HARM_CATEGORY_HATE_SPEECH",
					"threshold": "BLOCK_NONE",
			},
			{
					"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
					"threshold": "BLOCK_NONE",
			},
			{
					"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
					"threshold": "BLOCK_NONE",
			},
		  
		]
		response = gemini_model.generate_content(prompt,
				generation_config=genai.types.GenerationConfig(
				candidate_count=1,
				max_output_tokens=completion_tokens,
				temperature=temp),
				safety_settings=safety_settings)		

		try:
			inference = response.text
		except Exception as e:
			print(response.parts)

		if inference:
			return inference.strip()
		else:
			print('Error: message is empty')
			time.sleep(5)
		
	except Exception as e:
		print("Request failed.")
		print(e)
		time.sleep(5)

	return None

def run_mistral_query(prompt, history, completion_tokens, temp, model, api_key):
	response = None
	try:
		url = 'https://api.mistral.ai/v1/chat/completions'
		messages = history + [{"role": "user", "content": prompt}]
		data = {
			"model": model,
        	"messages": messages,
		  	"temperature": temp,
			"max_tokens": completion_tokens,
			"stream": False,
		}

		headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " + api_key
		}		

		try:
			response = requests.post(url, headers=headers, json=data, verify=False, timeout=200)			
			response = response.json()
			#print(response)
			content = response['choices'][0]['message']['content']
			if content:
				return content.strip()
			else:
				print('Error: message is empty')
				time.sleep(5)
			
		except Exception as e:
			print(response)
			print(e)
			time.sleep(5)
			return None

	except Exception as e:
		print(response)
		print(e)
		print("Request failed.")
	return None

def run_ooba_query(prompt, history, prompt_format, completion_tokens, temp, ooba_instance, launch_ooba, ooba_request_timeout):
	if launch_ooba and (not ooba_instance or not ooba_instance.url):
		raise Exception("Error: Ooba api not initialised")
	if launch_ooba:
		ooba_url = ooba_instance.url
	else:
		ooba_url = "http://127.0.0.1:5000"

	try:
		messages = history + [{"role": "user", "content": prompt}]
		data = {
        	"mode": "instruct",        
        	"messages": messages,
		  	"instruction_template": prompt_format,
		  	"max_tokens": completion_tokens,
    		"temperature": temp,
			"user_bio": "", # workaround for ooba bug
		}

		headers = {
			"Content-Type": "application/json"
		}		

		try:
			response = requests.post(ooba_url + '/v1/chat/completions', headers=headers, json=data, verify=False, timeout=ooba_request_timeout)			
			response = response.json()
		except Exception as e:
			print(e)
			# Sometimes the ooba api stops responding. If this happens we will get a timeout exception.
			# In this case we will try to restart ooba & reload the model.
			if launch_ooba:
				print('! Request failed to Oobabooga api. Attempting to reload Ooba & model...')
				ooba_instance.restart()				
		
		content = response['choices'][0]['message']['content']
		if content:
			return content.strip()
		else:
			print('Error: message is empty')
	except KeyboardInterrupt:
		print("Operation cancelled by user.")
		raise  # Re-raising the KeyboardInterrupt exception
	except Exception as e:
		print("Request failed.")
		print(e)
	return None


def run_openai_query(prompt, history, completion_tokens, temp, model, openai_client):
	response = None
	try:
		messages = history + [{"role": "user", "content": prompt}]
		
		if model in OPENAI_COMPLETION_MODELS and openai_client.base_url == 'https://api.openai.com/v1/':
			response = openai_client.completions.create(
					model=model,
					temperature=temp,
					max_tokens=completion_tokens,
					prompt=prompt,
			)
			content = response.choices[0].text
		else: # assume it's a chat model
			response = openai_client.chat.completions.create(
					model=model,
					temperature=temp,
					max_tokens=completion_tokens,
					messages=messages,
			)
			content = response.choices[0].message.content

		if content:
			return content.strip()
		else:
			print(response)
			print('Error: message is empty')
			time.sleep(5)

	except Exception as e:
		print(response)
		print("Request failed.")
		print(e)
		time.sleep(5)

	return None


OPENAI_COMPLETION_MODELS = [
	'text-davinci-003',
	'text-davinci-002',
	'text-davinci-001',
	'text-curie-001',
	'text-babbage-001',
	'text-ada-001',
	'davinci-instruct-beta',
	'davinci',
	'curie-instruct-beta',
	'curie',
	'babbage',
	'ada',
	'gpt-3.5-turbo-instruct-0914',
	'gpt-3.5-turbo-instruct',
	'babbage-002',
	'davinci-002'
]

def parse_yaml(template_path):
	try:
		with open(template_path, 'r') as file:
			data = yaml.safe_load(file)
			# If the data is a string, replace \\n with \n
			if isinstance(data, str):
					data = data.replace('\\n', '\n')
			# If data is a dictionary, replace \\n in all string values
			elif isinstance(data, dict):
					for key, value in data.items():
						if isinstance(value, str):
							data[key] = value.replace('\\n', '\n')
			return data
	except FileNotFoundError:
		raise FileNotFoundError(f"Template file not found: {template_path}")
	
def generate_prompt_from_template(prompt, prompt_type):
	if not prompt_type:
		return prompt
	template_path = f"instruction-templates/{prompt_type}.yaml"
	template = parse_yaml(template_path)
	default_system_message = ""
	
	context = template["context"]
	if '<|system-message|>' in template['context']:
		if "system_message" in template and template["system_message"].strip():			
			context = context.replace("<|system-message|>", template["system_message"])
		else:
			context = context.replace("<|system-message|>", default_system_message)

	turn_template = template["turn_template"].replace("<|user|>", template["user"]).replace("<|bot|>", template["bot"])	
	formatted_prompt = context + turn_template
	formatted_prompt = formatted_prompt.split("<|bot-message|>")[0]
	return formatted_prompt.replace("<|user-message|>", prompt)

def run_query(model_path, prompt_format, prompt, history, completion_tokens, model, tokenizer, temp, inference_engine, ooba_instance, launch_ooba, ooba_request_timeout, openai_client, api_key = None):
	if inference_engine == 'llama.cpp':
		return run_llamacpp_query(prompt, prompt_format, completion_tokens, temp)
	elif inference_engine == 'openai':
		return run_openai_query(prompt, history, completion_tokens, temp, model_path, openai_client)
	elif inference_engine == 'anthropic':
		return run_anthropic_query(prompt, history, completion_tokens, temp, model_path, api_key)
	elif inference_engine == 'mistralai':		
		return run_mistral_query(prompt, history, completion_tokens, temp, model_path, api_key)
	elif inference_engine == 'gemini':		
		return run_gemini_query(prompt, history, completion_tokens, temp, model_path, api_key)
	elif inference_engine == 'ooba':
		return run_ooba_query(prompt, history, prompt_format, completion_tokens, temp, ooba_instance, launch_ooba, ooba_request_timeout)
	else: # transformers
		# figure out the correct inference method to use
		if model_path in OPENSOURCE_MODELS_INFERENCE_METHODS:
			inference_fn = OPENSOURCE_MODELS_INFERENCE_METHODS[model_path]
		else:
			inference_fn = run_pipeline_query

		if inference_fn in [run_chat_template_query, run_chat_query]:
			formatted_prompt = prompt
		else:
			if prompt_format:
				formatted_prompt = generate_prompt_from_template(prompt, prompt_format)		
			elif inference_fn == run_pipeline_query:
				# If no prompt format has been specified and we are using pipeline for inference,
	 			# then format the pipeline in a messages dict so pipeline knows to apply the 
	  			# model's encoded chat template.
				formatted_prompt = [
					{"role": "system", "content": "You are an expert in emotional intelligence."},
					{"role": "user", "content": prompt},
				]
			else:
				formatted_prompt = prompt
		return inference_fn(formatted_prompt, completion_tokens, model, tokenizer, temp)


