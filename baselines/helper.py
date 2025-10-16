import json
from transformers import OPTForCausalLM, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

def load_config(config_path='../config.json'):
	with open(config_path) as cf:
		config = json.load(cf)
	
	return config

def load_data(folds_path,data_path):
  with open(folds_path, 'r', encoding='utf-8') as f:
    folds = json.load(f)
  
  with open(data_path) as f:
    all_data = json.load(f)

  all_train = []
  all_val = []
  all_test = []

  for i in range(len(folds)):
    train_inds = folds[i][0]
    val_inds = folds[i][1]
    test_inds = folds[i][2]

    train_data = [all_data[j] for j in train_inds]
    val_data = [all_data[j] for j in val_inds]
    test_data = [all_data[j] for j in test_inds]
    all_train.append(train_data)
    all_val.append(val_data)
    all_test.append(test_data)

  return all_train, all_val, all_test

_GM_EPS = 1e-12

def custom_gmean(lst):
  arr = np.array(lst, dtype=np.float64)
  arr = arr[~np.isnan(arr)]
  if arr.size == 0:
    return float("nan")

  has_pos = np.any(arr > 0)
  has_neg = np.any(arr < 0)

  if not has_pos and not has_neg:
    return 0.0

  if has_pos and has_neg:
    return float(np.mean(arr))

  working = np.abs(arr)
  working[working < _GM_EPS] = _GM_EPS
  gmean = float(np.exp(np.mean(np.log(working))))

  if has_neg and not has_pos:
    return -gmean
  return gmean

# aggregation function
def aggregate(scores, fcn):
  agg_scores = []

  for ind in range(len(scores[0])):
    l = [float(sublst[ind]) for sublst in scores]
    val = fcn(l)
    agg_scores.append(val)

  return agg_scores


def get_model_and_tokenizer(model_name):
  model_config = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "facebook/opt-1.3b": (OPTForCausalLM, AutoTokenizer)
  }
  if model_name in model_config:
    return model_config[model_name]
  return AutoModelForCausalLM, AutoTokenizer
