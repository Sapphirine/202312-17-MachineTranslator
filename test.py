import pandas as pd
import numpy as np
from transformers import MarianMTModel, MarianTokenizer,file_utils
import os
from evaluate import load
import time
bertscore = None
model = None
tokenizer = None
# enter the model name here "Helsinki-NLP/opus-mt-en-zh" for baseline, and the checkpoint dir path if used the fine tuned one
model name = "Helsinki-NLP/opus-mt-en-zh"


def check_model_exists(model_name):
    cache_dir = file_utils.default_cache_path
    model_dir = os.path.join(cache_dir, model_name)
    return os.path.exists(model_dir)
def translate_eng_to_chi(text):
    global model,tokenizer

    # Tokenize the text
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True), header = None)

    # Decode the output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True, header = None)
    return translated_text
def main():
    start_time = time.time()
    global model,tokenizer,bertscore
    model_name = 'tmp/checkpoint-45000'
    if not check_model_exists(model_name):
        print("Downloading the model...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    else:
        print("Model already exists. Loading from cache...")
        tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
        model = MarianMTModel.from_pretrained(model_name, local_files_only=True)
    bertscore = load("bertscore")
    en_corpus = pd.read_csv("data/nejm.test.en",delimiter = '\\n', engine = 'python',dtype=str,header = None)
    zh_corpus = pd.read_csv("data/nejm.test.zh",delimiter = '\\n', engine = 'python',dtype=str,header = None)
    zh_predict = []
    zh_reference = []
    for i in range(1000):
        translate_sentence = en_corpus.iloc[i][0].replace("@-@","")
        zh_predict.append(translate_eng_to_chi(translate_sentence))
        reference_sentence = (str(zh_corpus.iloc[i][0])).replace(" ", "")
        zh_reference.append(reference_sentence)
        if (i % 100 == 0):
            print(i/100,'-th setence translated, time:',time.time() - start_time)
    results = bertscore.compute(predictions = zh_predict, references = zh_reference, lang = 'zh')
    result_df = pd.DataFrame({'Reference':zh_reference,'Prediction':zh_predict,'f1':results['f1']})
    result_df.to_csv('nejm.test.result.csv',index = False)
    print(np.mean(np.array(results['f1'])))
    end_time = time.time()
    print("time = ", end_time - start_time)
main()
