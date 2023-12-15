#!/usr/bin/python3

# Run: flask run -h <IP> -p 7979

from flask import Flask,jsonify,request,render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

model_name = "SFZheng7/MarianMT-Finetuned"
# model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


@app.route("/")
def hello_world():
    return render_template("index.html")

@app.post("/translate")
def translate():
    sentence = request.form['sentence']
    translated = model.generate(**tokenizer(sentence, return_tensors="pt", padding=True), header=None)

    # Decode the output
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True, header=None)
    return jsonify({'sentence': translated_text})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)
