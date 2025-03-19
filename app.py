from flask import Flask, request, render_template
from transformers import BartForConditionalGeneration, BartTokenizer, BertForSequenceClassification, BertTokenizer
import nltk

app = Flask(__name__)

# Load models
bart_model = BartForConditionalGeneration.from_pretrained("./results")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

bert_model = BertForSequenceClassification.from_pretrained("./results_extractive")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        method = request.form["method"]
        if method == "abstractive":
            inputs = bart_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=128)
            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            sentences = nltk.sent_tokenize(text)
            inputs = bert_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
            outputs = bert_model(**inputs).logits
            selected = [sentences[i] for i, logit in enumerate(outputs) if logit.argmax() == 1]
            summary = " ".join(selected[:3])  # Top 3 sentences
        return render_template("index.html", summary=summary)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)