from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load CodeBERT for vulnerability detection
codebert_model_name = "microsoft/codebert-base"
codebert_tokenizer = RobertaTokenizer.from_pretrained(codebert_model_name)
codebert_model = RobertaForSequenceClassification.from_pretrained(codebert_model_name, num_labels=5)  # 5 labels for multi-class classification

# Load CodeT5 for generating code fixes
codet5_model_name = "Salesforce/codet5-base"
codet5_tokenizer = AutoTokenizer.from_pretrained(codet5_model_name)
codet5_model = AutoModelForSeq2SeqLM.from_pretrained(codet5_model_name)

# Vulnerability classes (customize based on your needs)
vulnerability_classes = ["No Vulnerability", "SQL Injection", "Buffer Overflow", "Command Injection", "Cross-site Scripting (XSS)"]

# Vulnerability explanations
vulnerability_explanations = {
    "SQL Injection": "SQL Injection is a code injection technique that allows attackers to interfere with the queries an application makes to its database. It's dangerous because attackers can access, modify, or delete data.",
    "Buffer Overflow": "A buffer overflow occurs when a program writes more data to a buffer than it can hold. This can lead to crashes or execution of malicious code.",
    "Command Injection": "Command injection is an attack where the goal is to execute arbitrary commands on the host operating system via a vulnerable application.",
    "Cross-site Scripting (XSS)": "XSS allows attackers to inject client-side scripts into web pages viewed by other users. This can lead to hijacking user sessions or malicious redirection.",
}

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page with the chat interface

@app.route('/check_vulnerability', methods=['POST'])
def check_vulnerability():
    data = request.get_json()
    code_snippet = data.get('code_snippet')
    
    # Step 1: Detect Vulnerability using CodeBERT
    inputs = codebert_tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = codebert_model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    
    predicted_class = vulnerability_classes[predictions.item()]
    
    response = {"detected_vulnerability": predicted_class}
    
    # Provide explanation of the vulnerability
    if predicted_class != "No Vulnerability":
        explanation = vulnerability_explanations.get(predicted_class, "No explanation available.")
        response["explanation"] = explanation
        
        # Step 2: Generate a fix using CodeT5
        inputs_codet5 = codet5_tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs_codet5 = codet5_model.generate(inputs_codet5.input_ids, max_length=150, num_beams=4, early_stopping=True)
        generated_fix = codet5_tokenizer.decode(outputs_codet5[0], skip_special_tokens=True)
        
        response["suggested_fix"] = generated_fix
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)