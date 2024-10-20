from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, ClassLabel, Value, DatasetDict

app = Flask(__name__)
data = {
    "train": [
        {"code": "os.system(command)", "label": 3},  # Example vulnerability labels
        {"code": "input(username)", "label": 0},
        # Add more training examples...
    ],
    "validation": [
        {"code": "os.system(command)", "label": 3},
        {"code": "input(username)", "label": 0},
        # Add more validation examples...
    ],
}

# Convert to Dataset
train_dataset = DatasetDict(data)
train_dataset = train_dataset["train"]
val_dataset = train_dataset["validation"]

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["code"], padding="max_length", truncation=True)

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Fine-tune the model
trainer.train()
codebert_model_name = "microsoft/codebert-base"
codebert_tokenizer = RobertaTokenizer.from_pretrained(codebert_model_name)

# Specify the number of labels your model should predict
num_labels = 5  # Adjust this based on your actual labels

# Load the model with the number of labels
codebert_model = RobertaForSequenceClassification.from_pretrained(codebert_model_name, num_labels=num_labels)

# Load CodeT5 for generating fixes
codet5_model_name = "Salesforce/codet5-base"
codet5_tokenizer = AutoTokenizer.from_pretrained(codet5_model_name)
codet5_model = AutoModelForSeq2SeqLM.from_pretrained(codet5_model_name)

# Define vulnerability classes
vulnerability_classes = ["No Vulnerability", "SQL Injection", "Buffer Overflow", "Command Injection", "Cross-site Scripting (XSS)"]

@app.route('/')
def home():
    return render_template('index.html')  # Load your HTML file

@app.route('/check_vulnerability', methods=['POST'])
def check_vulnerability():
    data = request.get_json()
    code_snippet = data.get('code_snippet')
    print(code_snippet)

    # Step 1: Detect Vulnerability using CodeBERT
    inputs = codebert_tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = codebert_model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

    predicted_class = vulnerability_classes[predictions.item()]
    response = {"detected_vulnerability": predicted_class}

    # Step 2: Generate a fix using CodeT5 if needed
    if predicted_class != "No Vulnerability":
        inputs_codet5 = codet5_tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs_codet5 = codet5_model.generate(inputs_codet5.input_ids, max_length=150, num_beams=4, early_stopping=True)
        generated_fix = codet5_tokenizer.decode(outputs_codet5[0], skip_special_tokens=True)
        
        response["suggested_fix"] = generated_fix
        print(jsonify(response))
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)


