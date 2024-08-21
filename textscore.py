from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceTB/fineweb-edu-classifier"
)

file = input("Please enter the filename: ")
with open(file, "r", encoding="utf-8") as file:
    text = file.read()

inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
outputs = model(**inputs)
logits = outputs.logits.squeeze(-1).float().detach().numpy()
score = logits.item()

rounded_score = round(score, 3)  # Round the score to the third decimal point

print(rounded_score)
