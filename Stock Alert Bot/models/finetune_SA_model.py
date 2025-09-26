import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate


print("Loading FinancialPhraseBank.csv...")
fpb_df = pd.read_csv(
    "SAmodel/data/FinancialPhraseBank.csv",
    header=None,
    encoding="latin-1",  
    on_bad_lines="skip"
)

fpb_df.columns = ["label", "text"]


if fpb_df["label"].dtype == object:
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    fpb_df["label"] = fpb_df["label"].map(label_map)

fpb_dataset = Dataset.from_pandas(fpb_df[["text", "label"]])


print("Loading fiqa-sentiment-classification.csv...")
fiqa_df = pd.read_csv(
    "SAmodel/data/fiqa-sentiment-classification.csv",
    encoding="latin-1",   
    on_bad_lines="skip"
)

def map_fiqa(score):
    if score < -0.05:
        return 0
    elif score > 0.05:
        return 2
    else:
        return 1

fiqa_df["label"] = fiqa_df["score"].apply(map_fiqa)
fiqa_df = fiqa_df.rename(columns={"sentence": "text"})
fiqa_dataset = Dataset.from_pandas(fiqa_df[["text", "label"]])


print("Loading Sentences_66Agree.txt...")
texts, labels = [], []
with open("SAmodel/data/Sentences_66Agree.txt", "r", encoding="latin-1", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if "@negative" in line:
            labels.append(0)
            texts.append(line.replace("@negative", "").strip())
        elif "@neutral" in line:
            labels.append(1)
            texts.append(line.replace("@neutral", "").strip())
        elif "@positive" in line:
            labels.append(2)
            texts.append(line.replace("@positive", "").strip())

agree66_df = pd.DataFrame({"text": texts, "label": labels})
agree66_dataset = Dataset.from_pandas(agree66_df)


print("Merging datasets...")
dataset = concatenate_datasets([fpb_dataset, fiqa_dataset, agree66_dataset]).shuffle(seed=42)

dataset = dataset.train_test_split(test_size=0.2)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy.compute(predictions=preds, references=p.label_ids)
    f1_score = f1.compute(predictions=preds, references=p.label_ids, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no", 
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,  
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("Training complete. Model saved at ./sentiment_model")
