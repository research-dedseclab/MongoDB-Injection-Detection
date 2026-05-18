import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    classification_report
)

from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

# Interpretability
from lime.lime_text import LimeTextExplainer
import shap
import re

# DEVICE
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLASS_NAMES = ["benign (0)", "malicious (1)"]

# Dataset
class QueryDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.queries = df["query"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.queries[idx]
        }

# Load data
train_df = pd.read_csv("our_dataset_train_queries.csv")[["query", "label"]]
test_df = pd.read_csv("our_dataset_test_queries.csv")[["query", "label"]]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = QueryDataset(train_df, tokenizer)
test_dataset = QueryDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1
)

# Training Loop
EPOCHS = 10
PATIENCE = 3
best_val_loss = float("inf")
patience_counter = 0

print("\nStarting training...\n")
start_train = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(test_loader)

    scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

    # EARLY STOPPING
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_bert_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

end_train = time.time()
print(f"\nTraining Time: {end_train - start_train:.2f} seconds")

# Evaluation
print("\nEvaluating trained model...\n")

model.eval()

all_preds = []
all_labels = []
all_texts = []

start_eval = time.time()

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch["labels"].numpy()
        texts = batch["text"]

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_texts.extend(texts)

end_eval = time.time()
print(f"Evaluation Time: {end_eval - start_eval:.2f} seconds")

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="binary")
recall = recall_score(all_labels, all_preds, average="binary")
f2 = fbeta_score(all_labels, all_preds, beta=2, average="binary")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F2-score: {f2:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# INTERPRETABILITY BLOCK 
# 0. Build pipeline + predict_proba (you already had this)
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0
)

def predict_proba(texts):
    outputs = pipeline(texts)
    probs = []
    for out in outputs:
        if out["label"] == "LABEL_1":
            probs.append([1-out["score"], out["score"]])
        else:
            probs.append([out["score"], 1-out["score"]])
    return np.array(probs)

# 1. Noise token filter (remove only true noise)
def is_noise_token(tok):
    tok = tok.strip()

    if tok == "":
        return True

    # remove brackets, braces, parentheses
    if tok in {"{", "}", "[", "]", "(", ")"}:
        return True

    # remove commas and dots
    if tok in {",", "."}:
        return True

    # remove pure punctuation except $, ;, quotes
    if re.fullmatch(r"[^\w$;\"']+", tok):
        return True

    # # remove BERT subword continuation
    # if tok.startswith("##"):
    #     return True

    # # remove tiny fragments like fi, dy, mo, di
    # if len(tok) <= 1 and tok not in {"$", ";"}:
    #     return True

    return False

# 2. MongoDB structural stopwords (schema-level, not semantic)
MONGO_STOPWORDS = {
    "db", "users", "products", "login", "orders" 
}

# 3. SHAP explainer
explainer = shap.Explainer(
    pipeline,
    shap.maskers.Text(tokenizer)
)

def shap_all_tokens(text):
    sv = explainer([text])
    tokens = sv.data[0]
    vals = sv.values[0][:, 1]

    min_len = min(len(tokens), len(vals))
    tokens = tokens[:min_len]
    vals = vals[:min_len]

    df = pd.DataFrame({"token": tokens, "value": vals})
    df["abs"] = df["value"].abs()

    # remove noise
    df = df[~df["token"].apply(is_noise_token)]

    # remove MongoDB structural tokens
    df = df[~df["token"].str.lower().isin(MONGO_STOPWORDS)]

    return df.sort_values("abs", ascending=False)

# 4. LIME explainer
lime_explainer = LimeTextExplainer(class_names=CLASS_NAMES)

def lime_all_tokens(text):
    exp = lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=50,
        labels=[1]
    )
    tokens = [t for t, w in exp.as_list(label=1)]
    weights = [w for t, w in exp.as_list(label=1)]

    df = pd.DataFrame({"token": tokens, "value": weights})
    df["abs"] = df["value"].abs()

    # remove noise + stopwords
    df = df[~df["token"].apply(is_noise_token)]
    df = df[~df["token"].str.lower().isin(MONGO_STOPWORDS)]

    return df.sort_values("abs", ascending=False)

# 5. Decision-driving tokens (SHAP)
def decision_driving_tokens(text, pred_label, top_n=5):
    df = shap_all_tokens(text)

    if pred_label == 1:
        df = df[df["value"] > 0]
    else:
        df = df[df["value"] < 0]

    return df.head(top_n)

def lime_decision_driving_tokens(text, pred_label, top_n=5):
    df = lime_all_tokens(text)

    if pred_label == 1:
        df = df[df["value"] > 0]   # tokens pushing toward malicious
    else:
        df = df[df["value"] < 0]   # tokens pushing toward benign

    return df.head(top_n)

# 6. Select 4 samples
benign_samples = [(t, l) for t, l in zip(all_texts, all_labels) if l == 0][:2]
malicious_samples = [(t, l) for t, l in zip(all_texts, all_labels) if l == 1][:2]
selected_samples = benign_samples + malicious_samples

# 7. Build interpretability table
rows = []
for text, label in selected_samples:
    pred = pipeline(text)[0]["label"]
    pred_label = 1 if pred == "LABEL_1" else 0
    correct = (pred_label == label)

    lime_df = lime_all_tokens(text)
    shap_df = shap_all_tokens(text)
    driving_df = decision_driving_tokens(text, pred_label)
    lime_driving_df = lime_decision_driving_tokens(text, pred_label)

    rows.append({
        "Query": text,
        "True Label": "benign" if label == 0 else "malicious",
        "Predicted Label": "benign" if pred_label == 0 else "malicious",
        "Correct?": "Yes" if correct else "No",

        "Decision-Driving Tokens (SHAP)": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(driving_df["token"], driving_df["value"])
        ),

        "Decision-Driving Tokens (LIME)": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(lime_driving_df["token"], lime_driving_df["value"])
        ),

        "LIME Top Tokens": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(lime_df["token"].head(5), lime_df["value"].head(5))
        ),

        "SHAP Top Tokens": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(shap_df["token"].head(5), shap_df["value"].head(5))
        )
    })

interpret_table = pd.DataFrame(rows)
interpret_table.to_csv("interpretability_table.csv", index=False)
print(interpret_table)

# Save full HTML outputs
print("\nSaving full LIME + SHAP HTML outputs...\n")

# LIME HTML
lime_exp = lime_explainer.explain_instance(
    selected_samples[0][0],
    predict_proba,
    num_features=10,
    labels=[1]
)
lime_exp.save_to_file("lime_explanation_sample.html")

# # SHAP HTML
# shap_values = explainer([s[0] for s in selected_samples])
# html_obj = shap.plots.text(shap_values, display=False)

# SHAP HTML for ALL test queries
print("\nGenerating SHAP HTML for ALL test queries...\n")

shap_values_all = explainer(all_texts)   # <--- ALL TEST QUERIES

html_obj = shap.plots.text(shap_values_all, display=False)

if hasattr(html_obj, "data"):
    html_str = html_obj.data
elif isinstance(html_obj, str):
    html_str = html_obj
else:
    html_str = str(html_obj)

with open("shap_text_summary.html", "w", encoding="utf-8") as f:
    f.write(html_str)

print("Saved SHAP HTML for ALL test queries.")
print("Saved LIME and SHAP HTML files.")
