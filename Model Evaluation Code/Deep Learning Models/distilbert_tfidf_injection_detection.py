import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

import shap
from lime.lime_text import LimeTextExplainer

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim, encoding_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# DistilBERT + TF-IDF Hybrid Classifier
class DistilBERTTextClassifier(nn.Module):
    def __init__(self, bert_model, tfidf_dim, encoding_dim, num_classes, dropout_rate=0.2):
        super(DistilBERTTextClassifier, self).__init__()
        self.bert = bert_model
        self.autoencoder = AutoEncoder(input_dim=tfidf_dim, encoding_dim=encoding_dim, dropout_rate=dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(768 + encoding_dim // 2, num_classes),
            nn.Dropout(dropout_rate)
        )

    def forward(self, input_ids, attention_mask, tfidf_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        cls_token_state = last_hidden_state[:, 0, :]
        encoded_tfidf, _ = self.autoencoder(tfidf_features)
        combined_features = torch.cat((cls_token_state, encoded_tfidf), dim=1)
        output = self.fc(combined_features)
        return output

# Load Data
train_df = pd.read_csv("our_dataset_train_queries.csv")
test_df = pd.read_csv("our_dataset_test_queries.csv")

train_texts = train_df["query"].astype(str).tolist()
test_texts = test_df["query"].astype(str).tolist()

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df["label"])
test_labels = label_encoder.transform(test_df["label"])

# Tokenizer + TF-IDF
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_tokens = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
test_tokens = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

vectorizer = TfidfVectorizer()
tfidf_train = vectorizer.fit_transform(train_texts).toarray()
tfidf_test = vectorizer.transform(test_texts).toarray()

tfidf_dim = tfidf_train.shape[1]

tfidf_train = torch.tensor(tfidf_train, dtype=torch.float32)
tfidf_test = torch.tensor(tfidf_test, dtype=torch.float32)

y_train_full = torch.tensor(train_labels, dtype=torch.long)
y_test = torch.tensor(test_labels, dtype=torch.long)

# Train/Validation Split
X_train_ids_full = train_tokens["input_ids"]
X_train_attention_full = train_tokens["attention_mask"]

(
    X_train_ids,
    X_val_ids,
    X_train_attention,
    X_val_attention,
    X_train_tfidf,
    X_val_tfidf,
    y_train,
    y_val,
) = train_test_split(
    X_train_ids_full,
    X_train_attention_full,
    tfidf_train,
    y_train_full,
    test_size=0.15,
    random_state=42,
)

# Move to GPU
X_train_ids = X_train_ids.to(device)
X_train_attention = X_train_attention.to(device)
X_train_tfidf = X_train_tfidf.to(device)
y_train = y_train.to(device)

X_val_ids = X_val_ids.to(device)
X_val_attention = X_val_attention.to(device)
X_val_tfidf = X_val_tfidf.to(device)
y_val = y_val.to(device)

X_test_ids = test_tokens["input_ids"].to(device)
X_test_attention = test_tokens["attention_mask"].to(device)
X_test_tfidf = tfidf_test.to(device)
y_test = y_test.to(device)

batch_size = 16

train_data = TensorDataset(X_train_ids, X_train_attention, X_train_tfidf, y_train)
val_data = TensorDataset(X_val_ids, X_val_attention, X_val_tfidf, y_val)
test_data = TensorDataset(X_test_ids, X_test_attention, X_test_tfidf, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Model + Optimizer
encoding_dim = 128
dropout_rate = 0.2
learning_rate = 1e-5
num_epochs = 10

distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    output_hidden_states=True,
    num_labels=2,
).to(device)

model = DistilBERTTextClassifier(
    bert_model=distilbert_model,
    tfidf_dim=tfidf_dim,
    encoding_dim=encoding_dim,
    num_classes=2,
    dropout_rate=dropout_rate,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1
)

patience = 3
best_val_loss = float("inf")
patience_counter = 0
best_model_path = "best_distilbert_tfidf_hybrid.pt"

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids, attention_mask, tfidf_features, labels = batch

        optimizer.zero_grad()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tfidf_features=tfidf_features,
        )
        loss = criterion(output, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, tfidf_features, labels = batch
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tfidf_features=tfidf_features,
            )
            loss = criterion(output, labels)
            val_loss += loss.item()

    val_avg_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Train Loss: {avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}"
    )

    scheduler.step(val_avg_loss)

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

# Test Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, tfidf_features, labels = batch
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tfidf_features=tfidf_features,
        )
        _, predicted = torch.max(output, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f2 = fbeta_score(all_labels, all_preds, beta=2)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F2 Score: {f2:.4f}")

# INTERPRETABILITY
CLASS_NAMES = ["benign", "malicious"]
all_texts = test_texts
all_labels = test_labels  # numpy array from LabelEncoder

def predict_proba(texts):
    model.eval()

    # Normalize SHAP inputs to List[str]
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = [str(t) for t in texts.tolist()]
    elif not isinstance(texts, (list, tuple)):
        texts = [str(texts)]
    else:
        texts = [str(t) for t in texts]

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    tfidf = vectorizer.transform(texts).toarray()
    tfidf = torch.tensor(tfidf, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            tfidf_features=tfidf,
        )
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    return probs

def is_noise_token(tok):
    tok = tok.strip()
    if tok == "":
        return True
    if tok in {"{", "}", "[", "]", "(", ")"}:
        return True
    if tok in {",", "."}:
        return True
    if re.fullmatch(r"[^\w$;\"']+", tok):
        return True
    return False

MONGO_STOPWORDS = {
    "db",
    "users",
    "products",
    "login",
    "orders",
}

explainer = shap.Explainer(
    predict_proba,
    shap.maskers.Text(tokenizer),
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

    df = df[~df["token"].apply(is_noise_token)]
    df = df[~df["token"].str.lower().isin(MONGO_STOPWORDS)]

    return df.sort_values("abs", ascending=False)

lime_explainer = LimeTextExplainer(class_names=CLASS_NAMES)

def lime_all_tokens(text):
    # exp = lime_explainer.explain_instance(
    #     text,
    #     predict_proba,
    #     num_features=50,
    #     labels=[1],
    # )
    exp = lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=50,
        labels=[1],
        num_samples=200
    )
    tokens = [t for t, w in exp.as_list(label=1)]
    weights = [w for t, w in exp.as_list(label=1)]

    df = pd.DataFrame({"token": tokens, "value": weights})
    df["abs"] = df["value"].abs()

    df = df[~df["token"].apply(is_noise_token)]
    df = df[~df["token"].str.lower().isin(MONGO_STOPWORDS)]

    return df.sort_values("abs", ascending=False)

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
        df = df[df["value"] > 0]
    else:
        df = df[df["value"] < 0]
    return df.head(top_n)

benign_samples = [(t, l) for t, l in zip(all_texts, all_labels) if l == 0][:2]
malicious_samples = [(t, l) for t, l in zip(all_texts, all_labels) if l == 1][:2]
selected_samples = benign_samples + malicious_samples

rows = []
for text, label in selected_samples:
    probs = predict_proba([text])[0]
    pred_label = int(np.argmax(probs))
    correct = pred_label == label

    lime_df = lime_all_tokens(text)
    shap_df = shap_all_tokens(text)
    driving_df = decision_driving_tokens(text, pred_label)
    lime_driving_df = lime_decision_driving_tokens(text, pred_label)

    rows.append(
        {
            "Query": text,
            "True Label": "benign" if label == 0 else "malicious",
            "Predicted Label": "benign" if pred_label == 0 else "malicious",
            "Correct?": "Yes" if correct else "No",
            "Decision-Driving Tokens (SHAP)": ", ".join(
                f'{t} ({v:.3f})'
                for t, v in zip(driving_df["token"], driving_df["value"])
            ),
            "Decision-Driving Tokens (LIME)": ", ".join(
                f'{t} ({v:.3f})'
                for t, v in zip(lime_driving_df["token"], lime_driving_df["value"])
            ),
            "LIME Top Tokens": ", ".join(
                f'{t} ({v:.3f})'
                for t, v in zip(
                    lime_df["token"].head(5), lime_df["value"].head(5)
                )
            ),
            "SHAP Top Tokens": ", ".join(
                f'{t} ({v:.3f})'
                for t, v in zip(
                    shap_df["token"].head(5), shap_df["value"].head(5)
                )
            ),
        }
    )

interpret_table = pd.DataFrame(rows)
interpret_table.to_csv(
    "distilbert_tfidf_hybrid_interpretability_table.csv", index=False
)
print("\nInterpretability table:")
print(interpret_table)

print("\nSaving full LIME + SHAP HTML outputs...\n")

# lime_exp = lime_explainer.explain_instance(
#     selected_samples[0][0],
#     predict_proba,
#     num_features=10,
#     labels=[1],
# )
lime_exp = lime_explainer.explain_instance(
    selected_samples[0][0],
    predict_proba,
    num_features=10,
    labels=[1],
    num_samples=200
)
lime_exp.save_to_file("distilbert_tfidf_hybrid_lime_sample.html")

print("\nGenerating SHAP HTML for ALL test queries...\n")
shap_values_all = explainer(all_texts)
html = shap.plots.text(shap_values_all, display=False)
with open("distilbert_tfidf_hybrid_shap_all_queries.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Saved SHAP HTML for ALL test queries.")

print("Saved:")
print(" - distilbert_tfidf_hybrid_interpretability_table.csv")
print(" - distilbert_tfidf_hybrid_lime_sample.html")
print(" - distilbert_tfidf_hybrid_shap_all_queries.html")
