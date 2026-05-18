import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import re
import unicodedata
import shap
from lime.lime_text import LimeTextExplainer

# Character-level tokenizer
class CharTokenizer:
    def __init__(self, texts):
        chars = set()
        for text in texts:
            chars.update(list(text))

        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        for ch in sorted(chars):
            self.char2idx[ch] = len(self.char2idx)

    def __call__(self, text):
        return [self.char2idx.get(ch, 1) for ch in list(text)]

    def vocab_size(self):
        return len(self.char2idx)

# Dataset class
class QueryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=400):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Collate function
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels

# Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

# BiLSTM + Attention Model
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.dropout(context)
        return self.fc(context)

# Load data
train_df = pd.read_csv('our_dataset_train_queries.csv')
test_df = pd.read_csv('our_dataset_test_queries.csv')

train_texts = train_df["query"].astype(str).tolist()
train_labels = train_df["label"].tolist()

test_texts = test_df["query"].astype(str).tolist()
test_labels = test_df["label"].tolist()

# Tokenizer built ONLY on training data
tokenizer = CharTokenizer(train_texts)

train_dataset = QueryDataset(train_texts, train_labels, tokenizer)
test_dataset = QueryDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Training setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = BiLSTM_Attention(vocab_size=tokenizer.vocab_size()).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1
)

epochs = 15
best_loss = float('inf')
patience = 3
counter = 0

# Training loop with early stopping
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()
        total_loss += loss.item()

    scheduler.step(total_loss)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    if total_loss < best_loss:
        best_loss = total_loss
        counter = 0
        torch.save(model.state_dict(), "best_bilstm.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# # Load best model
# model.load_state_dict(torch.load("best_bilstm.pt"))

# Evaluation
model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        outputs = model(texts)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())

print("Accuracy:", accuracy_score(all_true, all_preds))
print("Precision:", precision_score(all_true, all_preds))
print("Recall:", recall_score(all_true, all_preds))
print("F2-score:", fbeta_score(all_true, all_preds, beta=2))

# INTERPRETABILITY (WORD-LEVEL, MONGODB-AWARE)
CLASS_NAMES = ["benign", "malicious"]

# 1. MongoDB-aware word tokenizer (for interpretability only)
MONGO_OPS = [
    "$ne", "$nin", "$in", "$gt", "$gte", "$lt", "$lte",
    "$regex", "$where", "$exists", "$and", "$or", "$not"
]

def mongo_tokenize(text):
    # split camelCase: deleteMany -> delete Many
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # keep operators and punctuation as separate tokens
    pattern = r'(\$[a-zA-Z]+|\$[a-zA-Z_]+|\$[a-z]+|\$[a-zA-Z0-9_]+|&&|\|\||==|!=|>=|<=|>|<|;|\[|\]|\{|\}|\(|\)|:|,|"[^"]*"|\S+)'
    tokens = re.findall(pattern, text)

    return tokens

# 2. Normalization + noise filtering
def normalize_token(tok):
    tok = unicodedata.normalize("NFKC", tok)
    tok = tok.replace("\u200b", "")
    tok = tok.strip()
    return tok

def is_noise_token(tok):
    tok = normalize_token(tok)

    if tok == "":
        return True

    # brackets, braces, parentheses
    if tok in {"{", "}", "[", "]", "(", ")"}:
        return True

    # commas and dots
    if tok in {",", "."}:
        return True

    # remove pure punctuation except $, ;, quotes
    if re.fullmatch(r"[^\w$;\"']+", tok):
        return True

    # tiny fragments (fi, di, mo, dy, etc.)
    # if len(tok) <= 1 and tok not in {"$", ";"}:
    #     return True

    return False

# 3. Structural stopwords (schema-level)
STRUCTURAL_STOPWORDS = {
    "db", "users", "products", "login", "orders"
}

def expand_stopwords(stopwords):
    expanded = set()
    for word in stopwords:
        toks = mongo_tokenize(word)
        for t in toks:
            expanded.add(normalize_token(t))
    return expanded

MONGO_STOPWORDS = expand_stopwords(STRUCTURAL_STOPWORDS)

# 4. Prediction wrapper (text -> chars -> model -> probs)
def predict_proba(texts):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for text in texts:
            char_ids = tokenizer(text)
            x = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    return np.array(all_probs)

# 5. SHAP explainer (word-level via Text masker)
MAX_LEN = 400  # same as dataset

def encode_text(text, max_len=MAX_LEN):
    """Convert raw text → padded numeric char IDs."""
    ids = tokenizer(text)
    ids = ids[:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

def model_numeric(x):
    """
    SHAP will pass a numpy array of shape (batch, max_len).
    We convert it to torch tensors and run the BiLSTM.
    """
    x = torch.tensor(x, dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

# SHAP background: a single zero vector
background = np.zeros((1, MAX_LEN), dtype=np.int64)

# KernelExplainer (correct for non-HF models)
explainer = shap.KernelExplainer(
    model_numeric,
    background
)
def shap_all_tokens(text):
    x = encode_text(text).reshape(1, -1)

    shap_vals = explainer.shap_values(x, nsamples=100)

    # SHAP returns shape (1, max_len) → flatten to (max_len,)
    vals = shap_vals[0].flatten()

    tokens = mongo_tokenize(text)

    min_len = min(len(tokens), len(vals))
    tokens = tokens[:min_len]
    vals = vals[:min_len]

    df = pd.DataFrame({"token": tokens, "value": vals})
    df["abs"] = df["value"].abs()

    df = df[~df["token"].apply(is_noise_token)]
    df = df[~df["token"].str.lower().isin(MONGO_STOPWORDS)]

    return df.sort_values("abs", ascending=False)

# 6. LIME explainer (word-level)
lime_explainer = LimeTextExplainer(class_names=CLASS_NAMES, split_expression=mongo_tokenize)

def lime_all_tokens(text):
    exp = lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=50,
        labels=[1]
    )
    tokens = [normalize_token(t) for t, w in exp.as_list(label=1)]
    weights = [w for t, w in exp.as_list(label=1)]

    df = pd.DataFrame({"token": tokens, "value": weights})
    df["abs"] = df["value"].abs()

    df = df[~df["token"].apply(is_noise_token)]
    df = df[~df["token"].str.lower().isin(MONGO_STOPWORDS)]

    return df.sort_values("abs", ascending=False)

# 7. Decision-driving tokens (SHAP + LIME)
# def shap_decision_driving_tokens(text, pred_label, top_n=5):
#     df = shap_all_tokens(text)
#     if pred_label == 1:
#         df = df[df["value"] > 0]
#     else:
#         df = df[df["value"] < 0]
#     return df.head(top_n)
def shap_decision_driving_tokens(text, pred_label, top_n=5):
    df = shap_all_tokens(text)

    # Filter by sign
    if pred_label == 1:
        df = df[df["value"] > 0]
    else:
        df = df[df["value"] < 0]

    # Apply noise + stopword filtering HERE
    df = df[~df["token"].apply(is_noise_token)]
    df = df[~df["token"].str.lower().isin(MONGO_STOPWORDS)]

    # Apply a tiny threshold HERE
    df = df[df["abs"] > 0.00001]

    return df.head(top_n)


def lime_decision_driving_tokens(text, pred_label, top_n=5):
    df = lime_all_tokens(text)
    if pred_label == 1:
        df = df[df["value"] > 0]
    else:
        df = df[df["value"] < 0]
    return df.head(top_n)

# 8. Select samples and build interpretability table
benign_samples = [(t, l) for t, l in zip(test_texts, test_labels) if l == 0][:2]
malicious_samples = [(t, l) for t, l in zip(test_texts, test_labels) if l == 1][:2]
selected_samples = benign_samples + malicious_samples

rows = []
for text, true_label in selected_samples:
    probs = predict_proba([text])[0]
    pred_label = int(np.argmax(probs))
    correct = (pred_label == true_label)

    shap_df = shap_all_tokens(text)
    lime_df = lime_all_tokens(text)
    shap_driving = shap_decision_driving_tokens(text, pred_label)
    lime_driving = lime_decision_driving_tokens(text, pred_label)

    rows.append({
        "Query": text,
        "True Label": "benign" if true_label == 0 else "malicious",
        "Predicted Label": "benign" if pred_label == 0 else "malicious",
        "Correct?": "Yes" if correct else "No",

        "Decision-Driving Tokens (SHAP)": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(shap_driving["token"], shap_driving["value"])
        ),
        "Decision-Driving Tokens (LIME)": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(lime_driving["token"], lime_driving["value"])
        ),

        "LIME Top Tokens": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(lime_df["token"].head(5), lime_df["value"].head(5))
        ),
        "SHAP Top Tokens": ", ".join(
            f"{t} ({v:.3f})" for t, v in zip(shap_df["token"].head(5), shap_df["value"].head(5))
        )
    })

interpret_table = pd.DataFrame(rows)
interpret_table.to_csv("bilstm_interpretability_table.csv", index=False)
print(interpret_table)

# 9. Save HTML explanations for one sample
print("\nSaving LIME + SHAP HTML outputs for BiLSTM...\n")

sample_text = selected_samples[0][0]

lime_exp = lime_explainer.explain_instance(
    sample_text,
    predict_proba,
    num_features=10,
    labels=[1]
)
lime_exp.save_to_file("bilstm_lime_explanation_sample.html")

# shap_values = explainer([s[0] for s in selected_samples])
# html_obj = shap.plots.text(shap_values, display=False)

# SHAP HTML for ALL test queries
# print("\nGenerating SHAP HTML for ALL test queries...\n")

# shap_values_all = explainer(test_texts)   # <--- ALL TEST QUERIES

# html_obj = shap.plots.text(shap_values_all, display=False)

# if hasattr(html_obj, "data"):
#     html_str = html_obj.data
# elif isinstance(html_obj, str):
#     html_str = html_obj
# else:
#     html_str = str(html_obj)

# with open("bilstm_shap_text_summary_ALL.html", "w", encoding="utf-8") as f:
#     f.write(html_str)

# print("Saved SHAP HTML for ALL test queries.")
print("Saved BiLSTM LIME and SHAP HTML files.")
