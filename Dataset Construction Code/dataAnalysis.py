import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# CONFIGURATION 
DATASET_PATH = "our_dataset.csv" # Dataset Path

# Manually curated list of MongoDB operators
CURATED_OPERATORS = [
    "$add", "$addFields", "$addToSet", "$all", "$and", "$avg", "$bucket", "$ceil", "$count",
    "$currentDate", "$each", "$elemMatch", "$eq", "$exists", "$expr", "$facet", "$floor", 
    "$function", "$group", "$gt", "$gte", "$in", "$inc", "$indexStats", "$jsonSchema",
    "$limit", "$lookup", "$lt", "$lte", "$match", "$max", "$min", "$mod", "$mul", "$ne",
    "$nin", "$nor", "$not", "$options", "$or", "$out", "$pop", "$project", "$pull", "$pullAll",
    "$push", "$regex", "$rename", "$replaceRoot", "$set", "$setOnInsert", "$size", "$skip",
    "$sort", "$sortByCount", "$strLenCP", "$subtract", "$sum", "$toUpper", "$trunc", "$type",
    "$unset", "$where"
]

# Comprehensive list of MongoDB operations
MONGO_OPERATIONS = [
    "find", "findOne", "insertOne", "insertMany",
    "updateOne", "updateMany", "replaceOne",
    "deleteOne", "deleteMany",
    "aggregate", "count", "countDocuments",
    "distinct", "bulkWrite", "estimatedDocumentCount",
    "findOneAndUpdate", "findOneAndDelete", "findOneAndReplace",
    "findAndModify"
]

# collections
COLLECTIONS = ["login", "users", "products", "orders"]

# Injection type distribution
INJECTION_TYPES = {
    "Tautologies": 340,
    "Piggybacked": 245,
    "JavaScript": 92,
    "Blind": 41,
    "Time-based": 23,
    "Union": 59
}

# HELPERS
def extract_collection(query):
    match = re.search(r"db\.(\w+)\.", query)
    return match.group(1) if match else None

def extract_operation(query):
    match = re.search(r"db\.\w+\.(\w+)", query)
    if match:
        op = match.group(1)
        return op if op in MONGO_OPERATIONS else None
    return None

def extract_operators(query):
    return [op for op in CURATED_OPERATORS if op in query]

# LOAD CSV DATA
df = pd.read_csv(DATASET_PATH)

if "query" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV must contain 'query' and 'label' columns.")

operation_counts = Counter()
operator_counts = Counter()
collection_counts = Counter()

# PARSE DATASET
for query in df["query"]:
    if not isinstance(query, str):
        continue

    col = extract_collection(query)
    if col in COLLECTIONS:
        collection_counts[col] += 1

    op = extract_operation(query)
    if op:
        operation_counts[op] += 1

    for o in extract_operators(query):
        operator_counts[o] += 1

# VISUALIZATION + SAVING
# ----------------------
# 1. CRUD Operations (VERTICAL BAR CHART)
plt.figure(figsize=(12, 6))
ops = list(operation_counts.keys())
vals = list(operation_counts.values())

plt.bar(ops, vals, color="teal")
plt.title("Distribution of Queries per CRUD Operation")
plt.xlabel("CRUD Operation")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("crud_operations_vertical.png", dpi=300)
plt.show()

# 2. Operators → SAVE AS CSV TABLE
operator_df = pd.DataFrame(operator_counts.items(), columns=["operator", "count"])
operator_df.sort_values(by="count", ascending=False, inplace=True)
operator_df.to_csv("operator_counts.csv", index=False)

print("Saved operator counts to operator_counts.csv")

# 3. Collections (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(collection_counts.values(),
        labels=collection_counts.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=["gold", "skyblue", "lightgreen", "salmon"])
plt.title("Distribution of Queries per Collection")
plt.tight_layout()
plt.savefig("collection_distribution.png", dpi=300)
plt.show()

# 4. Injection Types (PIE CHART)
plt.figure(figsize=(8, 8))
plt.pie(INJECTION_TYPES.values(),
        labels=INJECTION_TYPES.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0", "#ffb3e6"])
plt.title("Distribution of Injection Types (Malicious Class)")
plt.tight_layout()
plt.savefig("injection_types_pie.png", dpi=300)
plt.show()


