import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from xgboost import XGBClassifier
import time

start_time = time.time()

RANDOM_STATE = 42

# 1. Load train and test CSV files
TRAIN_PATH = "our_dataset_train_queries_features.csv"
TEST_PATH  = "our_dataset_test_queries_features.csv"
TARGET_COL = "label"       

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

feature_names = list(train_df.drop(columns=[TARGET_COL]).columns)

X_train = train_df.drop(columns=[TARGET_COL]).values
y_train = train_df[TARGET_COL].values

X_test = test_df.drop(columns=[TARGET_COL]).values
y_test = test_df[TARGET_COL].values

# 2. Define F2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2, average="binary")

# 3. Base models
models = {}

models["DecisionTree_ID3"] = DecisionTreeClassifier(
    criterion="entropy",
    random_state=RANDOM_STATE
)

models["RandomForest"] = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

base_tree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=RANDOM_STATE
)
models["AdaBoost_J48"] = AdaBoostClassifier(
    estimator=base_tree,
    random_state=RANDOM_STATE
)

models["NeuralNetwork"] = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(random_state=RANDOM_STATE))
])

models["SVM_C_RBF"] = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        C=1.0,
        kernel="rbf",
        probability=False,
        random_state=RANDOM_STATE
    ))
])

models["kNN"] = KNeighborsClassifier()

models["XGBoost"] = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

# 4. Hyperparameter grids
param_grids = {}

param_grids["DecisionTree_ID3"] = {
    "criterion": ["entropy", "gini"],
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 20]
}

param_grids["RandomForest"] = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

param_grids["AdaBoost_J48"] = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.1, 0.5, 1.0],
    "estimator__max_depth": [1, 2, 3],
    "estimator__criterion": ["entropy", "gini"]
}

# param_grids["NeuralNetwork"] = {
#     "mlp__hidden_layer_sizes": [
#         (32,), (64,), (32, 16), (64, 32), (32, 32, 16)
#     ],
#     "mlp__activation": ["relu", "tanh"],
#     "mlp__solver": ["adam"],
#     "mlp__learning_rate_init": [0.0005, 0.001, 0.005],
#     "mlp__alpha": [0.0001, 0.001, 0.01],
#     "mlp__max_iter": [2000, 4000]
# }
# param_grids["NeuralNetwork"] = {
#     "mlp__hidden_layer_sizes": [
#         (64,), (128,),
#         (64, 32), (128, 64),
#         (64, 32, 16), (128, 64, 32)
#     ],
#     "mlp__activation": ["relu", "tanh"],
#     "mlp__solver": ["adam"],
#     "mlp__learning_rate_init": [0.0001, 0.0005, 0.001],
#     "mlp__alpha": [0.0001, 0.001, 0.01],
#     "mlp__max_iter": [3000, 5000],
#     "mlp__early_stopping": [True],
#     "mlp__validation_fraction": [0.1]
# }
param_grids["NeuralNetwork"] = {
    "mlp__hidden_layer_sizes": [
        (128, 64, 32),
        (128, 128, 64, 32),
        (128, 128, 64, 64, 32)
    ],
    "mlp__learning_rate_init": [0.01, 0.05],
    "mlp__alpha": [0.0001, 0.001, 0.01],
    "mlp__max_iter": [5000, 6000],
    "mlp__activation": ["relu"],
    "mlp__solver": ["adam"],
    "mlp__early_stopping": [False]
}

param_grids["SVM_C_RBF"] = {
    "svm__gamma": [0.1, 0.5, 1.0, 2.0, 5.0],
    "svm__kernel": ["rbf"]
}

param_grids["kNN"] = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

param_grids["XGBoost"] = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 400, 600],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0]
}

# 5. Run GridSearchCV for each model
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

best_models = {}

for name, model in models.items():
    print(f"\n=== Tuning {name} ===")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        scoring=f2_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"Best params for {name}: {grid.best_params_}")
    print(f"Best CV F2 for {name}: {grid.best_score_:.4f}")

    best_models[name] = grid.best_estimator_

def evaluate_with_cv(model, X, y, n_folds=10):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    accuracies = []
    precisions = []
    recalls = []
    f2_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred) * 100)
        precisions.append(precision_score(y_val, y_pred, zero_division=0) * 100)
        recalls.append(recall_score(y_val, y_pred, zero_division=0) * 100)
        f2_scores.append(fbeta_score(y_val, y_pred, beta=2, zero_division=0) * 100)

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f2": np.mean(f2_scores)
    }

print("\n=== 10-fold Cross-Validation Results (Train Data) ===")

for name, model in best_models.items():
    metrics = evaluate_with_cv(model, X_train, y_train, n_folds=10)
    print(f"\n{name}")
    print(f"  Accuracy : {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall   : {metrics['recall']:.2f}%")
    print(f"  F2-score : {metrics['f2']:.2f}%")

# 6. Evaluate best models on test set
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    return acc, prec, rec, f2

print("\n\n=== Test Set Results (Best Models) ===\n")

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc, prec, rec, f2 = compute_metrics(y_test, y_pred)

    print(f"{name}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F2-score : {f2 * 100:.2f}%\n")


# 7. Interpretability functions
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def tree_feature_importance(model, feature_names):
    # Handle plain estimators and Pipelines
    est = model
    if hasattr(model, "named_steps"):
        # try to find a step with feature_importances_
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                est = step
                break
    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        print("\nTree Feature Importances:")
        for idx in sorted_idx[:15]:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")

def permutation_importance_global(model, X, y, feature_names):
    r = permutation_importance(model, X, y, n_repeats=10, random_state=RANDOM_STATE)
    sorted_idx = r.importances_mean.argsort()[::-1]
    print("\nPermutation Importance:")
    for idx in sorted_idx[:15]:
        print(f"{feature_names[idx]}: {r.importances_mean[idx]:.4f}")

def plot_permutation_importance(model, X, y, feature_names):
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importances = r.importances_mean * 100  
    sorted_idx = importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title("Permutation Importance (%)")
    plt.xlabel("Importance (percentage scale)")
    plt.tight_layout()
    plt.show()

def plot_tree_importance(model, feature_names):
    # Extract estimator if model is a pipeline
    est = model
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                est = step
                break

    # If the estimator does NOT have feature_importances_, skip
    if not hasattr(est, "feature_importances_"):
        print("Tree importance not available for this model. Skipping...")
        return

    # Otherwise, plot normally
    importances = est.feature_importances_ * 100
    sorted_idx = importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title("Tree-Based Feature Importance (%)")
    plt.xlabel("Importance (percentage scale)")
    plt.tight_layout()
    plt.show()

def plot_feature_importance_heatmap(models, X, y, feature_names, save_path="perm_importance_heatmap.png"):
    importance_matrix = []

    for name, model in models.items():
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance_matrix.append(r.importances_mean * 100)

    importance_matrix = np.array(importance_matrix)

    plt.figure(figsize=(10, 6)) 
    sns.heatmap(
        importance_matrix,
        annot=False,
        cmap="viridis",
        xticklabels=feature_names,
        yticklabels=list(models.keys())
    )
    plt.title("Feature Importance Across All Models (Permutation ×100)")
    plt.xlabel("Features")
    plt.ylabel("Models")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved heatmap to {save_path}")

    plt.show()

def plot_tree_importance_heatmap(models, feature_names, save_path="tree_importance_heatmap.png"):
    tree_models = {}
    for name, model in models.items():
        est = model
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_importances_"):
                    est = step
                    break
        if hasattr(est, "feature_importances_"):
            tree_models[name] = est

    importance_matrix = []
    model_names = []

    for name, est in tree_models.items():
        importance_matrix.append(est.feature_importances_ * 100)
        model_names.append(name)

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        importance_matrix,
        annot=False,
        cmap="viridis",
        xticklabels=feature_names,
        yticklabels=list(model_names)
    )
    plt.title("Tree-Based Feature Importance Across Models (×100)")
    plt.xlabel("Features")
    plt.ylabel("Models")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved heatmap to {save_path}")

    plt.show()

def interpret_model(name, model, X_train, y_train, X_test, feature_names):
    print(f"\n=== INTERPRETABILITY FOR {name} ===")

    # 1. Global: Tree-based feature importance
    tree_feature_importance(model, feature_names)
    # plot_tree_importance(model, feature_names)

    # 2. Global: Permutation importance
    permutation_importance_global(model, X_test, y_test, feature_names)
    # plot_permutation_importance(model, X_test, y_test, feature_names)

# 8. Run interpretability
for name, model in best_models.items():
    interpret_model(name, model, X_train, y_train, X_test, feature_names)

# Plot heat maps for all models
plot_tree_importance_heatmap(best_models, feature_names)
plot_feature_importance_heatmap(best_models, X_test, y_test, feature_names)

end_time = time.time()
total_minutes = (end_time - start_time) / 60

print(f"\nTotal execution time: {total_minutes:.2f} minutes")
