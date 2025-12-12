## Diabetes

```python
# ==============================================================================
# Step 1 & 2: Multi-Model Pipeline & Hyperparameter Tuning
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, calibration_curve, accuracy_score

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced Learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Define all models and their respective pipelines
models_config = {
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.05, 0.1]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42), 
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear']
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()), 
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__n_neighbors': [5, 10, 15],
            'classifier__weights': ['uniform', 'distance']
        }
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__var_smoothing': [1e-9, 1e-8]
        }
    }
}

# 2. Iteratively execute grid search.
best_estimators = {}
cv_results = {}

print("begin analysis")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, config in models_config.items():
    print(f"🔹 train: {name} ...")
    
    # Construct Pipeline
    steps = config['pipeline_steps'] + [('classifier', config['model'])]
    pipeline = ImbPipeline(steps)
    
    # Grid Search
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=config['params'],
        scoring='roc_auc',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    # Save best model
    best_estimators[name] = grid.best_estimator_
    cv_results[name] = grid.best_score_
    
    print(f"   ✅ best AUC (CV): {grid.best_score_:.4f}")
    print(f"   ✅ best params: {grid.best_params_}\n")

print("all finished")

# ==============================================================================
# Step 3: All Models Evaluation & Visualization
# ==============================================================================

plt.figure(figsize=(16, 7))

# --- sub 1: ROC Curves ---
ax1 = plt.subplot(1, 2, 1)
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=0.8)

# --- sub 2: Calibration Curves ---
ax2 = plt.subplot(1, 2, 2)
ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Perfectly Calibrated', alpha=0.8)

colors = ['#d9534f', '#5bc0de', '#5cb85c', '#f0ad4e', '#428bca', '#967adc'] 
summary_metrics = []

print(f"{'Model':<20} | {'AUC':<10} | {'Brier Score':<12} | {'Accuracy':<10}")
print("-" * 60)

for idx, (name, model) in enumerate(best_estimators.items()):
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    final_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    # Print
    print(f"{name:<20} | {final_auc:.4f}     | {brier:.4f}       | {accuracy_score(y_test, y_pred):.4f}")
    
    # Draw ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {final_auc:.3f})')
    
    # Draw Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='s', markersize=4, color=colors[idx], lw=2, label=f'{name} (Brier = {brier:.3f})')

# Style ROC
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(True, alpha=0.3)

# Style Calibration 
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
ax2.set_ylabel('Fraction of Positives', fontsize=12)
ax2.set_title('Calibration Curves Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc="upper left", fontsize=10) 
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Multi_Model_Comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# Step 4: SHAP Analysis for the Best Model (e.g., XGBoost)
# ==============================================================================
import shap

# Select best model manually or logically
best_model_name = 'XGBoost' 
final_best_model = best_estimators[best_model_name] 

# Extract classifier and imputer
final_xgb_model = final_best_model.named_steps['classifier']
imputer = final_best_model.named_steps['imputer']

# Transform Test Set for SHAP (Imputation only, no SMOTE)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Calculate SHAP values
explainer = shap.TreeExplainer(final_xgb_model)
shap_explanation = explainer(X_test_imputed) 

# Plot Beeswarm
plt.figure()
shap.summary_plot(shap_explanation, X_test_imputed, show=False)
plt.title("SHAP Summary Plot (Global Importance)", fontsize=16)
plt.tight_layout()
plt.savefig("shap_beeswarm.pdf", bbox_inches='tight', dpi=300)
plt.show()

# Plot Waterfall (First sample)
plt.figure()
shap.plots.waterfall(shap_explanation[0], show=False)
plt.title("SHAP Waterfall Plot (Individual Prediction)", fontsize=16)
plt.tight_layout()
plt.savefig("shap_waterfall.pdf", bbox_inches='tight', dpi=300)
plt.show()
```



## Dyslipidemia

```python
# ==============================================================================
# Step 1 & 2: Multi-Model Pipeline & Hyperparameter Tuning (Dyslipidemia Version)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, calibration_curve, accuracy_score

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced Learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- 1. Data Loading & Preprocessing (Specific to Dyslipidemia) ---
DATA_FILE_PATH = 'dyslipe_FOR_PYTHON.txt' 
TARGET_VARIABLE = 'status'

# 核心变量列表 
CATEGORICAL_VARS = [
    "hukou", "edu", "gender", "hypertension", "diabetes", 
    "lung_disease", "heart_disease", "digestive_disease", "drink", "liver_disease"
]
CONTINUOUS_VARS = ["BMI", "systolic", "diastolic", "RC", "age"]

print("--- Step 1: Loading Data... ---")


# 排除失访 
if 'lost_to_followup' in df_original.columns:
    df_filtered = df_original[df_original['lost_to_followup'] == 0].copy()
else:
    df_filtered = df_original.copy()

# One-Hot Encoding
X_categorical = pd.get_dummies(df_filtered[CATEGORICAL_VARS].astype(str), drop_first=True)
X_continuous = df_filtered[CONTINUOUS_VARS]
X = pd.concat([X_continuous, X_categorical], axis=1)
y = df_filtered[TARGET_VARIABLE]

# 划分训练集与测试集 (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 2. Define Models & Pipelines ---
models_config = {
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__min_child_weight': [1, 5] 
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5, 10] 
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42), 
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear']
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()), 
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__n_neighbors': [5, 10, 15, 19], 
            'classifier__weights': ['uniform', 'distance']
        }
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1.0] 
        }
    }
}

# --- 3. Grid Search Execution ---
best_estimators = {}
cv_results = {}

print("--- Step 2: Starting Grid Search... ---")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, config in models_config.items():
    print(f"🔹 Training: {name} ...")
    
    # Construct Pipeline
    steps = config['pipeline_steps'] + [('classifier', config['model'])]
    pipeline = ImbPipeline(steps)
    
    # Grid Search
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=config['params'],
        scoring='roc_auc',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    # Save best model
    best_estimators[name] = grid.best_estimator_
    cv_results[name] = grid.best_score_
    
    print(f"   ✅ Best AUC (CV): {grid.best_score_:.4f}")
    print(f"   ✅ Best Params: {grid.best_params_}\n")

print("All models trained successfully.")

# ==============================================================================
# Step 3: All Models Evaluation & Visualization
# ==============================================================================

plt.figure(figsize=(16, 7))

# --- sub 1: ROC Curves ---
ax1 = plt.subplot(1, 2, 1)
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=0.8)

# --- sub 2: Calibration Curves ---
ax2 = plt.subplot(1, 2, 2)
ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Perfectly Calibrated', alpha=0.8)

colors = ['#d9534f', '#5bc0de', '#5cb85c', '#f0ad4e', '#428bca', '#967adc'] 
summary_metrics = []

print(f"{'Model':<20} | {'AUC':<10} | {'Brier Score':<12} | {'Accuracy':<10}")
print("-" * 60)

for idx, (name, model) in enumerate(best_estimators.items()):
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    final_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    # Print
    print(f"{name:<20} | {final_auc:.4f}     | {brier:.4f}       | {accuracy_score(y_test, y_pred):.4f}")
    
    # Draw ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {final_auc:.3f})')
    
    # Draw Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='s', markersize=4, color=colors[idx], lw=2, label=f'{name} (Brier = {brier:.3f})')

# Style ROC
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(True, alpha=0.3)

# Style Calibration 
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
ax2.set_ylabel('Fraction of Positives', fontsize=12)
ax2.set_title('Calibration Curves Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc="upper left", fontsize=10) 
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Multi_Model_Comparison_Dyslipidemia.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# Step 4: SHAP Analysis for the Best Model
# ==============================================================================
import shap

# Select best model manually or logically
best_model_name = 'XGBoost' 
final_best_model = best_estimators[best_model_name] 

print(f"Performing SHAP analysis for {best_model_name}...")

# Extract classifier and imputer
final_clf_model = final_best_model.named_steps['classifier']
imputer = final_best_model.named_steps['imputer']

# Transform Test Set for SHAP 
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Calculate SHAP values
if best_model_name in ['XGBoost', 'RandomForest']:
    explainer = shap.TreeExplainer(final_clf_model)
    shap_explanation = explainer(X_test_imputed) 
else:
    X_train_summary = shap.kmeans(imputer.transform(X_train), 10)
    explainer = shap.KernelExplainer(final_clf_model.predict_proba, X_train_summary)
    shap_explanation = explainer.shap_values(X_test_imputed)[1] 

# Plot Beeswarm (Global Importance)
plt.figure()
try:
    shap.summary_plot(shap_explanation, X_test_imputed, show=False)
except:
    shap.summary_plot(shap_explanation[1], X_test_imputed, show=False)
    
plt.title(f"SHAP Summary Plot ({best_model_name})", fontsize=16)
plt.tight_layout()
plt.savefig(f"shap_beeswarm_{best_model_name}.pdf", bbox_inches='tight', dpi=300)
plt.show()

# Plot Waterfall (Local Interpretation for the first sample)
if best_model_name in ['XGBoost', 'RandomForest']:
    plt.figure()
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.title("SHAP Waterfall Plot (Individual Prediction)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_{best_model_name}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
```





## Kidney disease

```python
# ==============================================================================
# Step 1 & 2: Multi-Model Pipeline & Hyperparameter Tuning (Kidney Disease Version)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, calibration_curve, accuracy_score

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced Learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- 1. Data Loading & Preprocessing (Specific to Kidney Disease) ---
DATA_FILE_PATH = 'kidneye_FOR_PYTHON.txt' 
TARGET_VARIABLE = 'status'

# 核心变量列表 (来自 kidney.ipynb)
CATEGORICAL_VARS = [
    "hukou", "edu", "gender", "hypertension", "diabetes", 
    "lung_disease", "heart_disease", "digestive_disease", "drink", "liver_disease"
]
CONTINUOUS_VARS = ["BMI", "systolic", "diastolic", "RC", "age"]

print("--- Step 1: Loading Data... ---")


# 排除失访
if 'lost_to_followup' in df_original.columns:
    df_filtered = df_original[df_original['lost_to_followup'] == 0].copy()
else:
    df_filtered = df_original.copy()

# One-Hot Encoding
X_categorical = pd.get_dummies(df_filtered[CATEGORICAL_VARS].astype(str), drop_first=True)
X_continuous = df_filtered[CONTINUOUS_VARS]
X = pd.concat([X_continuous, X_categorical], axis=1)
y = df_filtered[TARGET_VARIABLE]

# 划分训练集与测试集 (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 2. Define Models & Pipelines ---
models_config = {
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__min_child_weight': [1, 5]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, None], 
            'classifier__min_samples_split': [2, 5, 10] 
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42), 
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear']
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()), 
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__n_neighbors': [5, 10, 15, 19], 
            'classifier__weights': ['uniform', 'distance']
        }
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1.0] 
        }
    }
}

# --- 3. Grid Search Execution ---
best_estimators = {}
cv_results = {}

print("--- Step 2: Starting Grid Search... ---")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, config in models_config.items():
    print(f"🔹 Training: {name} ...")
    
    # Construct Pipeline
    steps = config['pipeline_steps'] + [('classifier', config['model'])]
    pipeline = ImbPipeline(steps)
    
    # Grid Search
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=config['params'],
        scoring='roc_auc',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    # Save best model
    best_estimators[name] = grid.best_estimator_
    cv_results[name] = grid.best_score_
    
    print(f"   ✅ Best AUC (CV): {grid.best_score_:.4f}")
    print(f"   ✅ Best Params: {grid.best_params_}\n")

print("All models trained successfully.")

# ==============================================================================
# Step 3: All Models Evaluation & Visualization
# ==============================================================================

plt.figure(figsize=(16, 7))

# --- sub 1: ROC Curves ---
ax1 = plt.subplot(1, 2, 1)
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=0.8)

# --- sub 2: Calibration Curves ---
ax2 = plt.subplot(1, 2, 2)
ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Perfectly Calibrated', alpha=0.8)

colors = ['#d9534f', '#5bc0de', '#5cb85c', '#f0ad4e', '#428bca', '#967adc'] 
summary_metrics = []

print(f"{'Model':<20} | {'AUC':<10} | {'Brier Score':<12} | {'Accuracy':<10}")
print("-" * 60)

for idx, (name, model) in enumerate(best_estimators.items()):
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    final_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    # Print
    print(f"{name:<20} | {final_auc:.4f}     | {brier:.4f}       | {accuracy_score(y_test, y_pred):.4f}")
    
    # Draw ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {final_auc:.3f})')
    
    # Draw Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='s', markersize=4, color=colors[idx], lw=2, label=f'{name} (Brier = {brier:.3f})')

# Style ROC
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(True, alpha=0.3)

# Style Calibration 
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
ax2.set_ylabel('Fraction of Positives', fontsize=12)
ax2.set_title('Calibration Curves Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc="upper left", fontsize=10) 
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Multi_Model_Comparison_Kidney.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# Step 4: SHAP Analysis for the Best Model
# ==============================================================================
import shap

# Select best model manually or logically
best_model_name = 'XGBoost' 
final_best_model = best_estimators[best_model_name] 

print(f"Performing SHAP analysis for {best_model_name}...")

# Extract classifier and imputer
final_clf_model = final_best_model.named_steps['classifier']
imputer = final_best_model.named_steps['imputer']

# Transform Test Set for SHAP 
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Calculate SHAP values
if best_model_name in ['XGBoost', 'RandomForest']:
    explainer = shap.TreeExplainer(final_clf_model)
    shap_explanation = explainer(X_test_imputed) 
else:
    X_train_summary = shap.kmeans(imputer.transform(X_train), 10)
    explainer = shap.KernelExplainer(final_clf_model.predict_proba, X_train_summary)
    shap_explanation = explainer.shap_values(X_test_imputed)[1]

# Plot Beeswarm (Global Importance)
plt.figure()
try:
    shap.summary_plot(shap_explanation, X_test_imputed, show=False)
except:
    shap.summary_plot(shap_explanation[1] if isinstance(shap_explanation, list) else shap_explanation, X_test_imputed, show=False)
    
plt.title(f"SHAP Summary Plot ({best_model_name})", fontsize=16)
plt.tight_layout()
plt.savefig(f"shap_beeswarm_{best_model_name}.pdf", bbox_inches='tight', dpi=300)
plt.show()

# Plot Waterfall (Local Interpretation for the first sample)
if best_model_name in ['XGBoost', 'RandomForest']:
    plt.figure()
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.title("SHAP Waterfall Plot (Individual Prediction)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_{best_model_name}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
```



## Liver disease

```python
# ==============================================================================
# Step 1 & 2: Multi-Model Pipeline & Hyperparameter Tuning (Liver Disease Version)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, calibration_curve, accuracy_score

# Models
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced Learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- 1. Data Loading & Preprocessing (Specific to Liver Disease) ---
DATA_FILE_PATH = 'livere_FOR_PYTHON.txt'
TARGET_VARIABLE = 'status'

# 核心变量列表 
CATEGORICAL_VARS = ["heart_disease", "edu", "drink", "gender", "hukou", "lung_disease"]
CONTINUOUS_VARS = ["age", "BMI", "RC"]

print("--- Step 1: Loading Data... ---")


# 排除失访 
if 'lost_to_followup' in df_original.columns:
    df_filtered = df_original[df_original['lost_to_followup'] == 0].copy()
else:
    df_filtered = df_original.copy()

# One-Hot Encoding
X_categorical = pd.get_dummies(df_filtered[CATEGORICAL_VARS].astype(str), drop_first=True)
X_continuous = df_filtered[CONTINUOUS_VARS]
X = pd.concat([X_continuous, X_categorical], axis=1)
y = df_filtered[TARGET_VARIABLE]

# 划分训练集与测试集 (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 2. Define Models & Pipelines ---
models_config = {
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.1, 0.2], 
            'classifier__min_child_weight': [1, 3] 
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__n_estimators': [200, 300], 
            'classifier__max_depth': [10, 15], 
            'classifier__min_samples_split': [5, 10] 
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42), 
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear']
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()),  
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__C': [0.01, 0.1, 1], 
            'classifier__penalty': ['l1', 'l2']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'pipeline_steps': [
            ('imputer', IterativeImputer(random_state=42)), 
            ('scaler', StandardScaler()), 
            ('smote', SMOTE(random_state=42))
        ],
        'params': {
            'classifier__n_neighbors': [5, 10, 19], 
            'classifier__weights': ['uniform', 'distance']
        }
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'pipeline_steps': [('imputer', IterativeImputer(random_state=42)), ('smote', SMOTE(random_state=42))],
        'params': {
            'classifier__var_smoothing': [1e-9, 1e-4] 
        }
    }
}

# --- 3. Grid Search Execution ---
best_estimators = {}
cv_results = {}

print("--- Step 2: Starting Grid Search... ---")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, config in models_config.items():
    print(f"🔹 Training: {name} ...")
    
    # Construct Pipeline
    steps = config['pipeline_steps'] + [('classifier', config['model'])]
    pipeline = ImbPipeline(steps)
    
    # Grid Search
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=config['params'],
        scoring='roc_auc',
        cv=cv_strategy,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    # Save best model
    best_estimators[name] = grid.best_estimator_
    cv_results[name] = grid.best_score_
    
    print(f"   ✅ Best AUC (CV): {grid.best_score_:.4f}")
    print(f"   ✅ Best Params: {grid.best_params_}\n")

print("All models trained successfully.")

# ==============================================================================
# Step 3: All Models Evaluation & Visualization
# ==============================================================================

plt.figure(figsize=(16, 7))

# --- sub 1: ROC Curves ---
ax1 = plt.subplot(1, 2, 1)
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance', alpha=0.8)

# --- sub 2: Calibration Curves ---
ax2 = plt.subplot(1, 2, 2)
ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Perfectly Calibrated', alpha=0.8)

colors = ['#d9534f', '#5bc0de', '#5cb85c', '#f0ad4e', '#428bca', '#967adc'] 
summary_metrics = []

print(f"{'Model':<20} | {'AUC':<10} | {'Brier Score':<12} | {'Accuracy':<10}")
print("-" * 60)

for idx, (name, model) in enumerate(best_estimators.items()):
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    final_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    # Print
    print(f"{name:<20} | {final_auc:.4f}     | {brier:.4f}       | {accuracy_score(y_test, y_pred):.4f}")
    
    # Draw ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {final_auc:.3f})')
    
    # Draw Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax2.plot(prob_pred, prob_true, marker='s', markersize=4, color=colors[idx], lw=2, label=f'{name} (Brier = {brier:.3f})')

# Style ROC
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(True, alpha=0.3)

# Style Calibration 
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
ax2.set_ylabel('Fraction of Positives', fontsize=12)
ax2.set_title('Calibration Curves Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc="upper left", fontsize=10) 
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Multi_Model_Comparison_Liver.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# Step 4: SHAP Analysis for the Best Model
# ==============================================================================
import shap

# Select best model manually or logically
best_model_name = 'XGBoost' 
final_best_model = best_estimators[best_model_name] 

print(f"Performing SHAP analysis for {best_model_name}...")

# Extract classifier and imputer
final_clf_model = final_best_model.named_steps['classifier']
imputer = final_best_model.named_steps['imputer']

# Transform Test Set for SHAP (Imputation only, no SMOTE)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Calculate SHAP values
if best_model_name in ['XGBoost', 'RandomForest']:
    explainer = shap.TreeExplainer(final_clf_model)
    shap_explanation = explainer(X_test_imputed) 
else:
    X_train_summary = shap.kmeans(imputer.transform(X_train), 10)
    explainer = shap.KernelExplainer(final_clf_model.predict_proba, X_train_summary)
    shap_explanation = explainer.shap_values(X_test_imputed)[1] # 取正类

# Plot Beeswarm (Global Importance)
plt.figure()
try:
    shap.summary_plot(shap_explanation, X_test_imputed, show=False)
except:
    shap.summary_plot(shap_explanation[1], X_test_imputed, show=False)
    
plt.title(f"SHAP Summary Plot ({best_model_name})", fontsize=16)
plt.tight_layout()
plt.savefig(f"shap_beeswarm_{best_model_name}.pdf", bbox_inches='tight', dpi=300)
plt.show()

# Plot Waterfall (Local Interpretation for the first sample)
if best_model_name in ['XGBoost', 'RandomForest']:
    plt.figure()
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.title("SHAP Waterfall Plot (Individual Prediction)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_{best_model_name}.pdf", bbox_inches='tight', dpi=300)
    plt.show()
```

