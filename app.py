import pandas as pd
import numpy as np
import streamlit as st
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

#  Custom Outlier Capper 
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bound_ = Q1 - self.factor * IQR
        self.upper_bound_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)

#  Streamlit App 
st.set_page_config(page_title="NBA Model Pipeline", layout="wide")
st.title("NBA Player Stats - ML Pipeline")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("nba.csv")
    return df

df = load_data()

st.subheader("Raw data preview")
st.dataframe(df.head(10), use_container_width=True)

# Basic cleaning: remove duplicates
df = df.drop_duplicates()

# Drop columns that are all NaN or completely constant
df = df.dropna(axis=1, how="all")
for col in df.columns:
    if df[col].nunique() <= 1:
        df.drop(columns=col, inplace=True)

# Remove identifier columns that are not useful for prediction
if "player_name" in df.columns:
    df.drop(columns=["player_name"], inplace=True)

# Let user select target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target = st.selectbox("Select target variable", options=numeric_cols, index=numeric_cols.index("pts") if "pts" in numeric_cols else 0)

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

st.write(f"Target: **{target}**")
st.write(f"Features shape: {X.shape}")

# Determine categorical and numerical features
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

st.write(f"Numerical features: {len(num_cols)}  |  Categorical features: {len(cat_cols)}")

# Run pipeline button
if st.button("Run Full Pipeline"):
    #  Split data 
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    st.write(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

    #  Preprocessor 
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("capper", OutlierCapper(factor=1.5)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    #  Candidate models 
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor()
    }

    results = []
    best_r2 = -np.inf
    best_model_name = ""

    st.write("---")
    st.subheader("Model comparison (on validation set)")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name} ...")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

        progress_bar.progress((i + 1) / len(models))

    status_text.text("")
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    st.dataframe(results_df, use_container_width=True)

    st.success(f"Best model based on R²: **{best_model_name}** (R² = {best_r2:.4f})")

    #  Hyperparameter tuning 
    st.subheader("Hyperparameter tuning on best model")
    param_grid = {}

    if best_model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        }
    elif best_model_name == "Gradient Boosting":
        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1]
        }
    elif best_model_name == "Ridge":
        param_grid = {"model__alpha": [0.1, 1, 10, 100]}
    elif best_model_name == "Lasso":
        param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1]}
    elif best_model_name == "KNN":
        param_grid = {"model__n_neighbors": [3, 5, 7, 9]}
    else:  # Linear Regression – no hyperparameters
        st.info("Linear Regression has no hyperparameters to tune.")
        param_grid = {}

    if param_grid:
        base_pipe = Pipeline([("preprocessor", preprocessor), ("model", models[best_model_name])])
        grid = GridSearchCV(base_pipe, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)

        st.write("Best parameters found:")
        st.json(grid.best_params_)
        best_pipe = grid.best_estimator_
    else:
        # No tuning, just refit the model (already trained on train, but for completeness)
        best_pipe = Pipeline([("preprocessor", preprocessor), ("model", models[best_model_name])])
        best_pipe.fit(X_train, y_train)

    #  Retrain on train + validation 
    st.write("Retraining tuned model on train + validation combined ...")
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    # Use the best parameters found (if any) to create a new pipeline and fit
    final_pipe = Pipeline([("preprocessor", preprocessor), ("model", models[best_model_name])])
    if param_grid:
        final_pipe.set_params(**grid.best_params_)
    final_pipe.fit(X_train_val, y_train_val)

    #  Final evaluation on test set 
    st.subheader("Final evaluation on untouched test set")
    y_test_pred = final_pipe.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{test_mae:.3f}")
    col2.metric("MSE", f"{test_mse:.3f}")
    col3.metric("RMSE", f"{test_rmse:.3f}")
    col4.metric("R²", f"{test_r2:.3f}")

    #  Save pipeline 
    joblib.dump(final_pipe, "model.pkl")
    st.success("Pipeline saved as model.pkl")

    with open("model.pkl", "rb") as f:
        st.download_button("Download model.pkl", f, file_name="model.pkl")

    # Optional: feature importance for tree-based models
    if best_model_name in ["Random Forest", "Gradient Boosting"]:
        try:
            # Get feature names after preprocessing
            preprocessor = final_pipe.named_steps["preprocessor"]
            model = final_pipe.named_steps["model"]
            cat_onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_feature_names = cat_onehot.get_feature_names_out(cat_cols).tolist()
            feature_names = num_cols + cat_feature_names

            importances = model.feature_importances_
            feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(20)
            st.subheader("Top 20 feature importances")
            st.dataframe(feat_imp, use_container_width=True)
        except Exception:
            pass