import os
import sys

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TRAINING_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from cleaning import CLEANED_DATA_PATH, clean, clean_and_save

DATA_PATH = os.path.join(PROJECT_DIR, "data", "hotel_bookings.csv")
MODEL_PATH = os.path.join(TRAINING_DIR, "model.joblib")

NUM_FEATURES = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
                "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
                "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces",
                "total_of_special_requests", "total_nights", "total_guests", "is_family",
                "has_special_req", "has_prev_cancel", "weekend_share"]
CAT_FEATURES = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type"]


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["total_guests"] = d["adults"] + d["children"] + d["babies"]
    d["is_family"] = ((d["children"] + d["babies"]) > 0).astype(int)
    d["has_special_req"] = (d["total_of_special_requests"] > 0).astype(int)
    d["has_prev_cancel"] = (d["previous_cancellations"] > 0).astype(int)
    d["weekend_share"] = d["stays_in_weekend_nights"] / d["total_nights"].replace(0, np.nan)
    return d


def train(data_path=DATA_PATH, model_path=MODEL_PATH, n_estimators=200):
    cleaned_df = clean_and_save(data_path, CLEANED_DATA_PATH)
    data = add_features(cleaned_df)
    X, y = data[NUM_FEATURES + CAT_FEATURES], data["is_canceled"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200))]),
         CAT_FEATURES)])
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced", n_jobs=-1, random_state=42)
    pipe = Pipeline([("prep", pre), ("model", rf)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    print(f"Test accuracy={accuracy_score(y_te, p >= .5):.3f}  ROC-AUC={roc_auc_score(y_te, p):.3f}")

    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)

    X_tr_prep = pipe.named_steps["prep"].transform(X_tr)
    linear_reg = LinearRegression().fit(X_tr_prep, y_tr)
    logistic_reg = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_prep, y_tr)
    decision_tree = DecisionTreeClassifier(max_depth=8, random_state=42).fit(X_tr_prep, y_tr)

    linear_imp = pd.Series(linear_reg.coef_, index=names)
    linear_top = linear_imp.reindex(linear_imp.abs().sort_values(ascending=False).head(12).index).to_dict()

    logistic_imp = pd.Series(logistic_reg.coef_[0], index=names)
    logistic_top = logistic_imp.reindex(logistic_imp.abs().sort_values(ascending=False).head(12).index).to_dict()

    tree_imp = pd.Series(decision_tree.feature_importances_, index=names).sort_values(ascending=False)
    tree_top = tree_imp.head(12).to_dict()

    bundle = {
        "model": pipe,
        "linear_model": linear_reg,
        "logistic_model": logistic_reg,
        "tree_model": decision_tree,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "countries": data["country"].value_counts().head(30).index.tolist(),
        "room_types": sorted(data["reserved_room_type"].unique().tolist()),
        "market_segments": sorted(data["market_segment"].unique().tolist()),
        "distribution_channels": sorted(data["distribution_channel"].unique().tolist()),
        "meals": sorted(data["meal"].unique().tolist()),
        "deposit_types": sorted(data["deposit_type"].unique().tolist()),
        "customer_types": sorted(data["customer_type"].unique().tolist()),
        "importance": imp.head(15).to_dict(),
        "linear_importance": linear_top,
        "logistic_importance": logistic_top,
        "tree_importance": tree_top,
    }
    joblib.dump(bundle, model_path, compress=3)
    print("Saved", model_path)
    return bundle


if __name__ == "__main__":
    train()
