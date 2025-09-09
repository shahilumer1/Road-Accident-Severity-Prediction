# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
from collections import Counter

# 1) Load dataset (point this to your file path if running elsewhere)
df = pd.read_csv(r"C:\Users\UMMAR FAROOK SHAHIL\OneDrive\Desktop\Road Accident Severity Prediction\accidents.csv")

# 2) Columns we’ll use (present in your CSV)
FEATURE_COLS = [
    "Weather_conditions",
    "Road_surface_conditions",
    "Type_of_vehicle",
    "Light_conditions",
]
TARGET_COL = "Accident_severity"

# 3) Basic cleaning: fill NaNs with the mode in each column
for c in FEATURE_COLS + [TARGET_COL]:
    if df[c].isna().any():
        mode_val = df[c].mode(dropna=True)[0]
        df[c] = df[c].fillna(mode_val)

# 4) Build category lists from your data (exact values)
def uniques(col):
    return sorted(df[col].dropna().unique().tolist())

weather_vals = uniques("Weather_conditions")
road_vals    = uniques("Road_surface_conditions")
vehicle_vals = uniques("Type_of_vehicle")
light_vals   = uniques("Light_conditions")
severity_vals= uniques("Accident_severity")   # ['Fatal injury','Serious Injury','Slight Injury']

# 5) Create mappings
def to_map(values):
    return {v:i for i, v in enumerate(values)}

weather_map  = to_map(weather_vals)
road_map     = to_map(road_vals)
vehicle_map  = to_map(vehicle_vals)
light_map    = to_map(light_vals)
severity_map = to_map(severity_vals)          # encode target
severity_rev = {v:k for k,v in severity_map.items()}  # for decoding

# 6) Apply mappings to dataframe
df_enc = pd.DataFrame({
    "weather": df["Weather_conditions"].map(weather_map),
    "road":    df["Road_surface_conditions"].map(road_map),
    "vehicle": df["Type_of_vehicle"].map(vehicle_map),
    "light":   df["Light_conditions"].map(light_map),
})
y = df[TARGET_COL].map(severity_map)

# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df_enc, y, test_size=0.2, random_state=42, stratify=y
)

# 8) Train model (robust + simple to deploy)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 9) Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10) Save model + mappings
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("mappings.pkl", "wb") as f:
    pickle.dump({
        "weather": weather_map,
        "road": road_map,
        "vehicle": vehicle_map,
        "light": light_map,
        "severity_rev": severity_rev,          # int -> label
        # also store the original option lists (handy for UI if ever needed)
        "options": {
            "weather": weather_vals,
            "road": road_vals,
            "vehicle": vehicle_vals,
            "light": light_vals
        }
    }, f)

print("\n✅ Saved: model.pkl and mappings.pkl")
