
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")



CSV_PATH = "indian_house_project.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{CSV_PATH}'. Please place 'indian_house_project.csv' in the working directory."
    )

df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded successfully!")
print(df.head())
print("\nDataset Shape:", df.shape)



print("\nSummary Statistics:")
print(df.describe())



plt.figure(figsize=(10, 4))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Distribution of House Prices (in Lakhs ₹)")
plt.xlabel("Price (Lakhs ₹)")
plt.ylabel("Count")
plt.show()



if "City" in df.columns and df["City"].dtype == object:
    df = pd.get_dummies(df, columns=["City"], drop_first=True)



base_features = ['Area', 'BHK', 'Bathrooms', 'Parking', 'Furnishing', 'Location_Score']
city_cols = [col for col in df.columns if col.startswith("City_")]
features = base_features + city_cols
target = "Price"

missing = [c for c in features + [target] if c not in df.columns]
if missing:
    raise ValueError(f"The following expected columns are missing: {missing}")

X = df[features]
y = df[target]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\n📈 Linear Regression Results:")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R² Score: {r2_lr:.2f}")



xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbosity=0,
    objective="reg:squarederror"
)

xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\n🚀 XGBoost Results:")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R² Score: {r2_xgb:.2f}")



plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, label="Linear Regression")
plt.scatter(y_test, y_pred_xgb, alpha=0.5, label="XGBoost")

min_val = min(y_test.min(), y_pred_lr.min(), y_pred_xgb.min())
max_val = max(y_test.max(), y_pred_lr.max(), y_pred_xgb.max())

plt.plot([min_val, max_val], [min_val, max_val], 'k--')
plt.xlabel("Actual Price (Lakhs ₹)")
plt.ylabel("Predicted Price (Lakhs ₹)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()



xgb_importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=xgb_importance.values,
    y=xgb_importance.index,
    hue=xgb_importance.index,
    palette="viridis",
    dodge=False
)
plt.legend([], [], frameon=False)
plt.title("Feature Importance (XGBoost Model)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\nTop features by importance:")
print(xgb_importance.head(10))



def predict_house_price(area, bhk, bathrooms, parking, furnishing, location_score, city_name):

    
    input_dict = {
        "Area": area,
        "BHK": bhk,
        "Bathrooms": bathrooms,
        "Parking": parking,
        "Furnishing": furnishing,
        "Location_Score": location_score
    }

    
    for c in city_cols:
        input_dict[c] = 0


    city_col = "City_" + city_name
    if city_col in input_dict:
        input_dict[city_col] = 1
    else:
        print("❌ City not found in dataset!")
        return None

    
    new_data = pd.DataFrame([input_dict])

    
    new_scaled = scaler.transform(new_data)

    
    predicted_price = xgb.predict(new_scaled)[0]

    return predicted_price



price = predict_house_price(
    area=1400,
    bhk=3,
    bathrooms=2,
    parking=1,
    furnishing=1,
    location_score=8,
    city_name="Hyderabad"
)

print("\n🏠 Predicted Price for New House (Lakhs ₹):", price)
