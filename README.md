# 🏠 Indian House Price Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting house prices in India using Machine Learning algorithms. It uses property-related features such as area, number of bedrooms, bathrooms, parking availability, furnishing status, location score, and city to estimate house prices.

Two Machine Learning algorithms are implemented and compared:

- Linear Regression
- XGBoost Regressor

The objective is to analyse housing data, train prediction models, evaluate their performance, and identify the important factors influencing house prices.

## 🎯 Project Objectives

- Predict Indian house prices using Machine Learning.
- Understand the relationship between property features and house prices.
- Implement and compare Linear Regression and XGBoost.
- Evaluate model performance using RMSE and R² Score.
- Visualize predictions and feature importance.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Model training and evaluation |
| XGBoost | Advanced regression model |

## 📂 Dataset Description

The project uses an Indian Housing Dataset in CSV format.

**Input Features:**

| Feature | Description |
|---|---|
| Area | House size |
| BHK | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Parking | Parking availability |
| Furnishing | Furnishing level |
| Location_Score | Locality rating |
| City | City of the property |

**Target Variable:** `Price` (in Lakhs ₹)

## ⚙️ Project Workflow

1. Load the housing dataset using Pandas.
2. Explore the dataset and summary statistics.
3. Visualize the distribution of house prices.
4. Apply One-Hot Encoding to the City column.
5. Split the dataset into training and testing sets (80:20).
6. Apply StandardScaler for feature scaling.
7. Train the Linear Regression model.
8. Train the XGBoost Regressor model.
9. Evaluate both models using RMSE and R² Score.
10. Visualize actual vs predicted prices and feature importance.

## 🤖 Machine Learning Algorithms

### 1. Linear Regression

Linear Regression is used as a baseline regression model. It predicts house prices by learning the relationship between input features and the target variable.

### 2. XGBoost Regressor

XGBoost is an ensemble learning algorithm based on decision trees. It builds trees sequentially to improve prediction performance by reducing previous errors.

## 📊 Model Performance

The following results are reported in the project presentation:

| Evaluation Metric | Linear Regression | XGBoost |
|---|---:|---:|
| RMSE | 22.29 | 12.87 |
| R² Score | 0.96 | 0.99 |

**Result:** XGBoost achieved a lower RMSE and higher R² Score compared with Linear Regression in the reported evaluation.

## 📈 Visualizations

The project generates the following visualizations:

- Distribution of House Prices
- Actual vs Predicted House Prices
- Feature Importance using XGBoost

The feature importance analysis highlights the contribution of different input features to the model's predictions.

## 🏡 House Price Prediction

The Python program also includes a `predict_house_price()` function to estimate the price of a new house using its property details.

The function accepts:

- Area
- BHK
- Bathrooms
- Parking
- Furnishing
- Location Score
- City Name

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/jay2977/Indian-House-Price-Prediction-project.git
```

### Step 2: Open the Project Folder

```bash
cd Indian-House-Price-Prediction-project
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Step 4: Run the Python Program

```bash
python ml2.py
```

**Note:** Keep `ml2.py` and `indian_house_project.csv` in the same project folder. The dataset filename must match the filename specified in the Python code.

## ⚠️ Project Limitations

- Limited dataset size.
- No geographical coordinates.
- No advanced hyperparameter tuning.
- Real-time market factors are not considered.

## 🔮 Future Enhancements

- Use a larger and more diverse housing dataset.
- Apply advanced hyperparameter tuning.
- Include additional property and location features.
- Develop an interactive web application for house price prediction.

## 👨‍💻 Author

**Jay Satish Gunjal**

M.Sc. Computer Science  
Modern College of Arts, Science & Commerce, Pune

## 📌 Conclusion

This project demonstrates how Machine Learning can be applied to estimate Indian house prices. Both Linear Regression and XGBoost were implemented and evaluated. Based on the reported results, XGBoost performed better than Linear Regression.

---

**If you find this project useful, feel free to explore the repository!**
