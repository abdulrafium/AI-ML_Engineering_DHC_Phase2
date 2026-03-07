# Telco Customer Churn Prediction

## Objective
Build a reusable and production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.

---

## Dataset
- Telco Churn Dataset (CSV format)
- Contains customer information and churn status.
- Target variable: `Churn` (Yes / No)

---

## Methodology / Approach
1. **Data Preprocessing**
   - Separate numeric and categorical columns.
   - Scale numeric features using `StandardScaler`.
   - Encode categorical features using `OneHotEncoder(handle_unknown='ignore')`.
   - Combine transformations using `ColumnTransformer`.

2. **Model Development**
   - Built a `Pipeline` combining preprocessing and classifier.
   - Trained models like `RandomForestClassifier` (or any other model you choose).

3. **Hyperparameter Tuning**
   - Optional: use `GridSearchCV` to tune hyperparameters.

4. **Model Export**
   - Exported the trained pipeline using `joblib` for reuse:
     ```python
     joblib.dump(pipeline, "telco_churn_pipeline.joblib")
     ```

5. **Testing**
   - Load pipeline:
     ```python
     loaded_pipeline = joblib.load("telco_churn_pipeline.joblib")
     ```
   - Predict on new customer data:
     ```python
     prediction = loaded_pipeline.predict(new_customer)
     probability = loaded_pipeline.predict_proba(new_customer)
     ```

---

## Key Results / Observations
- The pipeline can safely handle **new customers** with unseen categorical values.  
- Provides **probabilities** for both churn and no churn classes.  
- Ready for **production deployment** or further integration in dashboards.

---

## Author
Abdul Rafiu