**Title:**
An Open-Source Approach to Predictive Energy Consumption Using Industrial Process Parameters and Streamlit-Based Web Interface

**Abstract:**
The integration of artificial intelligence (AI) with industrial automation enables more efficient energy management through predictive analytics. This paper presents a practical methodology for predicting instantaneous energy consumption using environmental and process temperature data. Leveraging open-source tools such as Python, Scikit-learn, and Streamlit, this project replaces proprietary platforms like MATLAB with accessible alternatives, thereby democratizing access to advanced analytics. The methodology, model development, and graphical user interface (GUI) design are detailed with clarity to support adoption by non-programmers and control engineers alike.

**1. Introduction**
The use of predictive models in industrial environments enables proactive control strategies, particularly in energy-intensive operations. Traditional implementations often rely on commercial software, which poses cost and accessibility barriers. This study addresses these challenges by demonstrating the use of open-source tools for developing a web-based predictive system for energy consumption.

**2. Data Acquisition and Preparation**
Process data were collected and logged into an InfluxDB database, and later exported in CSV format. The dataset includes:

* Process Temperature (PT)
* Environmental Temperature (ET)
* Instantaneous Energy Consumption (IEC)

The dataset was pre-processed using Python's pandas library to standardize column headers and handle any missing values.

**3. Model Development**
We utilized the scikit-learn library to train a linear regression model. The features (PT, ET) were used to predict the target variable (IEC).

**3.1 Training Procedure:**

1. The dataset was split into training and testing sets (80:20 ratio).
2. A Linear Regression model was instantiated and trained on the training data.
3. The model's performance was validated using the test data.

**3.2 Code Explanation:**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Define features and target
X = df[['process_temperature', 'environmental_temperature']]
y = df['instantaneous_energy_consumption']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'energy_predictor_model.pkl')
```

This code defines the training pipeline, including data splitting, model fitting, and saving the model to a file.

**4. GUI Development Using Streamlit**
Streamlit is a powerful Python library that allows rapid development of interactive web applications.

**4.1 GUI Features:**

* Two input fields: Process Temperature and Environmental Temperature.
* A prediction button that outputs the estimated energy consumption.

**4.2 Streamlit Application Code:**

```python
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('energy_predictor_model.pkl')

st.title('Energy Consumption Predictor')

pt = st.number_input("Enter Process Temperature")
et = st.number_input("Enter Environmental Temperature")

if st.button("Predict Energy Consumption"):
    prediction = model.predict(np.array([[pt, et]]))
    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f}")
```

This code creates a simple interface where users input data and receive instant predictions.

**5. Deployment**
The application can be run locally using the command:

```bash
streamlit run app.py
```

For online deployment, platforms like Streamlit Community Cloud or Heroku can be used.

**6. Conclusion**
This work highlights the feasibility of implementing predictive maintenance and energy optimization using open-source technologies. It offers an accessible pathway for engineers and practitioners without programming backgrounds to leverage AI in industrial contexts.

**References:**

* Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.
* Streamlit Inc. (2023). Streamlit Documentation. [https://docs.streamlit.io/](https://docs.streamlit.io/)
* McKinney, W. (2010). Data Structures for Statistical Computing in Python. In Proceedings of the 9th Python in Science Conference.


**Appendix A: CSV Data Sample Format**

| process\_temperature | environmental\_temperature | instantaneous\_energy\_consumption |
| -------------------- | -------------------------- | ---------------------------------- |
| 74.1                 | 22.5                       | 3.21                               |
| 70.4                 | 23.1                       | 3.08                               |
| 76.3                 | 21.8                       | 3.34                               |

**Appendix B: Model Evaluation Metrics**
To assess the accuracy of the linear regression model, the following metrics were computed using the test data:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
```

These metrics provide quantitative insight into the model’s predictive performance and reliability.
