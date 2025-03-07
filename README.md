# **Car Price Prediction Using Deep Learning**

## **📌 Project Overview**
This project builds a **deep learning regression model** using **TensorFlow/Keras** to predict **car prices** based on multiple vehicle attributes. The dataset contains **205 car records with 26 attributes**, including **engine type, fuel type, horsepower, and car dimensions**.

### **🚀 Key Features**
✅ **Data Preprocessing (Handling Missing Values, Encoding, Scaling)**  
✅ **Feature Engineering for Regression Tasks**  
✅ **Neural Network Model with ReLU Activation**  
✅ **Training and Evaluating the Model**  
✅ **Comparison of Training and Validation Losses**  

---

## **📌 Dataset Description**
The dataset contains **205 rows** and **26 features**, including:
- **Categorical Attributes**: Fuel Type, Aspiration, Car Body, Drive Wheel, etc.
- **Numerical Attributes**: Wheelbase, Car Length, Car Width, Horsepower, Price, etc.

### **📌 Sample Columns**
| symboling | fueltype | carbody | wheelbase | carwidth | horsepower | price |
|-----------|---------|--------|----------|---------|-----------|------|
| 3         | gas     | sedan  | 88.6     | 64.1    | 111       | 13495 |
| 2         | diesel  | hatchback | 94.5   | 65.2    | 154       | 16500 |

---

## **📌 Data Preprocessing**

### **1️⃣ Removing Unnecessary Columns**
```python
df = df.drop(['car_ID', 'CarName'], axis=1)
```

### **2️⃣ Encoding Categorical Features**
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
```

### **3️⃣ Handling Missing Values & Duplicates**
```python
df = df.dropna()
df = df.drop_duplicates()
```

### **4️⃣ Splitting Dataset**
```python
from sklearn.model_selection import train_test_split
X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
```

### **5️⃣ Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## **📌 Neural Network Model**
The model consists of **three fully connected layers** with **ReLU activation** and a final regression output layer.
```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1)
])
```

---

## **📌 Model Compilation & Training**
### **1️⃣ Define Loss and Optimizer**
```python
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
```

### **2️⃣ Training the Model**
```python
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.MeanAbsoluteError()])

history = model.fit(X_train, y_train, epochs=100, validation_split=0.15, batch_size=10)
```

---

## **📌 Training Performance Visualization**
```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.show()
```

### **📊 Key Observations**
✅ **Loss decreases consistently, indicating effective learning.**  
✅ **Mean Absolute Error stabilizes over training epochs.**  

---

## **📌 Installation & Setup**

### **📌 Prerequisites**
- Python 3.x
- TensorFlow, NumPy, Pandas, Matplotlib

### **📌 Install Required Libraries**
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

---

## **📌 Running the Notebook**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YourGitHubUsername/car-price-prediction-ml.git
cd car-price-prediction-ml
```

### **2️⃣ Launch Jupyter Notebook**
```bash
jupyter notebook
```

### **3️⃣ Open and Run the Notebook**
Open `car_price_prediction.ipynb` and execute all cells.

---

## **📌 Conclusion**
This project demonstrates how **deep learning can be applied to predict car prices** based on historical vehicle data. The trained model successfully generalizes relationships between **car attributes and their prices**.

---

## **📌 License**
This project is licensed under the **MIT License**.

