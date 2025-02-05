import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Framingham dataset
df = pd.read_csv("C:\\Users\\bodeg\\Desktop\\Tobi project\\app.py\\framingham.csv")  # Ensure the correct file path

# Drop missing values
df.dropna(inplace=True)

# Define input features and target variable
X = df.drop(columns=["TenYearCHD"])  # 'TenYearCHD' is the target variable
y = df["TenYearCHD"]

# Convert to numpy arrays
X = X.values.astype(np.float32)
y = y.values.astype(np.float32).reshape(-1, 1)  # Ensure correct shape

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the Neural Network Model
class CADNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(CADNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.output(x))

# Initialize model
input_size = X_train.shape[1]
model = CADNeuralNetwork(input_size)

# Define Loss and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the Model
epochs = 50
batch_size = 16

for epoch in range(epochs):
    permutation = torch.randperm(X_train_tensor.size(0))
    
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate Model Accuracy
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item() * 100
    print(f"Model Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "cad_model.pth")

# Function for making predictions
def predict_cad(user_input):
    user_input = scaler.transform([user_input])  # Scale input
    user_input_tensor = torch.tensor(user_input, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(user_input_tensor)
    return "High Risk" if prediction.item() > 0.5 else "Low Risk"

# -------------------------- STREAMLIT APP --------------------------

st.title("Coronary Artery Disease Detection")

st.write("Enter your health parameters to predict CAD risk.")

# User inputs
user_inputs = []
for col in df.columns[:-1]:  # Excluding target column
    user_inputs.append(st.number_input(f"{col}", min_value=0.0))

if st.button("Predict CAD Risk"):
    result = predict_cad(user_inputs)
    st.write(f"Prediction: **{result}**")
