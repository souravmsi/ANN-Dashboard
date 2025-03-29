import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import Huber
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import shap  # SHAP for feature importance
import tensorflow.keras.backend as K
import streamlit as st

dataset_url = "https://raw.githubusercontent.com/souravmsi/dataset-ann/refs/heads/main/dataset.csv"

# Set Streamlit Page Configuration
st.set_page_config(page_title="ANN CLTV Prediction", layout="wide")

# Load dataset
original_dataset = pd.read_csv(dataset_url);
df = pd.read_csv(dataset_url)
df.drop(columns=['Customer_ID'], inplace=True)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

X = df.drop(columns=['Customer_Lifetime_Value'])
y = df['Customer_Lifetime_Value']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize input features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Normalize target values
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# 🏗️ Streamlit UI
st.title("📊 ANN Model Dashboard - CLTV Prediction")
st.sidebar.header("🔧 Model Hyperparameters")

# Hyperparameters from user input:
epochs = st.sidebar.slider("Epochs", 1, 50)  # range: 5 to 50, default: 20
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
activation_function = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
dense_layers = st.sidebar.selectbox("Number of Dense Layers", [1, 2, 3, 4, 5])
neurons_per_layer = st.sidebar.selectbox("Neurons per Layer", [32, 64, 128, 256, 512, 1024])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.7, 0.2, 0.05)

# Select Optimizer based on user input
optimizers = {"adam": Adam(learning_rate), "sgd": SGD(learning_rate), "rmsprop": RMSprop(learning_rate)}
optimizer = optimizers[optimizer_choice]

# 🎛️ Train Model Button
if st.button("🚀 Train Model"):
    with st.spinner("Training model... ⏳"):
        # Build the ANN model dynamically using user inputs
        model = Sequential()
        # First hidden layer requires input_dim specification
        model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation=activation_function))
        model.add(Dropout(dropout_rate))
        # Add additional hidden layers if specified (dense_layers includes first layer)
        for _ in range(dense_layers - 1):
            model.add(Dense(neurons_per_layer, activation=activation_function))
            model.add(Dropout(dropout_rate))
        # Output layer for regression
        model.add(Dense(1, activation='linear'))

        # Define a custom accuracy function based on tolerance (e.g., 10% of true value)
        def regression_accuracy(y_true, y_pred):
            tolerance = 0.10  # Define 10% tolerance
            correct_predictions = tf.keras.backend.abs(y_true - y_pred) <= (tolerance * y_true)
            return tf.keras.backend.mean(tf.keras.backend.cast(correct_predictions, tf.float32))

        # Compile the model with custom metric
        model.compile(optimizer=optimizer, 
                      loss=Huber(delta=1.0),
                      metrics=['mae', 'mse', regression_accuracy])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

        st.success("🎉 Model training complete!")

        # Model Performance
        test_loss, test_mae, test_mse, test_accuracy = model.evaluate(X_test, y_test)
        st.subheader("📊 Model Performance")
        st.metric(label="Test Accuracy", value=f"{test_accuracy:.4f}")
        st.metric(label="Test Loss", value=f"{test_loss:.4f}")

        # Show training history in console (optional)
        print(history.history)

        # 📈 Training Performance Plots
        st.subheader("📈 Training Performance")
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Function to plot training and validation accuracy
        def plot_accuracy(history):
            if 'regression_accuracy' in history.history and 'val_regression_accuracy' in history.history:
                ax[0].plot(history.history['regression_accuracy'], label='Training Accuracy')
                ax[0].plot(history.history['val_regression_accuracy'], label='Validation Accuracy')
                ax[0].set_title('Regression Accuracy over Epochs')
                ax[0].set_xlabel('Epochs')
                ax[0].set_ylabel('Accuracy')
                ax[0].legend()
                ax[0].grid()
            else:
                st.warning("Could not plot accuracy. Check if 'regression_accuracy' and 'val_regression_accuracy' are in the history.")

        # Function to plot training and validation loss
        def plot_loss(history):
            ax[1].plot(history.history['loss'], label='Training Loss')
            ax[1].plot(history.history['val_loss'], label='Validation Loss')
            ax[1].set_title('Loss over Epochs')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Loss')
            ax[1].legend()
            ax[1].grid()

        plot_accuracy(history)
        plot_loss(history)
        st.pyplot(fig)

        # 🔄 Confusion Matrix (using discretized outputs for regression)
        st.subheader("📊 Confusion Matrix")
        def plot_confusion_matrix(y_true, y_pred, bins=10):
            y_true_binned = np.digitize(y_true, bins=np.linspace(y_true.min(), y_true.max(), bins))
            y_pred_binned = np.digitize(y_pred, bins=np.linspace(y_pred.min(), y_pred.max(), bins))
            cm = confusion_matrix(y_true_binned, y_pred_binned)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
        y_pred = model.predict(X_test).flatten()
        plot_confusion_matrix(y_test, y_pred)

        # Classification Report (using discretized outputs)
        def print_classification_report(y_true, y_pred, bins=10):
            y_true_binned = np.digitize(y_true, bins=np.linspace(y_true.min(), y_true.max(), bins))
            y_pred_binned = np.digitize(y_pred, bins=np.linspace(y_pred.min(), y_pred.max(), bins))
            report = classification_report(y_true_binned, y_pred_binned, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        print_classification_report(y_test, y_pred)

        # Feature Importance using SHAP
        def plot_shap_feature_importance(model, X_train):
            st.subheader("🔍 Feature Importance")
            # Load original feature names (excluding dropped columns)
            feature_names = np.array(pd.read_csv(dataset_url).drop(columns=['Customer_ID', 'Customer_Lifetime_Value']).columns)
            # Use a smaller sample for SHAP calculations (for speed)
            sample_X = X_train[:100]
            explainer = shap.Explainer(model, sample_X)
            shap_values = explainer(sample_X)
            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values.values, sample_X, plot_type='bar', feature_names=feature_names, show=False)
            st.pyplot(fig_shap)
            st.subheader("📌 Feature Importance Stats")
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            sorted_idx = np.argsort(feature_importance)[::-1]
            importance_df = pd.DataFrame({
                'Feature': feature_names[sorted_idx],
                'Importance': feature_importance[sorted_idx]
            })
            st.dataframe(importance_df)
        plot_shap_feature_importance(model, X_train)



# Display the dataset and descriptive statistics below the training button.
st.subheader("📋 Sample Dataset")
st.dataframe(original_dataset.head(100))

st.subheader("📊 Descriptive Statistics")
st.write(original_dataset.describe())

# Footer with GitHub Follow Button
footer = """
<div style="text-align: center; margin-top: 50px; display: flex;
    align-items: center;
    justify-items: center;
">
    <img style="width: 4rem; height: 3.5rem; margin-right: 1rem; border-radius: 1rem" src="https://m.media-amazon.com/images/M/MV5BYmMzNTlhZjItYWJiOC00MzdiLTgzYmUtMTE4ZmY3MGNlY2M1XkEyXkFqcGc@._V1_FMjpg_UX1068_.jpg" alt='test'/>
    <a href="https://github.com/swati825" target="_blank">
        <button style="background-color: #24292e; color: white; padding: 10px 20px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer;">
            ⭐ Follow Me on GitHub
        </button>
    </a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)

