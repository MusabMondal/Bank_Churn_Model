import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess the dataset
def load_dataset():
    global churn_model, sc

    # Load the saved model
    churn_model = tf.keras.models.load_model('Bank_Churn_model/churn_model.h5')

    # Load the scaler
    sc = StandardScaler()

def predict():
    global sc

    # Get the input values
    gender = gender_var.get()
    credit_score = float(credit_score_entry.get())
    age = int(age_entry.get())
    tenure = int(tenure_entry.get())
    balance = float(balance_entry.get())
    num_of_products = int(num_of_products_entry.get())
    has_credit_card = int(has_credit_card_var.get() == 'Yes')
    is_active_member = int(is_active_member_var.get() == 'Yes')
    estimated_salary = float(estimated_salary_entry.get())

    # Encode Gender
    if gender == 'Male':
        gender_encoded = 1
    else:
        gender_encoded = 0

    # Encode Geography
    if country_var.get() == 'France':
        geography = [1, 0, 0]
    elif country_var.get() == 'Spain':
        geography = [0, 0, 1]
    else:  # Germany
        geography = [0, 1, 0]

    # Concatenate features
    features = [geography + [credit_score,gender_encoded,  age, tenure, balance, num_of_products,has_credit_card, is_active_member, estimated_salary]]

    # Scale the features
    features_scaled = sc.transform(features)

    # Predict
    prediction = churn_model.predict(features_scaled)

    # Show the result
    result_label.config(text='Yes' if prediction > 0.5 else 'No')


# Create the main window
root = tk.Tk()
root.title('Bank Churn Prediction')
root.geometry('400x400')

# Load the dataset
load_dataset()

# Create labels and entries for the input features
ttk.Label(root, text='Country:').grid(row=0, column=0, sticky='w')
country_var = tk.StringVar()
country_combobox = ttk.Combobox(root, textvariable=country_var, values=['France', 'Spain', 'Germany'])
country_combobox.grid(row=0, column=1)

ttk.Label(root, text='Credit Score:').grid(row=1, column=0, sticky='w')
credit_score_entry = ttk.Entry(root)
credit_score_entry.grid(row=1, column=1)

ttk.Label(root, text='Gender:').grid(row=9, column=0, sticky='w')
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(root, textvariable=gender_var, values=['Male', 'Female'])
gender_combobox.grid(row=9, column=1)

ttk.Label(root, text='Age:').grid(row=2, column=0, sticky='w')
age_entry = ttk.Entry(root)
age_entry.grid(row=2, column=1)

ttk.Label(root, text='Tenure:').grid(row=3, column=0, sticky='w')
tenure_entry = ttk.Entry(root)
tenure_entry.grid(row=3, column=1)

ttk.Label(root, text='Balance:').grid(row=4, column=0, sticky='w')
balance_entry = ttk.Entry(root)
balance_entry.grid(row=4, column=1)

ttk.Label(root, text='Number of Products:').grid(row=5, column=0, sticky='w')
num_of_products_entry = ttk.Entry(root)
num_of_products_entry.grid(row=5, column=1)

ttk.Label(root, text='Has Credit Card:').grid(row=6, column=0, sticky='w')
has_credit_card_var = tk.StringVar()
has_credit_card_combobox = ttk.Combobox(root, textvariable=has_credit_card_var, values=['Yes', 'No'])
has_credit_card_combobox.grid(row=6, column=1)

ttk.Label(root, text='Is Active Member:').grid(row=7, column=0, sticky='w')
is_active_member_var = tk.StringVar()
is_active_member_combobox = ttk.Combobox(root, textvariable=is_active_member_var, values=['Yes', 'No'])
is_active_member_combobox.grid(row=7, column=1)

ttk.Label(root, text='Estimated Salary:').grid(row=8, column=0, sticky='w')
estimated_salary_entry = ttk.Entry(root)
estimated_salary_entry.grid(row=8, column=1)

# Create a button to make the prediction
predict_button = ttk.Button(root, text='Predict', command=predict)
predict_button.grid(row=10, column=0, columnspan=2)

# Create a label to show the result
result_label = ttk.Label(root, text='')
result_label.grid(row=11, column=0, columnspan=2)

root.mainloop()
