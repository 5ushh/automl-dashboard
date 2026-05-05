import pandas as pd
import numpy as np


def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def generate_classification_demo():
    np.random.seed(42)
    n = 300
    age = np.random.randint(22, 60, n)
    experience = np.random.randint(1, 20, n)
    training_score = np.random.randint(50, 100, n)
    department = np.random.choice(["Engineering", "Sales", "HR", "Finance"], n)
    promoted = ((age < 40) & (training_score > 70) & (experience > 5)).astype(int)
    return pd.DataFrame({
        "age": age,
        "experience": experience,
        "training_score": training_score,
        "department": department,
        "promoted": promoted,
    })


def generate_regression_demo():
    np.random.seed(42)
    n = 300
    size = np.random.randint(500, 4000, n)
    bedrooms = np.random.randint(1, 6, n)
    age = np.random.randint(1, 50, n)
    location = np.random.choice(["Urban", "Suburban", "Rural"], n)
    price = (size * 150 + bedrooms * 20000 - age * 1000
             + np.random.normal(0, 15000, n)).astype(int)
    return pd.DataFrame({
        "size_sqft": size,
        "bedrooms": bedrooms,
        "building_age": age,
        "location": location,
        "price": price,
    })


def results_to_dataframe(results):
    rows = []
    for name, info in results.items():
        row = {"Model": name, "Training Time (s)": info.get("training_time", 0)}
        row.update(info.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)
