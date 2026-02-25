import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Strong Dataset banayein (Har career ke liye clear rules)
data = {
    'Maths':      [90, 85, 20, 10, 40, 50, 80, 10, 30, 95],
    'Biology':    [10, 20, 95, 85, 30, 40, 15, 20, 90, 10],
    'Commerce':   [30, 40, 10, 20, 90, 85, 40, 25, 15, 35],
    'Creativity': [50, 45, 30, 40, 20, 30, 20, 95, 40, 60],
    'Career': [
        'Software Developer', 'Software Developer', # High Maths
        'Doctor', 'Doctor',                         # High Biology
        'Business Analyst', 'Business Analyst',     # High Commerce
        'Graphic Designer', 'Graphic Designer',     # High Creativity
        'Doctor', 'Software Developer'              # Mix/Strong cases
    ]
}

df = pd.DataFrame(data)

# 2. Features aur Target alag karein
X = df[['Maths', 'Biology', 'Commerce', 'Creativity']]
y = df['Career']

# 3. Random Forest use karein (Ye Decision Trees se behtar hai accuracy ke liye)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 4. Model save karein
with open("career_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Trained Successfully with Clear Patterns!")