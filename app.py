import pandas as pd
from dotenv import load_dotenv
import os
from google import genai
import streamlit as st
import pickle
import matplotlib.pyplot as plt

# ---------- Load API Key ----------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

# ---------- Load ML Model ----------
model = pickle.load(open("career_model.pkl", "rb"))

# ---------- UI ----------
st.title("🎓 Smart Career Guidance System")
st.write("Fill your interest levels and get career suggestion")
# 👇 YAHAN PE PASTE KAREIN (st.title ke niche)
if st.sidebar.button("🗑️ Reset All Data"):
    df_reset = pd.DataFrame(columns=["Name","Maths","Biology","Commerce","Creativity","Career"])
    df_reset.to_csv("student_data.csv", index=False)
    st.sidebar.success("Sari history delete ho gayi!")
# 👆 YAHAN TAK

st.write("Fill your interest levels and get career suggestion")

name = st.text_input("Enter your Name")

maths = st.slider("Maths Interest", 0, 100, 50)
biology = st.slider("Biology Interest", 0, 100, 50)
commerce = st.slider("Commerce Interest", 0, 100, 50)
creativity = st.slider("Creativity Level", 0, 100, 50)

# ---------- Prediction ----------
if st.button("Predict Career"):

    prediction = model.predict([[maths, biology, commerce, creativity]])[0]
    
    prediction = model.predict([[maths, biology, commerce, creativity]])[0]

    # --- Manual Logic Override (Special Case) ---
    if creativity > 70 and biology < 40 and maths < 40:
        prediction = "Graphic Designer"
    elif biology > 80 and maths < 50:
        prediction = "Doctor"
    # --------------------------------------------

    # ---- Save Student Record ----
    record = pd.DataFrame(
        [[name, maths, biology, commerce, creativity, prediction]],
        columns=["Name","Maths","Biology","Commerce","Creativity","Career"]
    )

    record.to_csv("student_data.csv", mode="a", header=False, index=False)

    st.success(f"Recommended Career: {prediction}")
    career_info = {
        "Software Developer": "Builds websites, apps and software using coding.",
        "Doctor": "Treats patients and works in hospitals.",
        "Business Analyst": "Analyzes company data and helps business decisions.",
        "Graphic Designer": "Creates logos, posters and social media designs."
    }

    # .get() use karne se agar career list mein nahi bhi hua toh error nahi aayega
    description = career_info.get(prediction, "Exciting career path with great future opportunities!")
    
    st.info(f"**What is {prediction}?** \n\n {description}")
    # --- 📊 CHART SECTION (YAHAN ADD KIYA HAI) ---
    st.subheader("📊 Your Skill Analysis")
    labels = ["Maths", "Biology", "Commerce", "Creativity"]
    values = [maths, biology, commerce, creativity]
    
    fig, ax = plt.subplots()
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B']
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Interest %')
    
    # Bar ke upar numbers likhne ke liye
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')
    
    st.pyplot(fig) 
    # --------------------------------------------
    # ---------- 5️⃣ RESUME RECOMMENDATION SECTION (Yahan add karein) ----------
    st.write("---")
    st.subheader("📝 Suggested Skills for your Resume")

    skills_db = {
        "Software Developer": ["Python", "HTML/CSS", "JavaScript", "GitHub", "SQL"],
        "Doctor": ["Medical Terminology", "Biology", "Anatomy", "Patient Care", "First Aid"],
        "Business Analyst": ["Excel (Advanced)", "SQL", "Data Visualization", "Communication"],
        "Graphic Designer": ["Canva", "Adobe Photoshop", "UI/UX Design", "Creativity", "Typography"]
    }

    # Prediction ke basis par list fetch karna
    recommended_skills = skills_db.get(prediction, ["Communication", "Problem Solving", "Time Management"])

    # Skills ko Bullet points mein dikhane ke liye
    for s in recommended_skills:
        st.write(f"✅ {s}")
    
    # -----------------------------------------------------------------------

    
    st.write("---")
    st.write("⏳ Generating your 3-month roadmap...")

    # ---------- Gemini Prompt ----------
    prompt = f"""
    Student Name: {name}

    A beginner college student wants to become a {prediction}.

    Create a simple 3 month roadmap including:
    1. Skills to learn
    2. Weekly plan
    3. Free courses (YouTube or free platforms)

    Keep explanation very simple and beginner friendly.
    """

    # ---------- Gemini AI ----------
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    st.write(response.text)
    
    # ========================================================
    # 6️⃣ DOWNLOAD SECTION (Yahan add karein)
    # ========================================================
    st.write("---")
    
    # Poore report ka text tayyar karna
    report_text = f"""
    SMART CAREER GUIDANCE REPORT
    Name: {name}
    Career: {prediction}
    
    Description: {description}
    
    Resume Skills: {', '.join(recommended_skills)}
    
    3-MONTH ROADMAP:
    {response.text}
    """

    st.download_button(
        label="📥 Download Full Roadmap & Career Info",
        data=report_text,
        file_name=f"{name}_Career_Report.txt",
        mime="text/plain"
    )