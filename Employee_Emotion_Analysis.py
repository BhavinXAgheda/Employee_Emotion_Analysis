import tkinter as tk
from tkinter import Toplevel, messagebox, scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
from datetime import datetime
import json
from textblob import TextBlob
import cv2
import numpy as np
from keras.models import load_model
import os


emotion_model = load_model('emotion_model.h5')  # Replace with your model path
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

data_file = "employee_mood_data.json"

def load_mood_data():
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            return json.load(file)
    return {}

def save_mood_data(data):
    with open(data_file, 'w') as file:
        json.dump(data, file, indent=4)

def update_mood_data(employee_id, mood, text=""):
    data = load_mood_data()
    if employee_id not in data:
        data[employee_id] = []
    data[employee_id].append({"timestamp": datetime.now().isoformat(), "mood": mood, "text": text})
    save_mood_data(data)

def analyze_text(text):
    """Analyze sentiment of text input."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    mood = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
    return mood

def analyze_facial_expression():
    """Analyze facial expressions from video feed."""
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Camera", "Press 'q' to capture emotion.")
    mood = 'Neutral'

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Facial Emotion Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi_gray = gray_frame[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray / 255.0
                    roi_gray = roi_gray.reshape(1, 48, 48, 1)
                    prediction = emotion_model.predict(roi_gray)
                    mood = emotion_labels[np.argmax(prediction)]
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return mood

def process_emotions():
    employee_id = employee_id_entry.get().strip()
    text_input = text_box.get("1.0", tk.END).strip()

    if not employee_id:
        messagebox.showwarning("Input Error", "Please enter the Employee ID.")
        return

    if not text_input:
        messagebox.showwarning("Input Error", "Please enter some text for analysis.")
        return

    text_mood = analyze_text(text_input)
    facial_mood = analyze_facial_expression()
    moods = [text_mood, facial_mood]
    final_mood = max(set(moods), key=moods.count)

    mood_result.set(f"Final Detected Mood: {final_mood}")
    task = assign_work_based_on_mood(final_mood)
    task_result.set(f"Suggested Task: {task}")

    update_mood_data(employee_id, final_mood, text_input)

    if final_mood in ['Sad', 'Angry', 'Fear']:
        notify_hr(employee_id, final_mood)

def assign_work_based_on_mood(mood):
    if mood == 'Happy':
        return "Assign a challenging project or collaborative team activity."
    elif mood == 'Sad':
        return "Provide supportive tasks or recommend short breaks."
    elif mood == 'Angry':
        return "Assign independent tasks with less pressure."
    elif mood == 'Fear':
        return "Provide tasks with clear guidance and reassurance."
    elif mood == 'Neutral':
        return "Assign routine tasks to maintain consistency."
    elif mood == 'Surprise':
        return "Encourage the employee to explore innovative tasks."
    elif mood == 'Disgust':
        return "Assign neutral tasks and address concerns privately."
    else:
        return "Monitor and assign tasks matching comfort level."

def notify_hr(employee_id, mood):
    messagebox.showinfo("HR Notification", f"HR has been notified about {employee_id}'s {mood} mood.")

def preprocess_mood_data(data):
    rows = []
    for employee_id, entries in data.items():
        for entry in entries:
            rows.append({
                "employee_id": employee_id,
                "timestamp": datetime.fromisoformat(entry["timestamp"]),
                "mood": entry["mood"],
                "text": entry.get("text", "")
            })
    return pd.DataFrame(rows)

def show_user_data():
    employee_id = employee_id_entry.get().strip()
    if not employee_id:
        messagebox.showwarning("Input Error", "Please enter the Employee ID to view their data.")
        return

    data = load_mood_data()
    if employee_id not in data:
        messagebox.showinfo("No Data", f"No data available for Employee ID: {employee_id}")
        return

    user_data = pd.DataFrame(data[employee_id])
    user_data["timestamp"] = pd.to_datetime(user_data["timestamp"])

    user_window = Toplevel()
    user_window.title(f"Data for Employee ID: {employee_id}")
    user_window.geometry("900x600")

    tree = ttk.Treeview(user_window, columns=("Timestamp", "Mood", "Text"), show="headings")
    tree.heading("Timestamp", text="Timestamp")
    tree.heading("Mood", text="Mood")
    tree.heading("Text", text="Text")

    tree.column("Timestamp", width=200)
    tree.column("Mood", width=100)
    tree.column("Text", width=500)

    for _, row in user_data.iterrows():
        tree.insert("", tk.END, values=(row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), row["mood"], row["text"]))

    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    close_button = tk.Button(
        user_window,
        text="Close",
        font=("Helvetica", 12),
        bg="#e74c3c",
        fg="#ecf0f1",
        command=user_window.destroy
    )
    close_button.pack(pady=10)

def show_advanced_analytics():
    data = load_mood_data()
    if not data:
        return

    df = preprocess_mood_data(data)
    if df.empty:
        messagebox.showinfo("No Data", "No data available for analysis.")
        return

    analytics_window = Toplevel()
    analytics_window.title("Advanced Mood Analytics")
    analytics_window.geometry("1200x900")

    mood_counts = df["mood"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(mood_counts, labels=mood_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
    plt.title("Mood Distribution")
    mood_dist_fig = plt.gcf()
    plt.close()

    canvas1 = FigureCanvasTkAgg(mood_dist_fig, master=analytics_window)
    canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas1.draw()

    transition_matrix = analyze_mood_transitions(df)
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Mood Transition Matrix")
    plt.xlabel("To Mood")
    plt.ylabel("From Mood")
    transition_fig = plt.gcf()
    plt.close()

    canvas2 = FigureCanvasTkAgg(transition_fig, master=analytics_window)
    canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas2.draw()

    wordcloud = generate_word_cloud(df, "Positive")
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud for Positive Mood")
    wordcloud_fig = plt.gcf()
    plt.close()

    canvas3 = FigureCanvasTkAgg(wordcloud_fig, master=analytics_window)
    canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas3.draw()

    close_button = tk.Button(
        analytics_window,
        text="Close",
        font=("Helvetica", 14),
        bg="#e74c3c",
        fg="#ecf0f1",
        command=analytics_window.destroy
    )
    close_button.pack(pady=10)

root = tk.Tk()
root.title("Employee Emotion Analysis")
root.geometry("800x700")
root.configure(bg="#2c3e50")

header = tk.Label(
    root, text="Employee Emotion Analysis System", font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="#ecf0f1"
)
header.pack(pady=20)

frame = tk.Frame(root, bg="#34495e", padx=20, pady=20)
frame.pack(pady=20)

tk.Label(frame, text="Employee ID:", font=("Helvetica", 14), bg="#34495e", fg="#ecf0f1").grid(row=0, column=0, sticky="w")
employee_id_entry = tk.Entry(frame, font=("Helvetica", 14), width=25, bg="#ecf0f1", fg="#2c3e50")
employee_id_entry.grid(row=0, column=1, pady=10)

tk.Label(frame, text="Enter Text for Sentiment Analysis:", font=("Helvetica", 14), bg="#34495e", fg="#ecf0f1").grid(
    row=1, column=0, columnspan=2, sticky="w", pady=(10, 5)
)
text_box = scrolledtext.ScrolledText(frame, font=("Helvetica", 14), width=50, height=5, bg="#ecf0f1", fg="#2c3e50")
text_box.grid(row=2, column=0, columnspan=2, pady=10)

analyze_button = tk.Button(
    root,
    text="Analyze Emotions",
    font=("Helvetica", 14, "bold"),
    bg="#27ae60",
    fg="#ecf0f1",
    padx=10,
    pady=5,
    command=process_emotions,
)
analyze_button.pack(pady=20)

user_data_button = tk.Button(
    root,
    text="Show User Data",
    font=("Helvetica", 14, "bold"),
    bg="#8e44ad",
    fg="#ecf0f1",
    padx=10,
    pady=5,
    command=show_user_data,
)
user_data_button.pack(pady=20)

analytics_button = tk.Button(
    root,
    text="Advanced Analytics",
    font=("Helvetica", 14, "bold"),
    bg="#2980b9",
    fg="#ecf0f1",
    padx=10,
    pady=5,
    command=show_advanced_analytics,
)
analytics_button.pack(pady=20)

mood_result = tk.StringVar()
task_result = tk.StringVar()

result_frame = tk.Frame(root, bg="#2c3e50")
result_frame.pack(pady=20)

tk.Label(result_frame, textvariable=mood_result, font=("Helvetica", 16), bg="#2c3e50", fg="#1abc9c").pack(pady=5)
tk.Label(result_frame, textvariable=task_result, font=("Helvetica", 14), bg="#2c3e50", fg="#ecf0f1", wraplength=600).pack(
    pady=10
)

footer = tk.Label(
    root, text="© BHAVIN AGHEDA", font=("Helvetica", 12), bg="#2c3e50", fg="#bdc3c7"
)
footer.pack(side=tk.BOTTOM, pady=20)

root.mainloop()
