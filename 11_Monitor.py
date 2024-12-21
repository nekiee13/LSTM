import os
import time
from flask import Flask, jsonify

# Initialize Flask app
app = Flask(__name__)

# Path to the progress file
progress_file = "Temp/Temp.txt"
current_progress = 0.0

# Function to read the progress file
def read_progress():
    global current_progress
    try:
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                current_progress = float(f.read().strip())
    except Exception as e:
        print(f"Error reading progress file: {e}")

# Route to display progress percentage
@app.route('/')
def display_progress():
    read_progress()  # Read the latest progress from file
    return jsonify(progress=f"{current_progress:.2f}%")

# Function to start the Flask server
def start_flask_server():
    print("Starting Flask server...")
    app.run(host='127.0.0.1', port=4444, debug=False)

if __name__ == "__main__":
    start_flask_server()
