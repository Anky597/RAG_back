---
title: SHL RAG Recommender API
emoji: 🤖🧠
colorFrom: blue
colorTo: green
sdk: python
python_version: 3.11 # Or your specific Python 3 version (e.g., 3.10)
app_file: app/main.py # <-- Points to your Flask app file
# Add suggested hardware if needed, but start with free tier
# suggested_hardware: cpu-basic
# Add custom startup command if needed (usually not required if app_file points correctly)
# startup_command: "gunicorn --worker-tmp-dir /dev/shm --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:7860 app.main:app" # Port 7860 is default for HF Spaces SDK apps
---

# Your regular README content starts here...

## SHL Assessment Recommendation Engine Backend

This Flask application provides an API endpoint (`/recommend`) to get SHL assessment recommendations based on user queries using a RAG pipeline.
