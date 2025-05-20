from flask import Flask, request, render_template_string, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from agent.data_agent import DataAgent

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this in production
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store agent in global scope for simplicity
agent = DataAgent()

template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DataSen Web Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: auto; }
        .chat-box { border: 1px solid #ccc; padding: 15px; min-height: 200px; margin-bottom: 20px; background: #f9f9f9; }
        .user { color: #0074D9; }
        .assistant { color: #2ECC40; }
        .question-form { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 8px; }
        input[type="submit"] { padding: 8px 16px; }
    </style>
</head>
<body>
<div class="container">
    <h2>DataSen - Intelligent Data Analysis Assistant</h2>
    <form method="POST" enctype="multipart/form-data" action="/upload">
        <label>Upload CSV file:</label>
        <input type="file" name="datafile" accept=".csv" required>
        <input type="submit" value="Upload">
    </form>
    {% if data_path %}
        <p><b>Current data file:</b> {{ data_path }}</p>
    {% endif %}
    <div class="chat-box">
        {% for msg in chat %}
            <div class="user"><b>You:</b> {{ msg['user'] }}</div>
            <div class="assistant"><b>Assistant:</b> {{ msg['assistant'] }}</div>
            <hr>
        {% endfor %}
    </div>
    <form method="POST" class="question-form">
        <input type="text" name="question" placeholder="Ask a question..." required>
        <input type="submit" value="Ask">
    </form>
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'chat' not in session:
        session['chat'] = []
    data_path = session.get('data_path')
    chat = session['chat']
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            if data_path:
                answer = agent.analyze(question, data_path)
            else:
                answer = "Please upload a CSV file first."
            chat.append({'user': question, 'assistant': answer})
            session['chat'] = chat
    return render_template_string(template, chat=chat, data_path=os.path.basename(data_path) if data_path else None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'datafile' not in request.files:
        return redirect(url_for('index'))
    file = request.files['datafile']
    if file.filename == '':
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['data_path'] = filepath
    session['chat'] = []  # Reset chat on new upload
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
