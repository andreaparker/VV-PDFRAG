import os
import uuid
import json
import time
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from markupsafe import Markup
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from logger import get_logger
from byaldi import RAGMultiModalModel
import markdown
from flask_login import LoginManager, UserMixin, UserMixin, login_user, login_required, logout_user, current_user

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# **Initialize the Flask application**
app = Flask(__name__)

# **Initialize the LoginManager**
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Specify the login route

# Use the secret key from the environment
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret_key')  # Replace with a secure secret key

# Initialize the logger
logger = get_logger(__name__)

# **Configuration Settings**

# Configure upload folders
app.config['UPLOAD_FOLDER'] = 'uploaded_documents'
app.config['STATIC_FOLDER'] = 'static'
app.config['SESSION_FOLDER'] = 'sessions'
app.config['INDEX_FOLDER'] = os.path.join(os.getcwd(), '.byaldi')  # Set to .byaldi folder in current directory

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)

# **User Authentication Setup**

# User model
class User(UserMixin):
    def __init__(self, id):
        self.id = id
        # You can add more attributes if needed

# In-memory user store (for demonstration purposes)
users = {
    'admin': {
        'password': generate_password_hash('ch4ng3m3!')  # Replace with your desired password
    }
}

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# **Application Initialization**

# Initialize global variables
RAG_models = {}  # Dictionary to store RAG models per session
app.config['INITIALIZATION_DONE'] = False  # Flag to track initialization
logger.info("Application started.")

def load_rag_model_for_session(session_id):
    """
    Loads the RAG model for the given session_id from the index on disk.
    """
    index_path = os.path.join(app.config['INDEX_FOLDER'], session_id)

    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

def load_existing_indexes():
    """
    Loads all existing indexes from the .byaldi folder when the application starts.
    """
    global RAG_models
    if os.path.exists(app.config['INDEX_FOLDER']):
        for session_id in os.listdir(app.config['INDEX_FOLDER']):
            if os.path.isdir(os.path.join(app.config['INDEX_FOLDER'], session_id)):
                load_rag_model_for_session(session_id)
    else:
        logger.warning("No .byaldi folder found. No existing indexes to load.")

@app.before_request
def initialize_app():
    """
    Initializes the application by loading existing indexes.
    This will run before the first request, but only once.
    """
    if not app.config['INITIALIZATION_DONE']:
        load_existing_indexes()
        app.config['INITIALIZATION_DONE'] = True
        logger.info("Application initialized and indexes loaded.")

@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

# **Routes**

# **Login and Logout Routes**

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user_record = users.get(username)
        if user_record and check_password_hash(user_record['password'], password):
            user = User(username)
            login_user(user)
            flash('Logged in successfully.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('chat'))
        else:
            flash('Invalid username or password.', 'danger')
            return render_template('login.html')
    else:
        return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# **Home Route**

@app.route('/', methods=['GET'])
@login_required
def home():
    return redirect(url_for('chat'))

# **Chat Route**

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_id = session['session_id']
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    # Load session data from file
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            chat_history = session_data.get('chat_history', [])
            session_name = session_data.get('session_name', 'Untitled Session')
            indexed_files = session_data.get('indexed_files', [])
    else:
        chat_history = []
        session_name = 'Untitled Session'
        indexed_files = []

    if request.method == 'POST' and 'send_query' in request.form:
        query = request.form['query']
        # Retrieve answer length from the form data
        answer_length = request.form.get('answer_length', 'short')
        if 'upload' in request.form:
            # Handle file upload and indexing
            files = request.files.getlist('file')
            session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            os.makedirs(session_folder, exist_ok=True)
            uploaded_files = []
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(session_folder, filename)
                    file.save(file_path)
                    uploaded_files.append(filename)
                    logger.info(f"File saved: {file_path}")
            
            if uploaded_files:
                try:
                    index_name = session_id
                    index_path = os.path.join(app.config['INDEX_FOLDER'], index_name)
                    indexer_model = session.get('indexer_model', 'vidore/colpali')
                    RAG = index_documents(session_folder, index_name=index_name, index_path=index_path, indexer_model=indexer_model)
                    if RAG is None:
                        raise ValueError("Indexing failed: RAG model is None")
                    RAG_models[session_id] = RAG
                    session['index_name'] = index_name
                    session['session_folder'] = session_folder
                    indexed_files.extend(uploaded_files)
                    session_data = {
                        'session_name': session_name,
                        'chat_history': chat_history,
                        'indexed_files': indexed_files
                    }
                    with open(session_file, 'w') as f:
                        json.dump(session_data, f)
                    logger.info("Documents indexed successfully.")
                    return jsonify({
                        "success": True, 
                        "message": "Files indexed successfully.",
                        "indexed_files": indexed_files
                    })
                except Exception as e:
                    logger.error(f"Error indexing documents: {str(e)}")
                    return jsonify({"success": False, "message": f"Error indexing files: {str(e)}"})
            else:
                return jsonify({"success": False, "message": "No files were uploaded."})

        elif 'send_query' in request.form:
            query = request.form['query']
            # Determine answer length preference
            answer_length = 'long' if 'answer_length' in request.form else 'short'
            
            try:
                generation_model = session.get('generation_model', 'qwen')
                resized_height = session.get('resized_height', 280)
                resized_width = session.get('resized_width', 280)
                
                # Retrieve relevant documents
                rag_model = RAG_models.get(session_id)
                if rag_model is None:
                    logger.error(f"RAG model not found for session {session_id}")
                    return jsonify({"success": False, "message": "RAG model not found for this session."})
                
                retrieved_images = retrieve_documents(rag_model, query, session_id)
                logger.info(f"Retrieved images: {retrieved_images}")
                
                # Generate response with full image paths
                full_image_paths = [os.path.join(app.static_folder, img) for img in retrieved_images]
                response = generate_response(
                    full_image_paths, query, session_id, resized_height,
                    resized_width, generation_model, answer_length=answer_length  # Pass the answer length parameter
                )
                
                # Parse markdown in the response
                parsed_response = Markup(markdown.markdown(response))

                # **Inline Section References**
                # Detect and inline section references in the response
                sections_referenced = find_section_references(response)
                if sections_referenced:
                    sections_dict = load_sections_for_session(session_id)
                    section_texts = get_section_texts(sections_referenced, sections_dict)
                    parsed_response = embed_section_text(parsed_response, section_texts)

                # Update chat history
                chat_history.append({"role": "user", "content": query})
                chat_history.append({
                    "role": "assistant", 
                    "content": parsed_response, 
                    "images": retrieved_images  # Keep relative paths for frontend
                })
                
                # Update session name if it's the first message
                if len(chat_history) == 2:  # First user message and AI response
                    session_name = query[:50]  # Truncate to 50 characters
                
                session_data = {
                    'session_name': session_name,
                    'chat_history': chat_history,
                    'indexed_files': indexed_files
                }
                with open(session_file, 'w') as f:
                    json.dump(session_data, f)
                
                # Render the new messages
                new_messages_html = render_template('chat_messages.html', messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": parsed_response, "images": retrieved_images}
                ])
                
                return jsonify({
                    "success": True,
                    "html": new_messages_html
                })
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return jsonify({"success": False, "message": f"An error occurred while generating the response: {str(e)}"})

    # For GET requests, render the chat page
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    chat_sessions = []
    for file in session_files:
        if file.endswith('.json'):
            s_id = file[:-5]
            with open(os.path.join(app.config['SESSION_FOLDER'], file), 'r') as f:
                data = json.load(f)
                name = data.get('session_name', 'Untitled Session')
                chat_sessions.append({'id': s_id, 'name': name})

    model_choice = session.get('model', 'qwen')
    resized_height = session.get('resized_height', 280)
    resized_width = session.get('resized_width', 280)

    return render_template('chat.html', chat_history=chat_history, chat_sessions=chat_sessions,
                           current_session=session_id, model_choice=model_choice,
                           resized_height=resized_height, resized_width=resized_width,
                           session_name=session_name, indexed_files=indexed_files)

# **Additional Routes with @login_required Decorator**

@app.route('/switch_session/<session_id>')
@login_required
def switch_session(session_id):
    session['session_id'] = session_id
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    flash(f"Switched to session.", "info")
    return redirect(url_for('chat'))

@app.route('/rename_session', methods=['POST'])
@login_required
def rename_session():
    session_id = request.form.get('session_id')
    new_session_name = request.form.get('new_session_name', 'Untitled Session')
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        session_data['session_name'] = new_session_name

        with open(session_file, 'w') as f:
            json.dump(session_data, f)

        return jsonify({"success": True, "message": "Session name updated."})
    else:
        return jsonify({"success": False, "message": "Session not found."})

@app.route('/delete_session/<session_id>', methods=['POST'])
@login_required
def delete_session(session_id):
    try:
        session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)
        
        session_images_folder = os.path.join('static', 'images', session_id)
        if os.path.exists(session_images_folder):
            import shutil
            shutil.rmtree(session_images_folder)
        
        RAG_models.pop(session_id, None)
        
        if session.get('session_id') == session_id:
            session['session_id'] = str(uuid.uuid4())
        
        logger.info(f"Session {session_id} deleted.")
        return jsonify({"success": True, "message": "Session deleted successfully."})
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({"success": False, "message": f"An error occurred while deleting the session: {str(e)}"})

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        indexer_model = request.form.get('indexer_model', 'vidore/colpali')
        generation_model = request.form.get('generation_model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)
        session['indexer_model'] = indexer_model
        session['generation_model'] = generation_model
        session['resized_height'] = resized_height
        session['resized_width'] = resized_width
        session.modified = True
        logger.info(f"Settings updated: indexer_model={indexer_model}, generation_model={generation_model}, resized_height={resized_height}, resized_width={resized_width}")
        flash("Settings updated.", "success")
        return redirect(url_for('chat'))
    else:
        indexer_model = session.get('indexer_model', 'vidore/colpali')
        generation_model = session.get('generation_model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)
        return render_template('settings.html', 
                               indexer_model=indexer_model,
                               generation_model=generation_model,
                               resized_height=resized_height, 
                               resized_width=resized_width)

@app.route('/new_session')
@login_required
def new_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    session_number = len([f for f in session_files if f.endswith('.json')]) + 1
    session_name = f"Session {session_number}"
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
    session_data = {
        'session_name': session_name,
        'chat_history': [],
        'indexed_files': []
    }
    with open(session_file, 'w') as f:
        json.dump(session_data, f)
    flash("New chat session started.", "success")
    return redirect(url_for('chat'))

@app.route('/get_indexed_files/<session_id>')
@login_required
def get_indexed_files(session_id):
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            indexed_files = session_data.get('indexed_files', [])
        return jsonify({"success": True, "indexed_files": indexed_files})
    else:
        return jsonify({"success": False, "message": "Session not found."})

# **Helper Functions for Inline Section References**

def find_section_references(answer_text):
    import re
    pattern = r'Section\s+(\d+(\.\d+)*)'
    matches = re.findall(pattern, answer_text)
    sections = [match[0] for match in matches]
    return sections

def load_sections_for_session(session_id):
    """
    Load parsed sections for the session's documents.
    """
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    sections = {}
    for filename in os.listdir(session_folder):
        file_path = os.path.join(session_folder, filename)
        if filename.endswith('.pdf') or filename.endswith('.txt'):
            file_sections = parse_document(file_path)
            sections.update(file_sections)
    return sections

def parse_document(file_path):
    """
    Parse the document and extract sections.
    """
    import re
    sections = {}
    try:
        if file_path.endswith('.pdf'):
            from pdfminer.high_level import extract_text
            text = extract_text(file_path)
        else:
            with open(file_path, 'r') as f:
                text = f.read()
        current_section = None
        for line in text.split('\n'):
            section_match = re.match(r'(Section\s+)?(\d+(\.\d+)*)', line.strip())
            if section_match:
                current_section = section_match.group(2)
                sections[current_section] = ''
            elif current_section:
                sections[current_section] += line + '\n'
    except Exception as e:
        logger.error(f"Error parsing document {file_path}: {e}")
    return sections

def get_section_texts(sections_referenced, sections_dict):
    section_texts = {}
    for sec in sections_referenced:
        text = sections_dict.get(sec)
        if text:
            section_texts[sec] = text
    return section_texts

def embed_section_text(answer_text, section_texts):
    for sec, text in section_texts.items():
        answer_text = answer_text.replace(
            f'Section {sec}',
            f'Section {sec}: "{text.strip()}"'
        )
    return answer_text

# **Run the App**

if __name__ == '__main__':
    app.run(port=5050, debug=True)
