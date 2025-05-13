from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.schema.messages import HumanMessage, AIMessage
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from src.memory import ChatbotMemoryManager
from store_index import process_files
from pinecone_setup import initialize_pinecone
from pinecone import Pinecone
from langchain.memory import ConversationBufferMemory
import secrets
import hashlib
import os
import shutil
import time

app = Flask(__name__)

def timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

app.jinja_env.filters['timestamp_to_datetime'] = timestamp_to_datetime

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'Data'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'json', 'txt', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"

chatbot_memory = ChatbotMemoryManager(
    expiry_minutes=30,
    max_sessions=100,
    max_history_length=10
)

print("Initializing Pinecone...")
index_name = "university"
pc = initialize_pinecone(index_name=index_name)

print("Loading embeddings...")
embeddings = download_hugging_face_embeddings()

try:
    print(f"Attempting to connect to existing index '{index_name}'...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("Successfully connected to Pinecone index!")
except ValueError as e:
    print(f"Error connecting to index: {e}")
    print("Creating empty vector store as fallback...")
    from langchain_core.documents import Document
    dummy_doc = Document(page_content="Initialization document", metadata={"source": "init"})
    docsearch = PineconeVectorStore.from_documents(
        documents=[dummy_doc],
        embedding=embeddings,
        index_name=index_name
    )
    print("Created empty vector store.")

print("Setting up retriever...")
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

print("Initializing LLM...")
llm = ChatGroq(
    temperature=0.2,
    max_tokens=500,
    model_name="llama3-70b-8192"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("human", "Chat History: {chat_history}"),
    ]
)

print("Creating question-answer chain...")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("Chatbot setup complete!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session or not session['admin_logged_in']:
            flash('Please log in to access the admin panel', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    session_id = request.form.get("session_id", "default")
    
    # Get or create session
    if not chatbot_memory.get_session(session_id):
        session_id = chatbot_memory.create_session(session_id)
    
    # Get the full conversation history
    chat_history = chatbot_memory.get_chat_history(session_id)
    
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history
    })
    
    chatbot_memory.add_user_message(session_id, msg)
    chatbot_memory.add_ai_message(session_id, response["answer"])
    
    return str(response["answer"])

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session_id = request.form.get("session_id")
    
    if not session_id and request.content_type == 'application/json':
        data = request.get_json()
        session_id = data.get('session_id')
    
    if session_id:
        chatbot_memory.clear_session(session_id)
        return jsonify({"status": "success", "message": "Chat history cleared"})
    
    return jsonify({"status": "error", "message": "No session ID provided"})

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            files.append({
                'name': filename,
                'size': f"{os.path.getsize(file_path) / 1024:.2f} KB",
                'date': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                'processed': is_file_processed(filename)
            })

    stats = {
        "active_sessions": len(chatbot_memory.sessions),
        "session_details": [
            {
                "id": sid, 
                "created_at": session.created_at, 
                "last_accessed": session.last_accessed_at,
                "message_count": len(session.get_messages())  # Updated line
            } for sid, session in chatbot_memory.sessions.items()
        ],
        "expiry_minutes": chatbot_memory.expiry_seconds // 60,
        "max_sessions": chatbot_memory.max_sessions,
        "max_history_length": chatbot_memory.max_history_length
    }

    return render_template('admin_dashboard.html', files=files, stats=stats)

def is_file_processed(filename):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("university")
        result = index.query(
            vector=[0]*384,
            top_k=1,
            filter={"pdf_name": filename}
        )
        return len(result['matches']) > 0
    except Exception:
        return False

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        if username == ADMIN_USERNAME and hashed_password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin/upload', methods=['POST'])
@admin_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('admin_dashboard'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('admin_dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(file_path):
            flash(f'File {filename} already exists. Please rename your file or delete the existing one.', 'warning')
            return redirect(url_for('admin_dashboard'))
        
        file.save(file_path)
        flash(f'File {filename} uploaded successfully! Click on Process button to add it to the knowledge base.', 'success')
        return redirect(url_for('admin_dashboard'))

    flash('File type not allowed', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/process/<filename>', methods=['POST'])
@admin_required
def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    try:
        new_data_folder = os.path.join('Data', 'temp')
        os.makedirs(new_data_folder, exist_ok=True)
        temp_file_path = os.path.join(new_data_folder, filename)
        
        shutil.copy2(file_path, temp_file_path)
        result = process_files(data_dir=new_data_folder, index_name='university')

        os.remove(temp_file_path)
        if os.path.exists(new_data_folder) and not os.listdir(new_data_folder):
            os.rmdir(new_data_folder)

        if result and hasattr(result, 'processed_chunks') and result.processed_chunks > 0:
            flash(f'File {filename} processed and added to knowledge base!', 'success')
        else:
            flash(f'No content could be extracted from {filename}. The file may contain only scanned images or unsupported content.', 'warning')
            
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'danger')

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/process-all', methods=['POST'])
@admin_required
def process_all_files():
    unprocessed_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)) and not is_file_processed(filename):
            unprocessed_files.append(filename)
    
    if not unprocessed_files:
        flash('No pending files to process', 'info')
        return redirect(url_for('admin_dashboard'))
    
    processed_count = 0
    error_count = 0
    
    for filename in unprocessed_files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            continue
            
        try:
            new_data_folder = os.path.join('Data', 'temp')
            os.makedirs(new_data_folder, exist_ok=True)
            temp_file_path = os.path.join(new_data_folder, filename)
            
            shutil.copy2(file_path, temp_file_path)
            process_files(data_dir=new_data_folder, index_name='university')

            os.remove(temp_file_path)
            if os.path.exists(new_data_folder) and not os.listdir(new_data_folder):
                os.rmdir(new_data_folder)
                
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            app.logger.error(f"Error processing file {filename}: {str(e)}")
    
    if processed_count > 0:
        flash(f'Successfully processed {processed_count} files', 'success')
    if error_count > 0:
        flash(f'Failed to process {error_count} files. Check logs for details.', 'warning')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete/<filename>')
@admin_required
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("university")
        index.delete(filter={"pdf_name": filename})
        flash(f'Successfully removed embeddings for {filename} from Pinecone', 'success')
    except Exception as e:
        app.logger.error(f"Error deleting vectors from Pinecone: {str(e)}")
        flash(f"Error deleting embeddings: {str(e)}", "danger")

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            flash(f'File {filename} deleted successfully', 'success')
        except Exception as e:
            app.logger.error(f"Error deleting local file {filename}: {str(e)}")
            flash(f"Failed to delete file {filename} locally", "danger")
    else:
        flash(f'File {filename} not found', 'danger')

    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)