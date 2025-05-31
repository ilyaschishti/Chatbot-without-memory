


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
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from store_index import process_files
from pinecone_setup import initialize_pinecone
from pinecone import Pinecone
import secrets
import hashlib
import os
import shutil
import json
import time
import tempfile

app = Flask(__name__)
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

print("Initializing Pinecone...")
index_name = "university"
pc = initialize_pinecone(index_name=index_name)

embeddings_model = download_hugging_face_embeddings()

try:
    print(f"Attempting to connect to existing index '{index_name}'...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings_model
    )
    print("Successfully connected to Pinecone index!")
except ValueError as e:
    print(f"Error connecting to index: {e}")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


print("Initializing LLM...")
llm = ChatGroq(
    temperature=0.4,
    max_tokens=500,
    model_name="llama3-70b-8192"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

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
    input = msg
    print("===User query===:",input)
    response = rag_chain.invoke({"input": msg})
    print("Respponse printed   ===============", response)
    return str(response["answer"])

# ---------------- REST OF YOUR EXISTING CODE UNCHANGED ----------------




# --- Admin Auth Routes ---
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


from datetime import datetime

# Register a custom Jinja2 filter
@app.template_filter('timestamp_to_datetime')
def timestamp_to_datetime_filter(timestamp):
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return 'Invalid Timestamp'


# --- Admin Dashboard ---
@app.route('/admin', methods=['GET'])
@admin_required
def admin_dashboard():
    files = []
    metadata_dict = get_all_file_metadata()
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        # Skip metadata.json and directories
        if filename == 'metadata.json' or os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            continue
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / 1024
            file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Get processed status from metadata
            processed = False
            for item in metadata_dict:
                if item.get('filename') == filename:
                    processed = item.get('processed', False)
                    break
                    
            files.append({
                'name': filename,
                'size': f"{file_size:.2f} KB",
                'date': file_date.strftime("%Y-%m-%d %H:%M:%S"),
                'processed': processed
            })
    
    return render_template('admin_dashboard.html', files=files, stats={})


# --- File Upload ---
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
        
        # Check if file already exists
        if os.path.exists(file_path):
            flash(f'File {filename} already exists. Please rename your file or delete the existing one.', 'warning')
            return redirect(url_for('admin_dashboard'))
        
        file.save(file_path)

        file_metadata = {
            'filename': filename,
            'processed': False,
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        save_file_metadata(file_metadata)
        
        flash(f'File {filename} uploaded successfully! Click on Process button to add it to the knowledge base.', 'success')
        return redirect(url_for('admin_dashboard'))

    flash('File type not allowed', 'danger')
    return redirect(url_for('admin_dashboard'))



# Add this to your app.py


# # Modify upload function to use temporary directory
# @app.route('/admin/upload', methods=['POST'])
# @admin_required
# def upload_file():
#     if 'file' not in request.files:
#         flash('No file part', 'danger')
#         return redirect(url_for('admin_dashboard'))

#     file = request.files['file']
    
#     if file.filename == '':
#         flash('No selected file', 'danger')
#         return redirect(url_for('admin_dashboard'))

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
        
#         # Use temporary directory instead of persistent storage
#         temp_dir = tempfile.mkdtemp()
#         file_path = os.path.join(temp_dir, filename)
#         file.save(file_path)
        
#         # Process immediately since storage is ephemeral
#         try:
#             result = process_files(data_dir=temp_dir, index_name='university')
#             flash(f'File {filename} processed and added to knowledge base!', 'success')
#         except Exception as e:
#             flash(f'Error processing file: {str(e)}', 'danger')
#         finally:
#             # Clean up
#             shutil.rmtree(temp_dir, ignore_errors=True)
        
#         return redirect(url_for('admin_dashboard'))

# --- Process Single File ---
# Replace the existing process_file function with this updated version
@app.route('/admin/process/<filename>', methods=['POST'])
@admin_required
def process_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Check if file is already processed
    metadata = get_file_metadata(filename)
    if metadata and metadata.get('processed', False):
        flash(f'File {filename} is already processed!', 'info')
        return redirect(url_for('admin_dashboard'))
    
    try:
        # Create a temporary directory for processing
        new_data_folder = os.path.join('Data', 'temp')
        os.makedirs(new_data_folder, exist_ok=True)
        temp_file_path = os.path.join(new_data_folder, filename)
        
        # Copy the file to the temporary directory
        shutil.copy2(file_path, temp_file_path)

        # Call the existing process_files function to handle indexing
        result = process_files(data_dir=new_data_folder, index_name='university')

        # Clean up temporary files
        os.remove(temp_file_path)
        if os.path.exists(new_data_folder) and not os.listdir(new_data_folder):
            os.rmdir(new_data_folder)

        # Check if any documents were processed
        if result and hasattr(result, 'processed_chunks') and result.processed_chunks > 0:
            # Update processed status in metadata
            update_file_processed_status(filename, True)
            flash(f'File {filename} processed and added to knowledge base!', 'success')
        else:
            flash(f'No content could be extracted from {filename}. The file may contain only scanned images or unsupported content.', 'warning')
            
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'danger')

    return redirect(url_for('admin_dashboard'))

# --- Process All Pending Files ---
@app.route('/admin/process-all', methods=['POST'])
@admin_required
def process_all_files():
    unprocessed_files = get_unprocessed_files()
    
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
            # Create a temporary directory for processing
            new_data_folder = os.path.join('Data', 'temp')
            os.makedirs(new_data_folder, exist_ok=True)
            temp_file_path = os.path.join(new_data_folder, filename)
            
            # Copy the file to the temporary directory
            shutil.copy2(file_path, temp_file_path)

            # Call the existing process_files function to handle indexing
            process_files(data_dir=new_data_folder, index_name='university')

            # Clean up temporary files
            os.remove(temp_file_path)
            if os.path.exists(new_data_folder) and not os.listdir(new_data_folder):
                os.rmdir(new_data_folder)
                
            # Update processed status in metadata
            update_file_processed_status(filename, True)
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            app.logger.error(f"Error processing file {filename}: {str(e)}")
    
    if processed_count > 0:
        flash(f'Successfully processed {processed_count} files', 'success')
    if error_count > 0:
        flash(f'Failed to process {error_count} files. Check logs for details.', 'warning')
    
    return redirect(url_for('admin_dashboard'))

# --- Helper functions for file metadata ---
def get_all_file_metadata():
    """Get all file metadata entries"""
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            return all_metadata
        else:
            return []
    except Exception as e:
        app.logger.error(f"Error getting all file metadata: {str(e)}")
        return []

def get_file_metadata(filename):
    """Get metadata for a specific file"""
    all_metadata = get_all_file_metadata()
    
    for item in all_metadata:
        if item.get('filename') == filename:
            return item
            
    return None

def save_file_metadata(metadata):
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = []
        
        for i, item in enumerate(all_metadata):
            if item['filename'] == metadata['filename']:
                all_metadata[i] = metadata
                break
        else:
            all_metadata.append(metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
            
    except Exception as e:
        app.logger.error(f"Error saving file metadata: {str(e)}")

def update_file_processed_status(filename, processed=True):
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = []
        
        updated = False
        for i, item in enumerate(all_metadata):
            if item['filename'] == filename:
                all_metadata[i]['processed'] = processed
                all_metadata[i]['process_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                updated = True
                break
        
        # If the file wasn't found in metadata, add it
        if not updated:
            all_metadata.append({
                'filename': filename,
                'processed': processed,
                'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'process_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
            
    except Exception as e:
        app.logger.error(f"Error updating file processed status: {str(e)}")

def get_unprocessed_files():
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    unprocessed = []
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            unprocessed = [item['filename'] for item in all_metadata if not item.get('processed', False)]
            
    except Exception as e:
        app.logger.error(f"Error getting unprocessed files: {str(e)}")
        
    return unprocessed

# # --- File Deletion ---
# @app.route('/admin/delete/<filename>')
# @admin_required
# def delete_file(filename):
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     if os.path.exists(file_path):
#         os.remove(file_path)
        
#         # Also remove from metadata
#         remove_file_metadata(filename)
        
#         flash(f'File {filename} deleted successfully', 'success')
#     else:
#         flash(f'File {filename} not found', 'danger')
#     return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete/<filename>')
@admin_required
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # --- Delete embeddings from Pinecone ---
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("university")
        
        # Try multiple possible filters since we have different metadata structures
        filters = [
            {"pdf_name": filename},  # New consistent metadata
            {"source": os.path.join(app.config['UPLOAD_FOLDER'], filename)},  # Old style
            {"source_full_path": os.path.join(app.config['UPLOAD_FOLDER'], filename)}  # New field
        ]
        
        deletion_count = 0
        for filter_obj in filters:
            try:
                # Try deleting with each filter
                index.delete(filter=filter_obj)
                print(f"Deleted vectors from Pinecone for file: {filename} with filter {filter_obj}")
                deletion_count += 1
            except Exception as e:
                print(f"Error with filter {filter_obj}: {str(e)}")
                continue
        
        if deletion_count > 0:
            flash(f'Successfully removed embeddings for {filename} from Pinecone', 'success')
        else:
            flash(f'No embeddings found for {filename} in Pinecone', 'warning')

    except Exception as e:
        app.logger.error(f"Error deleting vectors from Pinecone: {str(e)}")
        flash(f"Error deleting embeddings: {str(e)}", "danger")

    # --- Delete file locally and update metadata ---
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            remove_file_metadata(filename)
            flash(f'File {filename} deleted successfully', 'success')
        except Exception as e:
            app.logger.error(f"Error deleting local file {filename}: {str(e)}")
            flash(f"Failed to delete file {filename} locally", "danger")
    else:
        flash(f'File {filename} not found', 'danger')

    return redirect(url_for('admin_dashboard'))

 # --- Remove file metadata ---............................................................................

def remove_file_metadata(filename):
    """Remove a file from metadata.json"""
    metadata_file = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            # Filter out the file to be removed
            all_metadata = [item for item in all_metadata if item.get('filename') != filename]
            
            with open(metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
                
    except Exception as e:
        app.logger.error(f"Error removing file metadata: {str(e)}")

# # === Main Entry Point ===
# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)



# At the end of your app.py file, replace the last section with:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
    