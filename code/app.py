from flask import *
import os
from werkzeug.utils import secure_filename
import label_image
from datetime import datetime

def load_image(image):
    text = label_image.main(image)
    return text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create function to ensure directory exists
def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}")

# Create required directories on startup
@app.before_first_request
def create_directories():
    static_img_path = os.path.join(app.root_path, 'static', 'images')
    ensure_dir_exists(static_img_path)
    print(f"Static images path: {static_img_path}")

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        if f:
            # Create a secure filename and save path
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the uploaded file
            f.save(filepath)
            
            try:
                # Process the image
                k = int(request.form.get('clusters', 3))  # Get number of clusters from form
                plot_cluster_img(filepath, k)
                
                # Generate URLs with timestamp to prevent caching
                timestamp = datetime.now().timestamp()
                return jsonify({
                    'orig_image': url_for('static', filename=f'images/orig_image.jpg?t={timestamp}'),
                    'em_image': url_for('static', filename=f'images/em_image.jpg?t={timestamp}')
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run()