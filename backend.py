
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
#import os
# from flask import Flask, request, redirect, url_for , render_template 																	
# # from werkzeug.utils import secure_filename

# # UPLOAD_FOLDER = '/path/to/the/uploads'
# # ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


# app = Flask(__name__)
# app.config['SQLAlchemy_DATABASE_URI'] = 'sqlite:'
# db = SQLAlchemy(app)
# class FileContents(db.Model)
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def index():
# 	return render_template('index.html')

# @app.route('/upload' , methods = ['POST'])
# def upload():
# 	file = request.files['inputfile']
# 	return file.filename 

# if __name__ == '__main__':
# 	app.run(debug=True)  