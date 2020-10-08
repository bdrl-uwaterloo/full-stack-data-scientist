@app.route('/')
def homepage():
    return 'Home Page'

@app.route('/mypage')
def mypage():
    return 'My personal page'

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return 'Subpath %s' % subpath

@app.route('/books/')
def books():
    return 'The project page'

@app.route('/book')
def book():
    return 'The about page'

from flask import request

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return upload_file_received()
    else:
        return show_your_webpage()

url_for('static', filename='pagestyle.css')

from flask import render_template

@app.route('/mypage/')
@app.route('/mypage/<name>')
def hello(name=None):
    return render_template('mytemp.html', name=name)

from flask import request

@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid(request.form['username'], request.form['password']):
            return login(request.form['username'])
        else:
            error = 'Invalid username/password'
    return render_template('login.html', error=error)

search = request.args.get('key', '')

from flask import request

@app.route('/file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file_name']
        f.save('file_name')