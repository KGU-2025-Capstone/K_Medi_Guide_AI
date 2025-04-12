from flask import Flask
from routes import symptom, select, detail, name

app = Flask(__name__)
app.register_blueprint(symptom.bp, url_prefix='/medicine')
app.register_blueprint(select.bp, url_prefix='/medicine')
app.register_blueprint(detail.bp, url_prefix='/medicine')
app.register_blueprint(name.bp, url_prefix='/medicine')

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
