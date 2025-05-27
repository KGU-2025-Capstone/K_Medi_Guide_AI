from flask import Flask
from flask_session import Session
from routes import symptom, select, detail, name, start
from config import FLASK_SECRET_KEY
from config import REDIS_HOST
from config import REDIS_PORT
from config import REDIS_PASSWORD
import redis



app = Flask(__name__)

# Flask-Session 설정
app.redis = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses = False
)

app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_SERIALIZER'] = 'json'
app.config['SESSION_ENCODING'] = 'utf-8'
app.config['SESSION_REDIS'] = app.redis  # app.redis 사용
app.config['JSON_AS_ASCII'] = False


# Flask-Session 초기화 (app 객체 생성 직후)
Session(app)

# app.config['PERMANENT_SESSION_LIFETIME'] = 600  #10분동안 세션 유지 (먄약 위 옵션 True 사용 시시)

# 연결 테스트 (선택 사항)
try:
    app.redis.ping()
    print("Redis 연결 성공!")
except redis.exceptions.ConnectionError as e:
    print(f"Redis 연결 실패: {e}")


app.secret_key = FLASK_SECRET_KEY
app.register_blueprint(symptom.bp, url_prefix='/api/medicine')
app.register_blueprint(select.bp, url_prefix='/api/medicine')
app.register_blueprint(detail.bp, url_prefix='/api/medicine')
app.register_blueprint(name.bp, url_prefix='/api/medicine')
app.register_blueprint(start.bp, url_prefix='/api/medicine') 




if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=False)
