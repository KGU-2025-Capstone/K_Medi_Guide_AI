from flask import Flask
from flask_session import Session
from routes import symptom, select, detail, name, start
from config import FLASK_SECRET_KEY
from config import REDIS_HOST
from config import REDIS_PORT
from config import REDIS_PASSWORD
import redis,subprocess,json,os,sys,shutil,glob



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

# docs 폴더에 문서가 있다고 가정
DOCS_DIR = "rag/docs"
CORPUS_DIR = "rag/data/corpus"
DATA_DIR1 = "rag/data/paragraphs"
DATA_DIR2 = "rag/data/summaries"
DATA_DIR3 = "rag/data/clusters"
DATA_DIR4 = "rag/data/corpus"

def run_preprocessing_pipeline():
    # 1. 문서 전처리
    subprocess.run([sys.executable, "rag/preprocess.py"], check=True)
    # 2. 키워드 요약
    subprocess.run([sys.executable, "rag/keyword_summary.py"], check=True)
    # 3. 클러스터링
    subprocess.run([sys.executable, "rag/cluster.py"], check=True)
    # 4. 코퍼스 빌드
    subprocess.run([sys.executable, "rag/corpus.py"], check=True)

def load_corpus():
    if not os.path.exists(CORPUS_DIR) or not os.listdir(CORPUS_DIR):
        print("코퍼스 디렉터리가 없거나 비어 있어요. 파이프라인을 먼저 실행합니다.")
        run_preprocessing_pipeline()

    corpus = []
    for filepath in glob.glob(os.path.join(CORPUS_DIR, '*.json')):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            corpus.extend(data)
    return corpus

if __name__ == '__main__':
    import os

    # data 폴더 내 모든 내용 삭제
    if os.path.exists(DATA_DIR1):
        shutil.rmtree(DATA_DIR1)

    if os.path.exists(DATA_DIR2):
        shutil.rmtree(DATA_DIR2)

    if os.path.exists(DATA_DIR3):
        shutil.rmtree(DATA_DIR3)

    if os.path.exists(DATA_DIR4):
        shutil.rmtree(DATA_DIR4)

    # 빈 data 폴더 새로 생성
    os.makedirs(DATA_DIR1, exist_ok=True)
    os.makedirs(DATA_DIR2, exist_ok=True)
    os.makedirs(DATA_DIR3, exist_ok=True)
    os.makedirs(DATA_DIR4, exist_ok=True)
    corpus = load_corpus()
    app.run('0.0.0.0', port=5000, debug=False)
