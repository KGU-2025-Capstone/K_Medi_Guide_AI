import re
from bs4 import BeautifulSoup
import numpy as np
import tiktoken


def clean_text(html):
    return BeautifulSoup(html or "", "html.parser").get_text(strip=True).replace("\n", " ").replace("\r", " ")

# 소프트맥스 with temperature
def softmax_with_temperature(weights, temperature):
    weights = np.array(weights)
    scaled = weights / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / np.sum(exp_scaled)

def trim_to_token_limit(text, max_tokens, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_text = enc.decode(tokens[:max_tokens])
    if "." in truncated_text:
        last_period = truncated_text.rfind(".")
        return truncated_text[:last_period+1].strip()
    else:
        return truncated_text.strip()

def detect_language(text):
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ가-힣]", text):
        return "ko"
    elif re.search(r"[A-Za-z]", text):
        return "en"
    elif re.search(r"[\u3040-\u30ff]", text):  # 히라가나 + 가타카나
        return "jp"
    elif re.search(r"[\u4e00-\u9fff]", text):  # 한자 (중국어)
        return "zh"
    else:
        return "ko"  # 기본값은 한국어
