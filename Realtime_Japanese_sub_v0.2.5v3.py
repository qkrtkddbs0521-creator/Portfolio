import sys
import os
IS_WIN = sys.platform.startswith("win")
# VLC 설치 경로를 파이썬 모듈 검색 경로 및 DLL 로딩 경로에 추가
# (반드시 `import vlc`보다 먼저 실행되어야 합니다)
VLC_PATH = r"C:\Program Files\VideoLAN\VLC"  # 64비트 VLC 기본 설치 경로

if os.path.exists(VLC_PATH):
    os.add_dll_directory(VLC_PATH)  # Python 3.8 이상 DLL 로드 지원
    os.environ['PYTHON_VLC_MODULE_PATH'] = VLC_PATH
else:
    # 32비트 경로 예외 처리
    VLC_PATH_32 = r"C:\Program Files (x86)\VideoLAN\VLC"
    if os.path.exists(VLC_PATH_32):
        os.add_dll_directory(VLC_PATH_32)
        os.environ['PYTHON_VLC_MODULE_PATH'] = VLC_PATH_32

import json
import math
import gc
import threading
import collections
import time
import unicodedata
from difflib import SequenceMatcher
import torch
import re
import concurrent.futures
# GPU 연산 최적화 옵션 (NVIDIA GPU 사용 시 적용)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if not torch.cuda.is_available():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
import site
os.environ["PATH"] = os.path.dirname(sys.executable) + ";" + os.environ["PATH"]
for site_dir in site.getsitepackages():
    cublas_bin_path = os.path.join(site_dir, "nvidia", "cublas", "bin")
    cudnn_bin_path = os.path.join(site_dir, "nvidia", "cudnn", "bin")

    if os.path.exists(cublas_bin_path):
        os.add_dll_directory(cublas_bin_path)
        os.environ["PATH"] = cublas_bin_path + ";" + os.environ["PATH"]
    if os.path.exists(cudnn_bin_path):
        os.add_dll_directory(cudnn_bin_path)
        os.environ["PATH"] = cudnn_bin_path + ";" + os.environ["PATH"]

from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMenu, QAction,
                             QFrame, QSlider, QColorDialog, QFontComboBox, QSpinBox,
                             QWidgetAction, QComboBox, QListWidget, QListWidgetItem,
                             QCheckBox, QAbstractItemView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QRectF, QPoint, QEvent, QSize, QSettings
from PyQt5.QtGui import (QFont, QCursor, QColor, QFontDatabase, QIcon, QPixmap,
                         QPainter, QPen, QPainterPath, QRegion)
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
from google import genai


# python-vlc는 반드시 위 DLL 경로 등록 이후에 import
# --no-sub-autodetect-file: 같은 폴더의 .srt를 VLC가 자동으로 찾아 원문
# 그대로 하단에 띄우는 기능을 꺼서, 우리가 만든 발음/뜻 오버레이와
# 중복 표시되지 않게 합니다.
VLC_HW = os.getenv("VLC_HW", "dxva2")   # 계속 깨지면 "none"으로
VLC_INSTANCE_ARGS = ['--no-sub-autodetect-file', f'--avcodec-hw={VLC_HW}', '--quiet']
import vlc

import demucs.api
try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False

GEN_CONFIG = {"response_mime_type": "application/json", "temperature": 0.2}

# --- Gemini API 설정 ---
# 실제 키는 반드시 환경변수로만 주입하세요. 코드에 직접 적지 마세요.


def load_api_keys():
    """환경변수(있으면 최우선) + 저장된 키 파일을 합쳐서 중복 없이 반환."""
    keys = []
    for i in range(1, 11):
        v = os.getenv(f"GEMINI_API_KEY_{i}")
        if v:
            keys.append(v.strip())
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, list):
            keys += [k.strip() for k in saved if isinstance(k, str) and k.strip()]
    except Exception:
        pass
    seen, result = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result

def _env_api_keys():
    keys = []
    for i in range(1, 11):
        v = os.getenv(f"GEMINI_API_KEY_{i}")
        if v:
            keys.append(v.strip())
    return keys

GEMINI_API_KEYS = _env_api_keys()   # 저장된 키는 플레이어가 load_settings()에서 합쳐 넣습니다.
if not GEMINI_API_KEYS:
    print("경고: 등록된 Gemini API 키가 없습니다. 우클릭 메뉴 → 🔑 API 키 관리에서 등록하세요.")

def set_api_keys(new_keys):
    """새 키 목록을 전역 리스트에 반영 (재바인딩 아님 → 다른 곳 참조도 동기화)."""
    seen, merged = set(), []
    for k in _env_api_keys() + list(new_keys):
        if k and k not in seen:
            seen.add(k)
            merged.append(k)
    GEMINI_API_KEYS[:] = merged



# gemini-2.5-flash는 2026-10-16 서비스 종료 예정이라 3.5로 통일합니다.
GEMINI_MODEL_NAME = "gemini-3.5-flash"
current_key_index = 0

def gemini_json(api_key, prompt, json_mode=True):
    """스레드마다 독립 Client → 전역 키 충돌 없음"""
    client = genai.Client(api_key=api_key)
    cfg = {"temperature": 0.2}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    r = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt, config=cfg)
    return (r.text or "").strip()

def parse_json_array(raw_text):
    """Gemini 응답에서 JSON 배열만 뽑아 파싱. 코드펜스나 설명이 섞여도 최대한 복구."""
    if not raw_text:
        return []
    text = re.sub(r'```json\s*|```', '', raw_text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


_GEMINI_KEY_COOLDOWNS = {}   # api_key -> 이 시각(monotonic) 전에는 다시 쓰지 않음
_GEMINI_COOLDOWN_LOCK = threading.Lock()
_RPM_LIMIT = int(os.getenv("GEMINI_RPM_PER_KEY", "10"))  # 키 1개당 분당 허용 요청 수 (무료 티어 기준 보수적으로)
_RPM_LOCK = threading.Lock()
_RPM_WINDOWS = collections.defaultdict(list)  # api_key -> 최근 호출 예정 시각들

def _throttle_for_key(key):
    """이 키가 최근 60초 안에 이미 한도만큼 호출했다면, 다음 호출 가능 시각까지 대기시킨다."""
    with _RPM_LOCK:
        now = time.monotonic()
        window = _RPM_WINDOWS[key]
        window[:] = [t for t in window if now - t < 60]
        wait = (60 - (now - window[0]) + 0.1) if len(window) >= _RPM_LIMIT else 0
        window.append(now + wait)
    if wait > 0:
        time.sleep(wait)

def gemini_call_with_backoff(prompt, preferred_key, max_wait=20, _already_waited=False):
    """429(쿼터 초과)를 만나면 그 키를 잠깐 쉬게 하고 다른 키로 자동 전환한다.
    모든 키가 쉬는 중이면 가장 빨리 풀리는 키까지 최대 max_wait초 기다렸다가 한 번 더 시도."""
    keys = GEMINI_API_KEYS
    if not keys:
        return ""
    order = [preferred_key] + [k for k in keys if k != preferred_key]
    now = time.monotonic()
    any_available = False
    for key in order:
        with _GEMINI_COOLDOWN_LOCK:
            cooldown_until = _GEMINI_KEY_COOLDOWNS.get(key, 0)
        if cooldown_until > now:
            continue
        any_available = True
        _throttle_for_key(key)
        try:
            return gemini_json(key, prompt)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", msg)
                delay = int(m.group(1)) if m else 30
                with _GEMINI_COOLDOWN_LOCK:
                    _GEMINI_KEY_COOLDOWNS[key] = time.monotonic() + delay
                print(f"[Gemini] 키 쿼터 초과 — {delay}초 쉬게 함")
            else:
                print(f"[Gemini 실패] {type(e).__name__}: {e}")
            continue
    if not any_available and not _already_waited and _GEMINI_KEY_COOLDOWNS:
        with _GEMINI_COOLDOWN_LOCK:
            soonest = min(_GEMINI_KEY_COOLDOWNS.values())
        wait = max(0.0, min(max_wait, soonest - time.monotonic()))
        if wait > 0:
            time.sleep(wait)
            return gemini_call_with_backoff(prompt, preferred_key, max_wait=max_wait, _already_waited=True)
    return ""

def get_gemini_translation(japanese_text):
    global current_key_index
    if not GEMINI_API_KEYS:
        return "번역 실패 (API 키 없음)"

    prompt = f"""다음은 일본어 애니메이션 대사입니다. 
애니메이션의 맥락과 감정, 자연스러운 구어체 뉘앙스를 살려서 가장 자연스러운 한국어 문장으로 번역해 주세요. 
오직 번역된 한국어 결과만 출력하세요.

일본어: {japanese_text}"""

    for _ in range(len(GEMINI_API_KEYS)):
        try:
            api_key = GEMINI_API_KEYS[current_key_index]
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[Gemini 실패] key#{current_key_index} {type(e).__name__}: {e}")
            current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
            continue
    return "번역 실패"


def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def ms_to_mmss(ms):
    if ms is None or ms < 0:
        ms = 0
    total_sec = ms // 1000
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}"


# ---------------- 일본어 → 한글 발음 표기 ----------------
# 1) 일본어 문장 → 가나 읽기
#    - 형태소 분석기 fugashi(+unidic-lite)가 설치돼 있으면: 조사 は/へ 를 わ/え 로 정확히 읽고,
#      こんにちは 같은 굳은 표현, 단어 단위 띄어쓰기까지 처리한다.
#    - 없으면: pykakasi + 규칙 기반 보정 (は/へ 조사를 최대한 추정)
# 2) 가나 → 한글 (작은 글자 ゃゅょ / 촉음 っ / ん / 장음 ー / 외래음 ファ·ティ 등 전부 한글로)
#    변환 결과에는 일본어 문자가 남지 않는다 (한자 읽기를 못 찾은 경우 제외).

_HANGUL_BASE = 0xAC00
_JONG_INDEX = {'ㄴ': 4, 'ㅁ': 16, 'ㅅ': 19, 'ㅇ': 21}

# 카타카나 기준 (히라가나는 변환 전에 카타카나로 바꿔서 처리)
_KANA_TO_HANGUL = {
    'ア': '아', 'イ': '이', 'ウ': '우', 'エ': '에', 'オ': '오',
    'カ': '카', 'キ': '키', 'ク': '쿠', 'ケ': '케', 'コ': '코',
    'ガ': '가', 'ギ': '기', 'グ': '구', 'ゲ': '게', 'ゴ': '고',
    'サ': '사', 'シ': '시', 'ス': '스', 'セ': '세', 'ソ': '소',
    'ザ': '자', 'ジ': '지', 'ズ': '즈', 'ゼ': '제', 'ゾ': '조',
    'タ': '타', 'チ': '치', 'ツ': '츠', 'テ': '테', 'ト': '토',
    'ダ': '다', 'ヂ': '지', 'ヅ': '즈', 'デ': '데', 'ド': '도',
    'ナ': '나', 'ニ': '니', 'ヌ': '누', 'ネ': '네', 'ノ': '노',
    'ハ': '하', 'ヒ': '히', 'フ': '후', 'ヘ': '헤', 'ホ': '호',
    'バ': '바', 'ビ': '비', 'ブ': '부', 'ベ': '베', 'ボ': '보',
    'パ': '파', 'ピ': '피', 'プ': '푸', 'ペ': '페', 'ポ': '포',
    'マ': '마', 'ミ': '미', 'ム': '무', 'メ': '메', 'モ': '모',
    'ヤ': '야', 'ユ': '유', 'ヨ': '요',
    'ラ': '라', 'リ': '리', 'ル': '루', 'レ': '레', 'ロ': '로',
    'ワ': '와', 'ヰ': '이', 'ヱ': '에', 'ヲ': '오', 'ヴ': '브',
    'ヵ': '카', 'ヶ': '케', 'ヷ': '바', 'ヸ': '비', 'ヹ': '베', 'ヺ': '보',
}

# 요음 (き+ゃ 등): 한국어에서는 ㅈ/ㅊ 뒤에 ㅑㅠㅛ가 안 쓰이므로 じゃ→자, ちゃ→차
_PALATAL = {
    'キ': '캬큐쿄', 'ギ': '갸규교', 'シ': '샤슈쇼', 'ジ': '자주조', 'ヂ': '자주조',
    'チ': '차추초', 'ニ': '냐뉴뇨', 'ヒ': '햐휴효', 'ビ': '뱌뷰뵤', 'ピ': '퍄퓨표',
    'ミ': '먀뮤묘', 'リ': '랴류료',
}

# 외래어 표기용 조합 (ファ, ティ, ウィ ...)
_FOREIGN = {
    'イェ': '예', 'ウィ': '위', 'ウェ': '웨', 'ウォ': '워', 'ウァ': '와',
    'ヴァ': '바', 'ヴィ': '비', 'ヴェ': '베', 'ヴォ': '보', 'ヴュ': '뷰',
    'クァ': '콰', 'クィ': '퀴', 'クェ': '퀘', 'クォ': '쿼',
    'グァ': '과', 'グィ': '귀', 'グェ': '궤', 'グォ': '궈',
    'シェ': '셰', 'ジェ': '제', 'チェ': '체',
    'ツァ': '차', 'ツィ': '치', 'ツェ': '체', 'ツォ': '초',
    'ティ': '티', 'トゥ': '투', 'ディ': '디', 'ドゥ': '두', 'テュ': '튜', 'デュ': '듀',
    'ファ': '파', 'フィ': '피', 'フェ': '페', 'フォ': '포', 'フュ': '퓨',
    'スィ': '시', 'ズィ': '지', 'ニェ': '녜',
}

_COMBO = dict(_FOREIGN)
for _base, _syls in _PALATAL.items():
    for _k, _small in enumerate('ャュョ'):
        _COMBO[_base + _small] = _syls[_k]

# 작은 가나가 만드는 모음 (한글 중성 인덱스)
_SMALL_VOWEL_IDX = {'ァ': 0, 'ィ': 20, 'ゥ': 13, 'ェ': 5, 'ォ': 8,
                    'ャ': 2, 'ュ': 17, 'ョ': 12, 'ヮ': 9}
# 작은 가나가 단독으로 나온 경우
_SMALL_ALONE = {'ァ': '아', 'ィ': '이', 'ゥ': '우', 'ェ': '에', 'ォ': '오',
                'ャ': '야', 'ュ': '유', 'ョ': '요', 'ヮ': '와'}

# 장음(ー)을 만났을 때 앞 글자의 모음을 다시 한 번 적기 위한 표 (중성 인덱스 → 글자)
_LONG_VOWEL_CHAR = {0: '아', 1: '애', 2: '아', 3: '애', 4: '어', 5: '에', 6: '어', 7: '에',
                    8: '오', 9: '아', 10: '애', 11: '외', 12: '오', 13: '우', 14: '오',
                    15: '에', 16: '이', 17: '우', 18: '으', 19: '이', 20: '이'}

# 일본어 문장부호 → 일반 문장부호
_PUNCT_MAP = {'。': '.', '、': ',', '「': '"', '」': '"', '『': '"', '』': '"',
              '・': ' ', '〜': '~', '～': '~', '　': ' ', '♪': '♪'}

LONG_VOWEL_STYLE = "repeat"   # "repeat"=코오히이 / "dash"=코-히-


def _to_katakana(s):
    return ''.join(chr(ord(c) + 0x60) if '\u3041' <= c <= '\u3096' else c for c in s)


def _is_syllable(ch):
    return '\uAC00' <= ch <= '\uD7A3'


def _add_jongseong(out, jong):
    """마지막 글자에 받침(ㄴ/ㅅ...)을 붙인다. 붙일 수 있는 글자가 아니면 False."""
    if not out:
        return False
    ch = out[-1]
    if len(ch) == 1 and _is_syllable(ch) and (ord(ch) - _HANGUL_BASE) % 28 == 0:
        out[-1] = chr(ord(ch) + _JONG_INDEX[jong])
        return True
    return False


def _replace_vowel(syllable, vowel_idx):
    """'쿠' + ァ(ㅏ) → '카' 처럼 한글 글자의 모음만 바꾼다."""
    code = ord(syllable) - _HANGUL_BASE
    lead = code // (21 * 28)
    return chr(_HANGUL_BASE + (lead * 21 + vowel_idx) * 28)


def _last_vowel_char(ch):
    if _is_syllable(ch) and (ord(ch) - _HANGUL_BASE) % 28 == 0:
        return _LONG_VOWEL_CHAR.get(((ord(ch) - _HANGUL_BASE) // 28) % 21)
    return None

def kana_to_hangul(text):
    """가나(히라가나/카타카나 혼합) 문자열을 한글 발음 표기로 바꾼다."""
    style = LONG_VOWEL_STYLE
    
    text = _to_katakana(unicodedata.normalize('NFKC', text or ''))
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ''

        if c == 'ッ':                      # 촉음: 앞 글자의 받침 ㅅ (ちょっと → 촛토)
            _add_jongseong(out, 'ㅅ')
            i += 1
            continue
        if c == 'ン':                      # 발음(ん): 앞 글자의 받침 ㄴ (こんにちは → 곤니치와)
            if not _add_jongseong(out, 'ㄴ'):
                out.append('응')
            i += 1
            continue
        if c == 'ー':
            if LONG_VOWEL_STYLE == "dash":
                out.append('-')
            else:
                vowel = _last_vowel_char(out[-1]) if out else None
                if vowel:
                    out.append(vowel)
            i += 1
            continue

        if nxt in _SMALL_VOWEL_IDX and c in _KANA_TO_HANGUL:
            syl = _COMBO.get(c + nxt)
            if syl is None:
                syl = _replace_vowel(_KANA_TO_HANGUL[c], _SMALL_VOWEL_IDX[nxt])
            out.append(syl)
            i += 2
            continue

        if c in _KANA_TO_HANGUL:
            out.append(_KANA_TO_HANGUL[c])
        elif c in _SMALL_ALONE:
            out.append(_SMALL_ALONE[c])
        else:
            out.append(_PUNCT_MAP.get(c, c))
        i += 1
    return ''.join(out)


# ---------- 일본어 문장 → 가나 읽기 ----------
_TAGGER = None
_TAGGER_FAILED = False
_TAGGER_LOCK = threading.Lock()   # MeCab 태거는 스레드 안전하지 않으므로 잠금 사용
_KAKASI = None

_ATTACH_POS = {'助詞', '助動詞', '接尾辞', '接尾'}   # 앞 단어에 붙여 쓸 품사
_PREFIX_POS = {'接頭辞', '接頭詞'}                    # 뒤 단어에 붙여 쓸 품사
_PUNCT_POS = {'補助記号', '記号', '空白'}


def _get_tagger():
    global _TAGGER, _TAGGER_FAILED
    if _TAGGER is None and not _TAGGER_FAILED:
        try:
            from fugashi import Tagger
            _TAGGER = Tagger()
        except Exception:
            _TAGGER_FAILED = True   # 미설치 → pykakasi 방식으로 대체
            print("[안내] fugashi 미설치: 발음 표기의 は/へ 조사 판별이 규칙 기반(근사)으로 동작합니다. "
                  "정확도를 높이려면: pip install fugashi unidic-lite")
    return _TAGGER


def _clean_reading(v):
    v = (v or '').strip() if isinstance(v, str) else ''
    return None if v in ('', '*') else _to_katakana(v)


def _restore_long_vowels(pron, faithful):
    """pron(실제 발음, 장음 ー 사용)에서 ー 자리를 faithful(표기 그대로의 읽기)의
    글자(우/이/오...)로 되돌린다. 나머지 글자가 서로 일치할 때만 적용."""
    if 'ー' not in pron or not faithful or len(faithful) < len(pron):
        return pron
    out = []
    for i, p in enumerate(pron):
        f = faithful[i]
        if p == 'ー':
            if f not in 'アイウエオー':
                return pron
            out.append(f)
        elif p == f:
            out.append(p)
        else:
            return pron
    return ''.join(out)


def _first_reading(feature, names):
    """feature에서 names 순서대로 찾아 값이 있는('*' 아님) 첫 읽기를 카타카나로 반환."""
    for name in names:
        v = _clean_reading(getattr(feature, name, None))
        if v:
            return v
    return None


def _token_reading(feature, surface):
    # pron: 실제 발음 (조사 は→ワ, へ→エ 가 이미 반영됨).  faithful: 표기 그대로의 읽기 (おう→オウ)
    pron = _first_reading(feature, ('pron', 'pronunciation'))
    faithful = _first_reading(feature, ('kana', 'reading', 'lForm'))
    if pron:
        return _restore_long_vowels(pron, faithful)
    if faithful:
        return faithful
    return surface


def _reading_with_fugashi(tagger, text):
    chunks = []
    attach_next = False
    force_new = False
    for w in tagger(text):
        surface = w.surface
        feature = w.feature
        pos1 = getattr(feature, 'pos1', '') or ''
        if pos1 in _PUNCT_POS or not surface.strip():
            if surface.strip():
                if chunks:
                    chunks[-1] += surface
                else:
                    chunks.append(surface)
            force_new = True
            attach_next = False
            continue
        reading = _token_reading(feature, surface)
        if chunks and not force_new and (pos1 in _ATTACH_POS or attach_next):
            chunks[-1] += reading
        else:
            chunks.append(reading)
        attach_next = pos1 in _PREFIX_POS
        force_new = False
    return ' '.join(chunks)


_KANA_RUN = re.compile(r'^[\u3041-\u309F]+$')
_HA_WORDS = ('これ|それ|あれ|どれ|ここ|そこ|あそこ|どこ|だれ|なに|なん|あなた|わたし|あたし|'
             'ぼく|おれ|きみ|うち|みんな')
_HA_PATTERNS = [
    (re.compile(r'(こんにち|こんばん)は'), r'\1わ'),
    (re.compile(r'(' + _HA_WORDS + r')は'), r'\1わ'),
    (re.compile(r'(で|に|と|へ|から|まで|だけ|より|しか)は'), r'\1わ'),
]
_ATTACH_ORIG = {'は', 'が', 'を', 'に', 'へ', 'で', 'と', 'も', 'の', 'や', 'か', 'ね', 'よ',
                'な', 'さ', 'わ', 'です', 'ます', 'でした', 'ました', 'だ', 'た', 'て'}


def _fix_particles_heuristic(orig, hira, idx):
    """pykakasi 결과(hira)에서 조사 は→わ, へ→え 를 규칙으로 추정해 고친다."""
    if orig == 'は' and idx > 0:
        return 'わ'
    if orig == 'へ' and idx > 0:
        return 'え'
    if not _KANA_RUN.match(orig) or orig != hira:
        return hira
    s = hira
    for pattern, repl in _HA_PATTERNS:
        s = pattern.sub(repl, s)
    # 히라가나 덩어리가 は/へ 로 끝나면 조사로 본다 (ははは 같은 웃음소리는 제외)
    if len(s) >= 2 and s.endswith('は') and s.strip('はあひふへほうえお'):
        s = s[:-1] + 'わ'
    elif len(s) >= 2 and s.endswith('へ'):
        s = s[:-1] + 'え'
    return s


def _reading_with_pykakasi(text):
    global _KAKASI
    try:
        if _KAKASI is None:
            import pykakasi
            _KAKASI = pykakasi.kakasi()
        items = _KAKASI.convert(text)
    except Exception:
        # pykakasi도 없으면 한자 읽기는 불가능하지만, 가나 부분의 は/へ 조사 보정은 그대로 적용
        items = [{'orig': run, 'hira': run}
                 for run in re.findall(r'[\u3041-\u309F]+|[^\u3041-\u309F]+', text)]
    chunks = []
    for idx, item in enumerate(items):
        orig = item.get('orig', '')
        hira = item.get('hira') or orig
        if not orig.strip():
            continue
        reading = _fix_particles_heuristic(orig, hira, idx)
        if chunks and orig in _ATTACH_ORIG:
            chunks[-1] += reading
        else:
            chunks.append(reading)
    return ' '.join(chunks)


def japanese_to_reading(text):
    """일본어 문장 → 가나 읽기 (조사 は/へ 는 わ/え 로, 띄어쓰기 포함)."""
    text = unicodedata.normalize('NFKC', text or '')
    tagger = _get_tagger()
    if tagger is not None:
        try:
            with _TAGGER_LOCK:
                return _reading_with_fugashi(tagger, text)
        except Exception as e:
            print(f"형태소 분석 실패, pykakasi로 대체합니다: {e}")
    return _reading_with_pykakasi(text)


def japanese_to_korean_phonetic(text):
    try:
        reading = japanese_to_reading(text)
    except Exception:
        reading = text
    hangul = kana_to_hangul(reading)
    hangul = re.sub(r'[ \t]+', ' ', hangul)
    hangul = re.sub(r'\s+([.,!?])', r'\1', hangul)
    return hangul.strip()


# 1. 백그라운드 자막 생성 스레드 (대기열 방식)
class SubtitleQueueWorker(QThread):
    """자막 생성 대기열을 '순서대로' 처리하는 백그라운드 스레드.

    - 영상을 보는 동안에도 계속 돌아가고, 영상을 여러 개 대기열에 넣어두면 하나씩 처리한다.
    - Whisper / Demucs 모델은 작업 사이에 재사용한다 (영상마다 다시 로딩하지 않음).
      대기열이 비면 GPU 메모리를 돌려주기 위해 모델을 해제한다.
    - 번역·STT 로직 자체는 기존 WhisperWorker와 동일하다.

    signals
      job_started(path) / job_progress(path, text) / job_finished(path, segments)
      job_failed(path, message) / queue_idle()
    """
    job_started = pyqtSignal(str)
    job_progress = pyqtSignal(str, str)
    job_finished = pyqtSignal(str, list)
    job_failed = pyqtSignal(str, str)
    queue_idle = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._cond = threading.Condition()
        self._jobs = collections.deque()
        self._stop_requested = False
        self._cancel_current = False
        self.current_path = ""
        self._whisper_model = None
        self._separator = None
        self._meta_hint = ""

    # ---------------- 대기열 조작 (메인 스레드에서 호출) ----------------
    def enqueue(self, path, front=False):
        with self._cond:
            if path == self.current_path or path in self._jobs:
                return False
            if front:
                self._jobs.appendleft(path)
            else:
                self._jobs.append(path)
            self._cond.notify()
        return True

    def remove_pending(self, path):
        with self._cond:
            try:
                self._jobs.remove(path)
                return True
            except ValueError:
                return False

    def clear_pending(self):
        with self._cond:
            removed = list(self._jobs)
            self._jobs.clear()
        return removed

    def pending_paths(self):
        with self._cond:
            return list(self._jobs)

    def cancel_current(self):
        self._cancel_current = True

    def stop(self):
        with self._cond:
            self._stop_requested = True
            self._cancel_current = True
            self._jobs.clear()
            self._cond.notify()

    # ---------------- 내부 ----------------
    def _progress(self, path, text):
        self.job_progress.emit(path, text)

    def _release_models(self):
        self._whisper_model = None
        self._separator = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def extract_clean_vocals(self, path):
        """Demucs를 이용해 보컬 음성만 정교하게 분리"""
        if not HAS_DEMUCS:
            return path
        try:
            if self._separator is None:
                self._progress(path, "1/5단계: 음성 정제용 Demucs AI 모델 로딩 중...")
                self._separator = demucs.api.Separator(
                    model="htdemucs",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            separator = self._separator

            self._progress(path, "1/5단계: 영상 오디오 데이터 읽기 및 BGM 제거 중...")
            origin, separated = separator.separate_audio_file(path)
            vocals = separated["vocals"]

            clean_wav_path = path + "_vocals.wav"
            demucs.api.save_audio(vocals, clean_wav_path, samplerate=separator.samplerate)
            return clean_wav_path
        except Exception as e:
            print(f"음성 분리 실패, 원본 영상을 사용합니다: {e}")
            return path

    def process_batch(self, batch_data):
        # batch_data = (index_map, batch_chunk, assigned_api_key, ctx, glossary)
        # glossary는 이제 _process_job에서 영상 하나당 딱 한 번만 계산해서
        # 넘겨받는다 (배치마다 매번 교정 사전을 다시 읽고 정렬하던 낭비 제거).
        index_map, batch_chunk, assigned_api_key, ctx, glossary = batch_data
        ctx_block = ("\n[참고용 직전 대사 — 번역하지 말 것]\n" + "\n".join(ctx) + "\n") if ctx else ""

        batch_text_lines = [f"[{idx}] {seg['text']}" for idx, seg in enumerate(batch_chunk)]
        combined_text = "\n".join(batch_text_lines)

        batch_prompt = f"""다음은 일본어 애니메이션 대사 목록이야. 각 대사를 자연스러운 한국어 구어체로 번역하고, 문장에 쓰인 주요 단어 3~5개의 원형(기본형)과 한국어 뜻을 함께 정리해줘.
        
[엄격한 제약 사항]
1. 번역 결과에 일본어와 한국어가 절대 뒤섞이지 않게 온전한 한국어로만 번역해줘.
2. 반드시 아래의 JSON 배열 형태로만 결과를 반환해줘. 다른 설명이나 사족은 절대 금지.
3. 목록에 있는 대사는 하나도 빠짐없이 전부({len(batch_chunk)}개) 번역해줘.

출력 JSON 구조 예시:
[
  {{
    "index": 0,
    "translation": "번역된 한국어 문장",
    "vocabulary": [
      {{"word": "단어원형1", "meaning": "뜻1"}},
      {{"word": "단어원형2", "meaning": "뜻2"}}
    ]
  }}
]
{glossary}{ctx_block}
번역할 대사 목록:
{combined_text}"""

        if index_map and index_map[0] == 0:
            print("----- 프롬프트 확인 -----\n", batch_prompt[:500])

        translated_raw = gemini_call_with_backoff(batch_prompt, assigned_api_key)

        results = []
        try:
            data_list = parse_json_array(translated_raw)

            # translated_raw가 아예 빈 문자열이면 API 호출 자체가 실패한 것
            # (거의 대부분 쿼터 초과)이다. 이 경우에 쪼개서 재요청하면 이미
            # 막혀 있는 키를 또 두드리는 꼴이라 과부하를 더 키운다. 실제로
            # 응답은 받았는데 JSON 형식이 이상해서 파싱만 실패한 경우에만
            # 쪼개서 재시도한다.
            if not data_list and translated_raw and len(batch_chunk) > 8:
                half = len(batch_chunk) // 2
                results += self.process_batch((index_map[:half], batch_chunk[:half], assigned_api_key, ctx, glossary))
                results += self.process_batch((index_map[half:], batch_chunk[half:], assigned_api_key, ctx, glossary))
                return results

            for item in data_list:
                rel_idx = item.get("index")
                if rel_idx is not None and 0 <= rel_idx < len(index_map):
                    results.append((index_map[rel_idx], item.get("translation", ""), item.get("vocabulary", [])))

            # [변경] 빠진 줄이 있어도 여기서 즉시 재요청하지 않는다 (배치 수만큼
            # 호출이 배로 늘어나는 원인이었음). 대신 _process_job이 모든 배치가
            # 끝난 뒤 영상 전체 기준으로 빠진 줄만 한 번에 모아 재요청한다.
            return results
        except Exception as parse_err:
            print(f"JSON 파싱 오류 (idx {index_map[:1]}...): {parse_err}")
            return results

    def _rebuild_from_srt(self, path, srt_path, json_path):
        """SRT만 있는 영상: 재생할 때와 똑같이 기존 SRT를 바탕으로 JSON을 만든다
        (사용자가 가진 SRT를 Whisper 결과로 덮어쓰지 않기 위함)."""
        rebuilder = SrtRebuilderWorker(srt_path, json_path)
        result = []
        rebuilder.progress.connect(lambda t: self._progress(path, t))
        rebuilder.finished.connect(lambda segs: result.extend(segs))
        rebuilder.run()   # 이 스레드 안에서 그대로 실행 (별도 스레드를 또 만들지 않음)
        return result

    def _process_job(self, path):
        """영상 1개 처리. 성공 시 segments 리스트, 취소되면 None."""
        srt_path = os.path.splitext(path)[0] + ".srt"
        meta = {}
        meta_path = os.path.splitext(path)[0] + ".meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
        self._meta_hint = ""
        if meta:
            self._meta_hint = (f"\n[작품 정보] 제목: {meta.get('title','')} / "
                               f"등장인물: {', '.join(meta.get('names', []))}\n")
        json_path = path + ".json"
        if os.path.exists(srt_path) and not os.path.exists(json_path):
            return self._rebuild_from_srt(path, srt_path, json_path)
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))   # 토큰 있으면 .env로 주입

        from faster_whisper import WhisperModel

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        clean_audio_path = self.extract_clean_vocals(path)
        raw_segments = []
        try:
            if self._cancel_current:
                return None

            if self._whisper_model is None:
                self._progress(path, "2/5단계: AI Whisper 모델 로딩 중...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                self._whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type)
            model = self._whisper_model

            self._progress(path, "3/5단계: 일본어 애니메이션 정밀 STT 분석 중...")

            segments_iter, info = model.transcribe(
                clean_audio_path,
                language="ja",
                beam_size=10,
                initial_prompt=("アニメの日本語セリフ、日常会話、正確な文字起こし。"
                                + " ".join(meta.get("names", []))),
                no_speech_threshold=0.6,          # 0.4 → 0.6: 무음으로 오판해 건너뛰는 대사를 줄임
                log_prob_threshold=-1.0,          # 확신 낮은 결과도 일단 살려서 "누락"보다 "오탈자"가 되게
                compression_ratio_threshold=2.6,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 150},
                word_timestamps=True
            )

            total_duration = info.duration

            for segment in segments_iter:
                if self._cancel_current:
                    return None

                current_time = segment.end
                text = segment.text.strip()

                if not text:
                    continue

                if total_duration > 0:
                    percent = int((current_time / total_duration) * 100)
                    if percent > 100:
                        percent = 100
                    self._progress(path, f"3/5단계: STT 텍스트 추출 중... ({percent}% 완료)")

                phonetic = japanese_to_korean_phonetic(text)

                words = [{'w': w.word, 's': w.start, 'e': w.end} for w in (segment.words or [])]
                seg_start, seg_end = segment.start, segment.end
                if words:
                    seg_start = max(seg_start, words[0]['s'] - 0.05)   # 첫 단어 시작 기준 (스포일러 방지)
                    seg_end = min(seg_end + 0.05, words[-1]['e'] + 0.15)  # 마지막 음절 안 잘리게 살짝 여유

                raw_segments.append({
                    'start': seg_start,
                    'end': seg_end,
                    'text': text,
                    'phonetic': phonetic,
                    'words': words
                })
        finally:
            if clean_audio_path.endswith("_vocals.wav") and os.path.exists(clean_audio_path):
                try:
                    os.remove(clean_audio_path)
                except Exception:
                    pass

        total_segs = len(raw_segments)
        segments = []

        if total_segs > 0:
            translation_map = {}
            vocab_map = {}

            # 이미 학습(사용자 교정 또는 과거 자동 캐시)된 대사는 API를 다시 부르지 않고 재사용한다.
            def _norm_key(t):
                return re.sub(r'[。、！？\s]+', '', t or '')
                
            corr_phrases_raw = load_corrections().get("phrases", {})
            corr_phrases = {_norm_key(k): v for k, v in corr_phrases_raw.items()}
            cached_indices = set()
            for idx, seg in enumerate(raw_segments):
                entry = corr_phrases.get(_norm_key(seg['text']))
                if entry and entry.get("meaning"):
                    translation_map[idx] = entry["meaning"]
                    vocab_map[idx] = entry.get("vocabulary", [])
                    if entry.get("phonetic"):
                        seg['phonetic'] = entry["phonetic"]
                    cached_indices.add(idx)

            to_translate = [(idx, seg) for idx, seg in enumerate(raw_segments) if idx not in cached_indices]

            # [변경] 교정 사전 글로서리는 영상 하나당 한 번만 계산한다
            # (이전에는 배치마다 매번 파일을 읽고 정렬해서 배치 수만큼 반복 계산했음).
            corr_for_glossary = load_corrections().get("phrases", {})
            top_corrections = sorted(
                corr_for_glossary.items(), key=lambda kv: -kv[1].get("count", 0))[:15]
            glossary = ""
            if top_corrections:
                gl_lines = [f'- "{jp}" → "{e.get("meaning","")}"' for jp, e in top_corrections]
                glossary = ("\n[사용자가 직접 교정한 번역 예시 — 같은 표현은 반드시 이대로 번역]\n"
                            + "\n".join(gl_lines) + "\n")

            BATCH_SIZE = 80
            num_keys = max(len(GEMINI_API_KEYS), 1)

            def build_batches(items):
                """(idx, seg) 목록을 BATCH_SIZE 단위로 쪼개 process_batch용 튜플 리스트로 만든다."""
                result = []
                bi = 0
                for i in range(0, len(items), BATCH_SIZE):
                    chunk = items[i:i + BATCH_SIZE]
                    index_map = [idx for idx, _ in chunk]
                    chunk_segs = [seg for _, seg in chunk]
                    assigned_key = GEMINI_API_KEYS[bi % num_keys] if GEMINI_API_KEYS else ""
                    ctx_start = index_map[0]
                    ctx = [s['text'] for s in raw_segments[max(0, ctx_start - 3):ctx_start]]
                    result.append((index_map, chunk_segs, assigned_key, ctx, glossary))
                    bi += 1
                return result

            batches = build_batches(to_translate)

            total_batches = len(batches)
            worker_count = max(min(total_batches, num_keys), 1)
            completed_batches = 0

            if total_batches > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_to_batch = {
                        executor.submit(self.process_batch, batch): batch
                        for batch in batches
                    }

                    for future in concurrent.futures.as_completed(future_to_batch):
                        if self._cancel_current:
                            for f in future_to_batch:
                                f.cancel()
                            return None

                        completed_batches += 1
                        percent = int((completed_batches / total_batches) * 100)
                        self._progress(path, f"4/5단계: AI 번역 중... ({percent}%, "
                                             f"캐시 재사용 {len(cached_indices)}개 제외)")

                        try:
                            batch_results = future.result()
                            for abs_idx, trans, vocab in batch_results:
                                translation_map[abs_idx] = trans
                                vocab_map[abs_idx] = vocab
                        except Exception as e:
                            print(f"배치 작업 중 에러 발생: {e}")
                            
            # [추가] 배치마다 개별 재요청하는 대신, 1차 라운드가 전부 끝난 뒤
            # 빠진 줄들만 영상 전체 기준으로 한 번에 모아 재요청한다.
            # (예: 배치 20개 중 3개에서 한 줄씩 빠졌다면, 이전에는 API 호출이
            #  3번 추가로 더 나갔지만 지금은 보통 1번만 더 나간다.)
            missing = [(idx, seg) for idx, seg in to_translate if idx not in translation_map]
            if missing and not self._cancel_current:
                self._progress(path, f"4/5단계: 누락된 {len(missing)}개 줄 재번역 중...")
                retry_batches = build_batches(missing)
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max(min(len(retry_batches), num_keys), 1)) as executor:
                    futures = [executor.submit(self.process_batch, b) for b in retry_batches]
                    for future in concurrent.futures.as_completed(futures):
                        if self._cancel_current:
                            for f in futures:
                                f.cancel()
                            break
                        try:
                            for abs_idx, trans, vocab in future.result():
                                translation_map[abs_idx] = trans
                                vocab_map[abs_idx] = vocab
                        except Exception as e:
                            print(f"재번역 배치 중 에러 발생: {e}")

            for idx, seg in enumerate(raw_segments):
                ai_meaning = translation_map.get(idx) or "(번역 실패 — 자막 편집에서 '번역 실패한 줄만 다시 요청')"
                ai_vocab = vocab_map.get(idx, [])
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'],
                    'phonetic': seg['phonetic'],
                    'meaning': ai_meaning,
                    'vocabulary': ai_vocab
                })

        if not segments:
            return []

        # 이번에 새로 성공한 번역을 로컬 사전(교정 파일)에 자동 누적 → 다음엔 API 없이 재사용
        corr_cache = load_corrections()
        cache_changed = False
        phrases = corr_cache.setdefault("phrases", {})
        for s in segments:
            jp = _norm_key(s['text'].strip())   # 여기를 정규화된 키로
            meaning = s.get('meaning', '')
            if jp and jp not in phrases and meaning and not meaning.startswith("(번역 실패"):
                phrases[jp] = {"meaning": meaning, "phonetic": s.get('phonetic', ''),
                               "vocabulary": s.get('vocabulary', []), "auto": True}
                cache_changed = True
        if cache_changed:
            save_corrections(corr_cache)

        self._progress(path, "5/5단계: 자막 파일 저장 중...")
        cache_path = path + ".json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)

        return segments

    def run(self):
        while True:
            with self._cond:
                while not self._jobs and not self._stop_requested:
                    self._cond.wait()
                if self._stop_requested:
                    break
                path = self._jobs.popleft()
                self.current_path = path
                self._cancel_current = False

            self.job_started.emit(path)
            try:
                segments = self._process_job(path)
                if segments is None:
                    self.job_failed.emit(path, "취소됨")
                elif not segments:
                    self.job_failed.emit(path, "인식된 대사가 없습니다")
                else:
                    self.job_finished.emit(path, segments)
            except Exception as e:
                self.job_failed.emit(path, str(e))

            with self._cond:
                self.current_path = ""
                idle = not self._jobs

            if idle:
                self._release_models()   # 대기열이 비면 GPU 메모리 반환
                self.queue_idle.emit()

        self._release_models()


class RetryTranslateWorker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(list)   # [(idx, translation, vocabulary)]

    def __init__(self, items):   # items: [(segment_idx, 일본어 원문)]
        super().__init__()
        self.items = items

    def run(self):
        results, BATCH = [], 20
        for b in range(0, len(self.items), BATCH):
            chunk = self.items[b:b + BATCH]
            lines = "\n".join(f"[{i}] {t}" for i, (_, t) in enumerate(chunk))
            prompt = f"""다음은 일본어 애니메이션 대사 목록이야. 각 대사를 자연스러운 한국어 구어체로 번역하고,
주요 단어 최대 5개의 원형과 한국어 뜻도 정리해줘. JSON 배열로만 답하고 전부 번역해줘.
[{{"index": 0, "translation": "번역", "vocabulary": [{{"word": "단어", "meaning": "뜻"}}]}}]

번역할 대사 목록:
{lines}"""
            key = GEMINI_API_KEYS[(b // BATCH) % len(GEMINI_API_KEYS)] if GEMINI_API_KEYS else ""
            for it in parse_json_array(gemini_call_with_backoff(prompt, key)):
                rel = it.get("index")
                if isinstance(rel, int) and 0 <= rel < len(chunk) and it.get("translation"):
                    results.append((chunk[rel][0], it["translation"], it.get("vocabulary", [])))
            self.progress.emit(f"재번역 중... {min(b + BATCH, len(self.items))}/{len(self.items)}")
        self.done.emit(results)

class SrtRebuilderWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, srt_path, json_path):
        super().__init__()
        self.srt_path = srt_path
        self.json_path = json_path

    def run(self):
        try:
            self.progress.emit("SRT 파싱 중...")
            meta_hint = ""
            meta_path = os.path.splitext(self.srt_path)[0] + ".meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        m = json.load(f)
                    meta_hint = (f"\n[작품 정보] 제목: {m.get('title','')} / "
                                f"등장인물: {', '.join(m.get('names', []))}\n")
                except Exception:
                    pass
            with open(self.srt_path, "r", encoding="utf-8") as f:
                content = f.read()

            pattern = re.compile(
                r'(\d+)\n(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})\n(.*?)(?=\n\n|\Z)',
                re.DOTALL)
            matches = pattern.findall(content)

            raw_segments = []
            for match in matches:
                start_sec = int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) + int(match[4]) / 1000.0
                end_sec = int(match[5]) * 3600 + int(match[6]) * 60 + int(match[7]) + int(match[8]) / 1000.0
                text = match[9].strip().replace('\n', ' ')
                raw_segments.append({'start': start_sec, 'end': end_sec, 'text': text})

            batch_text_lines = [f"[{idx}] {seg['text']}" for idx, seg in enumerate(raw_segments)]
            combined_text = "\n".join(batch_text_lines)

            self.progress.emit("AI 번역 및 단어장 추출 요청 중...")

            batch_prompt = f"""다음은 일본어 애니메이션 대사 목록이야. 각 대사를 자연스러운 한국어 구어체로 번역하고, 문장에 쓰인 주요 단어 최대 5개의 원형(기본형)과 한국어 뜻을 함께 정리해줘.
{meta_hint}
[엄격한 제약 사항]
1. 번역 결과에 일본어와 한국어가 절대 뒤섞이지 않게 온전한 한국어로만 번역해줘.
2. 반드시 아래의 JSON 배열 형태로만 결과를 반환해줘.

출력 JSON 구조 예시:
[
  {{
    "index": 0,
    "translation": "번역된 한국어 문장",
    "vocabulary": [
      {{"word": "단어원형1", "meaning": "뜻1"}}
    ]
  }}
]

번역할 대사 목록:
{combined_text}"""

            global current_key_index
            translated_raw = ""
            for _ in range(len(GEMINI_API_KEYS)):
                try:
                    api_key = GEMINI_API_KEYS[current_key_index]
                    translated_raw = gemini_json(api_key, batch_prompt)
                    break
                except Exception as e:
                    print(f"[Gemini SRT복구 실패] key#{current_key_index} {type(e).__name__}: {e}")
                    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
                    continue

            translation_map = {}
            vocab_map = {}
            if translated_raw:
                try:
                    clean_json_str = re.sub(r'```json\s*|\s*```', '', translated_raw).strip()
                    data_list = json.loads(clean_json_str)
                    for item in data_list:
                        idx = item.get("index")
                        if idx is not None:
                            translation_map[idx] = item.get("translation", "")
                            vocab_map[idx] = item.get("vocabulary", [])
                except Exception as parse_err:
                    print(f"SRT 복구 JSON 파싱 오류: {parse_err}")

            final_segments = []
            for idx, seg in enumerate(raw_segments):
                phonetic = japanese_to_korean_phonetic(seg['text'])
                ai_meaning = ai_meaning = translation_map.get(idx) or "(번역 실패)"
                ai_vocab = vocab_map.get(idx, [])

                final_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'],
                    'phonetic': phonetic,
                    'meaning': ai_meaning,
                    'vocabulary': ai_vocab
                })

            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(final_segments, f, ensure_ascii=False, indent=4)

            self.finished.emit(final_segments)
        except Exception as e:
            self.progress.emit(f"SRT 복구 실패: {str(e)}")
            self.finished.emit([])


# ---------------- 자막 스타일 공용 상수 / 헬퍼 ----------------
MENU_STYLE = """
    QMenu { background-color: #2b2b2b; color: white; border: 1px solid #555; }
    QMenu::item { padding: 6px 20px; }
    QMenu::item:selected { background-color: #007acc; }
    QMenu::separator { height: 1px; background: #555; margin: 4px 8px; }
"""

# 메뉴 안에 내장되는 컨트롤(폰트 콤보, 스핀박스, 슬라이더 등)의 다크 테마
MENU_WIDGET_STYLE = """
    QLabel { color: #DDDDDD; background: transparent; }
    QSpinBox, QComboBox, QFontComboBox {
        background-color: #3a3a3a; color: white; border: 1px solid #666;
        border-radius: 3px; padding: 2px 6px; min-height: 20px;
    }
    QComboBox QAbstractItemView {
        background-color: #2b2b2b; color: white;
        selection-background-color: #007acc; selection-color: white;
    }
    QPushButton {
        background-color: #3a3a3a; color: white; border: 1px solid #666;
        border-radius: 3px; min-width: 26px; max-width: 26px;
        min-height: 22px; font-size: 14px; font-weight: bold;
    }
    QPushButton:hover { background-color: #007acc; }
    QSlider::groove:horizontal { height: 4px; background: #555; border-radius: 2px; }
    QSlider::handle:horizontal {
        background: #007acc; width: 14px; margin: -5px 0; border-radius: 7px;
    }
"""

# (메뉴에 보일 이름, QFont 굵기값). 실제로 구분되어 보이는 단계 수는 폰트가 가진
# 굵기 종류에 따라 다르다. 예: 맑은 고딕은 보통/굵게 두 가지뿐이라 중간 단계는
# 가장 가까운 쪽으로 붙는다. (Yu Gothic 등은 Light/Medium/Bold 등을 따로 가짐)
FONT_WEIGHTS = [
    ("가늘게 (Light)", 25),
    ("보통 (Normal)", 50),
    ("중간 (Medium)", 57),
    ("약간 굵게 (DemiBold)", 63),
    ("굵게 (Bold)", 75),
    ("매우 굵게 (ExtraBold)", 81),
    ("최대 (Black)", 87),
]


def pick_default_font(writing_system, candidates):
    """시스템에 설치된 폰트 중 해당 언어를 지원하는 것 가운데 candidates 순서대로
    첫 번째를 기본 폰트로 고른다. 하나도 없으면 그 언어를 지원하는 아무 폰트나,
    그것도 없으면 시스템 기본 폰트."""
    try:
        installed = QFontDatabase().families(writing_system)
    except Exception:
        installed = []
    for name in candidates:
        if name in installed:
            return name
    if installed:
        return installed[0]
    return QFont().family()


def color_icon(color_hex, size=14):
    """메뉴 항목 옆에 현재 색상을 보여주는 작은 색 견본 아이콘."""
    pm = QPixmap(size, size)
    pm.fill(QColor(color_hex))
    painter = QPainter(pm)
    painter.setPen(QColor("#AAAAAA"))
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.end()
    return QIcon(pm)


def make_menu_widget_action(parent_menu, label_text, widget):
    """QMenu 안에 '라벨 + 컨트롤 위젯(스핀박스, 폰트콤보 등)'을 한 줄로 넣기 위한 헬퍼."""
    container = QWidget()
    container.setObjectName("menuRow")
    row = QHBoxLayout()
    row.setContentsMargins(14, 4, 14, 4)
    row.setSpacing(10)
    if label_text:
        lbl = QLabel(label_text)
        lbl.setMinimumWidth(56)
        row.addWidget(lbl)
    row.addWidget(widget, stretch=1)
    container.setLayout(row)
    container.setStyleSheet("QWidget#menuRow { background-color: #2b2b2b; }" + MENU_WIDGET_STYLE)
    action = QWidgetAction(parent_menu)
    action.setDefaultWidget(container)
    return action


def make_size_control(value, minimum, maximum, on_change, unit="pt"):
    """[ − ] [ 숫자 ] [ + ] 크기 조절 위젯.
    - − / + 버튼(누르고 있으면 연속 증감)
    - 가운데 숫자 칸에 직접 입력 가능, 현재 값이 항상 숫자로 표시됨"""
    box = QWidget()
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    minus = QPushButton("\u2212")
    plus = QPushButton("+")
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(int(value))
    spin.setAlignment(Qt.AlignCenter)
    spin.setButtonSymbols(QSpinBox.NoButtons)
    spin.setFixedWidth(56)

    for b in (minus, plus):
        b.setAutoRepeat(True)
        b.setFocusPolicy(Qt.NoFocus)
    minus.clicked.connect(lambda _=False: spin.setValue(spin.value() - 1))
    plus.clicked.connect(lambda _=False: spin.setValue(spin.value() + 1))
    spin.valueChanged.connect(on_change)

    row.addWidget(minus)
    row.addWidget(spin)
    row.addWidget(plus)
    if unit:
        row.addWidget(QLabel(unit))
    row.addStretch()
    box.setLayout(row)
    return box

#발음/뜻 이모지(🗣️💡)가 외곽선에 안 걸리게
_ICON_STRIP_RE = re.compile(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF\uFE0F]')

class OutlinedLabel(QWidget):
    """외곽선(테두리)을 지원하는 자막 텍스트 블록.

    동일한 텍스트를 담은 QLabel을 16방향으로 살짝 어긋나게 겹쳐 그려서
    외곽선처럼 보이게 하고, 그 위에 실제 글자색 라벨을 올린다.
    outline_width가 0이면(기본값) 외곽선 라벨들은 숨겨지고 일반 텍스트만 보인다.
    QLabel과 동일한 방식으로 setText/setFont/setFixedWidth/heightForWidth를
    지원해서 기존 QLabel을 쓰던 자리에 그대로 바꿔 끼울 수 있다."""

    OUTLINE_STEPS = 16  # 외곽선을 만드는 방향 수 (많을수록 굵어도 매끈함)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._outline_width = 0
        self._alignment = Qt.AlignCenter
        self._fill_style = None      # 마지막으로 적용한 스타일시트 (같으면 재적용 생략)
        self._outline_style = None

        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

        # 오버레이 위에서 클릭/드래그/우클릭이 항상 부모(SubtitleOverlayWindow)로
        # 그대로 전달되도록, 텍스트 라벨들은 마우스 이벤트를 받지 않게 한다.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._outline_labels = []
        for _ in range(self.OUTLINE_STEPS):
            lbl = QLabel(self)
            lbl.setWordWrap(True)
            lbl.setAlignment(self._alignment)
            lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            lbl.hide()
            self._outline_labels.append(lbl)

        self._fill_label = QLabel(self)
        self._fill_label.setWordWrap(True)
        self._fill_label.setAlignment(self._alignment)
        self._fill_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def setText(self, text):
        self._fill_label.setText(text)
        outline_text = _ICON_STRIP_RE.sub('', text)   # 이모지는 외곽선 복제본에서 제외 (뭉개짐 방지)
        for lbl in self._outline_labels:
            lbl.setText(outline_text)
        self.updateGeometry()

    def text(self):
        return self._fill_label.text()

    def setFont(self, font):
        super().setFont(font)
        self._fill_label.setFont(font)
        for lbl in self._outline_labels:
            lbl.setFont(font)
        self.updateGeometry()

    def setAlignment(self, align):
        self._alignment = align
        self._fill_label.setAlignment(align)
        for lbl in self._outline_labels:
            lbl.setAlignment(align)

    def setWordWrap(self, wrap):
        self._fill_label.setWordWrap(wrap)
        for lbl in self._outline_labels:
            lbl.setWordWrap(wrap)

    def set_colors(self, text_color, outline_color, outline_width, outline_opacity=255):
        self._outline_width = max(0, int(outline_width))
        fill_style = f"color: {QColor(text_color).name()}; background: transparent;"
        oc = QColor(outline_color)
        oc.setAlpha(max(0, min(255, int(outline_opacity))))
        outline_style = f"color: rgba({oc.red()}, {oc.green()}, {oc.blue()}, {oc.alpha()}); background: transparent;"
        if fill_style != self._fill_style:
            self._fill_label.setStyleSheet(fill_style)
            self._fill_style = fill_style
        if outline_style != self._outline_style:
            for lbl in self._outline_labels:
                lbl.setStyleSheet(outline_style)
            self._outline_style = outline_style
        for lbl in self._outline_labels:
            lbl.setVisible(self._outline_width > 0)
        self.updateGeometry()
        self._relayout()

    def heightForWidth(self, w):
        ow = self._outline_width
        inner_w = max(1, int(w) - 2 * ow)
        return self._fill_label.heightForWidth(inner_w) + 2 * ow

    def sizeHint(self):
        w = self._fill_label.sizeHint().width() + 2 * self._outline_width
        return QSize(w, self.heightForWidth(w))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()


    def _relayout(self):
        w, h = self.width(), self.height()
        ow = self._outline_width
        inner_w = max(1, w - 2 * ow)
        inner_h = max(1, h - 2 * ow)
        # 원(반지름 ow) 위의 16개 지점으로 어긋나게 겹쳐 그려 외곽선을 만든다
        for i, lbl in enumerate(self._outline_labels):
            ang = 2 * math.pi * i / self.OUTLINE_STEPS
            dx = int(round(ow * math.cos(ang)))
            dy = int(round(ow * math.sin(ang)))
            lbl.setGeometry(ow + dx, ow + dy, inner_w, inner_h)
        self._fill_label.setGeometry(ow, ow, inner_w, inner_h)


class SubtitleOverlayWindow(QWidget):
    """영상 위에 얹는 드래그/리사이즈 가능한 자막 오버레이 - '창 안의 창'.

    VLC 영상은 네이티브 윈도우 핸들에 그려지기 때문에 일반 자식 위젯을
    그 위에 얹으면 z-order가 꼬입니다. 그래서 이 오버레이는 별도의
    프레임리스 최상위(top-level) 창으로 만들고, 매 프레임 video_frame의
    화면 좌표에 맞춰 위치/크기를 재계산해서 따라다니게 합니다."""

    RESIZE_MARGIN = 10
    MIN_W = 160
    MIN_H = 46

    def __init__(self, player):
        # v0.2.2처럼 '항상 위' 플래그로 시작한다. 항상 위 모드가 아닐 때만 Win32로 해제한다 (apply_topmost_state)
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.player = player
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        # 배경은 스타일시트가 아니라 paintEvent에서 직접 그린다.
        # (투명 창 + 스타일시트 조합은 배경이 안 그려지는 문제가 있었음)
        self.setMouseTracking(True)
        self._hover = False  # 마우스가 올라와 있을 때만 테두리를 보여줘 '자막 영역'을 알려줌

        self._drag_pos = None
        self._resize_edge = None
        self._resize_start_geo = None
        self._resize_start_mouse = None

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        self.kanji_label = OutlinedLabel()
        layout.addWidget(self.kanji_label)

        self.meaning_label = OutlinedLabel()
        layout.addWidget(self.meaning_label)

        self.setLayout(layout)
        self.apply_style()

    def apply_style(self):
        """폰트/글자색/외곽선을 두 라벨(일본어 줄, 한국어 줄)에 반영하고 다시 그린다."""
        p = self.player
        for prefix, label in (("kanji", self.kanji_label), ("meaning", self.meaning_label)):
            label.setFont(p.make_subtitle_font(prefix))
            label.set_colors(getattr(p, f"overlay_{prefix}_color"),
                             getattr(p, f"overlay_{prefix}_outline_color"),
                             p.overlay_outline_width,
                             p.overlay_outline_opacity)
        self.update()

    def paintEvent(self, event):
        p = self.player
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = QColor(p.overlay_bg_color)
        # 알파가 완전히 0이면 Windows가 그 영역을 '클릭 통과'로 처리해서
        # 우클릭 메뉴/드래그가 안 먹게 된다. 눈에 안 보이는 최소값(1)을 유지.
        bg.setAlpha(max(1, int(p.overlay_bg_opacity)))
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(self.rect(), 8, 8)

        if self._hover:
            pen = QPen(QColor(255, 255, 255, 120))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    # ---------------- 모서리 리사이즈 판별 (메인 창과 동일한 방식) ----------------
    def _edge_at(self, pos):
        rect = self.rect()
        m = self.RESIZE_MARGIN
        left = pos.x() <= m
        right = pos.x() >= rect.width() - m
        top = pos.y() <= m
        bottom = pos.y() >= rect.height() - m
        if top and left:
            return 'top_left'
        if top and right:
            return 'top_right'
        if bottom and left:
            return 'bottom_left'
        if bottom and right:
            return 'bottom_right'
        if left:
            return 'left'
        if right:
            return 'right'
        if top:
            return 'top'
        if bottom:
            return 'bottom'
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            edge = self._edge_at(event.pos())
            if edge:
                self._resize_edge = edge
                self._resize_start_geo = QRect(self.geometry())
                self._resize_start_mouse = event.globalPos()
            else:
                self._drag_pos = event.globalPos()
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        if self._resize_edge and (event.buttons() & Qt.LeftButton):
            delta = event.globalPos() - self._resize_start_mouse
            geo = QRect(self._resize_start_geo)
            if 'left' in self._resize_edge:
                new_left = geo.left() + delta.x()
                if geo.right() - new_left >= self.MIN_W:
                    geo.setLeft(new_left)
            if 'right' in self._resize_edge:
                new_right = geo.right() + delta.x()
                if new_right - geo.left() >= self.MIN_W:
                    geo.setRight(new_right)
            if 'top' in self._resize_edge:
                new_top = geo.top() + delta.y()
                if geo.bottom() - new_top >= self.MIN_H:
                    geo.setTop(new_top)
            if 'bottom' in self._resize_edge:
                new_bottom = geo.bottom() + delta.y()
                if new_bottom - geo.top() >= self.MIN_H:
                    geo.setBottom(new_bottom)
            self.setGeometry(geo)
            self.player.on_overlay_user_resized(geo)
            return
        if self._drag_pos and (event.buttons() & Qt.LeftButton):
            delta = event.globalPos() - self._drag_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self._drag_pos = event.globalPos()
            self.player.on_overlay_user_moved(self.geometry())
            return

        edge = self._edge_at(event.pos())
        cursor_map = {
            'left': Qt.SizeHorCursor, 'right': Qt.SizeHorCursor,
            'top': Qt.SizeVerCursor, 'bottom': Qt.SizeVerCursor,
            'top_left': Qt.SizeFDiagCursor, 'bottom_right': Qt.SizeFDiagCursor,
            'top_right': Qt.SizeBDiagCursor, 'bottom_left': Qt.SizeBDiagCursor,
        }
        self.setCursor(cursor_map.get(edge, Qt.ArrowCursor))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
            self._resize_edge = None
            self.player.schedule_save_settings()   # 드래그/리사이즈가 끝나면 위치·크기 저장

    def show_context_menu(self, global_pos):
        # 본체 창 우클릭 메뉴와 동일한 메뉴(폰트/굵기/크기/색/외곽선/배경)를 공용으로 사용
        p = self.player
        menu = QMenu(self)
        p.build_subtitle_menu(menu)
        p.exec_subtitle_menu(menu, global_pos)


class SubtitleBox(QWidget):
    """좌/우 패널 모드에서 쓰는 자막 한 블록: 배경(기본 검정, 색/투명도 설정 반영) + OutlinedLabel."""

    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        self.label = OutlinedLabel()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def paintEvent(self, event):
        p = self.player
        bg = QColor(p.overlay_bg_color)
        bg.setAlpha(int(p.overlay_bg_opacity))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(self.rect(), 6, 6)

class SubtitleEditorDialog(QDialog):
    """자막(JSON/SRT) 직접 편집 + 수정 내역을 교정 사전에 기록"""
    COLS = ["시작(초)", "일본어 원문", "한글 발음", "한국어 뜻"]

    def __init__(self, player, segments, json_path, srt_path, parent=None):
        super().__init__(parent)
        self.player = player
        self.segments = segments
        self.json_path = json_path
        self.srt_path = srt_path
        self.original = [dict(s) for s in segments]   # 변경 비교용 스냅샷
        self.setWindowTitle("자막 편집기")
        self.resize(950, 620)
        self.setStyleSheet(PLAYLIST_STYLE + "QDialog{background:#1a1a1a;}"
                           "QTableWidget{background:#222;color:#DDD;gridline-color:#444;}"
                           "QHeaderView::section{background:#333;color:#DDD;padding:4px;}")

        self.table = QTableWidget(len(segments), len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        for r, s in enumerate(segments):
            for c, v in enumerate([f"{s.get('start', 0):.2f}", s.get('text', ''),
                                   s.get('phonetic', ''), s.get('meaning', '')]):
                item = QTableWidgetItem(str(v))
                if c == 0:
                    # 시작 시간은 표에서 직접 고치지 않고 아래 '싱크 편집' 패널로만
                    # 바꾸게 한다 (뒤 자막들과의 자동 보정을 거치게 하기 위해).
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, c, item)
        self.table.cellDoubleClicked.connect(self.on_seek)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)

        # --- 싱크 편집 패널: 자막을 한 줄만 선택하면 표 아래에 나타난다 ---
        self.sync_panel = QWidget()
        sync_layout = QHBoxLayout()
        sync_layout.setContentsMargins(6, 6, 6, 6)
        self.sync_row_label = QLabel("")
        self.sync_row_label.setStyleSheet("color:#DDD;")
        sync_layout.addWidget(self.sync_row_label, stretch=1)

        sync_layout.addWidget(QLabel("시작(초):"))
        self.sync_start_spin = QDoubleSpinBox()
        self.sync_start_spin.setRange(0.0, 999999.0)
        self.sync_start_spin.setDecimals(2)
        self.sync_start_spin.setSingleStep(0.1)
        sync_layout.addWidget(self.sync_start_spin)

        self.sync_keep_duration_chk = QCheckBox("길이 유지 (끝시간도 함께 이동)")
        self.sync_keep_duration_chk.setChecked(True)
        sync_layout.addWidget(self.sync_keep_duration_chk)

        self.sync_cascade_chk = QCheckBox("이후 모든 자막도 같이 밀기")
        self.sync_cascade_chk.setChecked(True)
        self.sync_cascade_chk.setToolTip("이 줄부터 끝까지, 이동한 만큼(오프셋) 전부 같이 밀어서\n"
                                         "이 지점 이후로 전체가 계속 밀려 있던 싱크를 한번에 맞춥니다.")
        sync_layout.addWidget(self.sync_cascade_chk)

        self.btn_apply_sync = QPushButton("✅ 싱크 적용")
        self.btn_apply_sync.setFocusPolicy(Qt.NoFocus)
        self.btn_apply_sync.clicked.connect(self.apply_sync_edit)
        sync_layout.addWidget(self.btn_apply_sync)

        self.sync_panel.setLayout(sync_layout)
        self.sync_panel.setStyleSheet("background:#222; border:1px solid #444; border-radius:4px;")
        self.sync_panel.hide()
        self._sync_row = None

        btn_row = QHBoxLayout()
        b_re = QPushButton("🔁 선택 줄 발음 재생성")
        b_re.clicked.connect(self.regen_phonetic)
        b_retry = QPushButton("❗ 번역 실패한 줄만 다시 요청")
        b_xlsx = QPushButton("📊 단어장 엑셀로 내보내기")
        b_xlsx.clicked.connect(self.export_vocabulary_xlsx)
        b_retry.clicked.connect(self.retry_failed_translations)
        b_save = QPushButton("💾 저장 (JSON + SRT + 교정 사전)")
        b_save.clicked.connect(self.save_all)
        b_close = QPushButton("닫기")
        b_close.clicked.connect(self.reject)
        for b in (b_re, b_retry, b_xlsx, b_save, b_close):
            b.setFocusPolicy(Qt.NoFocus)
            btn_row.addWidget(b)

        self.status_label = QLabel("")

        lay = QVBoxLayout()
        lay.addWidget(QLabel("행을 더블클릭하면 해당 위치로 영상이 이동합니다. "
                             "뜻을 고쳐 저장하면 같은 대사에 자동으로 재적용됩니다."))
        lay.addWidget(self.table, stretch=1)
        lay.addWidget(self.sync_panel)
        lay.addLayout(btn_row)
        lay.addWidget(self.status_label)
        self.setLayout(lay)

    def on_seek(self, row, _col):
        try:
            self.player.media_player.set_time(int(float(self.table.item(row, 0).text()) * 1000))
        except Exception:
            pass
    def on_selection_changed(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1:
            self.sync_panel.hide()
            self._sync_row = None
            return
        r = next(iter(rows))
        self._sync_row = r
        seg = self.segments[r]
        preview = seg.get('text', '')[:24]
        self.sync_row_label.setText(
            f"[{r + 1}번째 줄] {preview}{'…' if len(seg.get('text', '')) > 24 else ''}")
        self.sync_start_spin.blockSignals(True)
        self.sync_start_spin.setValue(seg.get('start', 0.0))
        self.sync_start_spin.blockSignals(False)
        self.sync_panel.show()

    def _refresh_start_column(self, from_row=0):
        for i in range(from_row, len(self.segments)):
            self.table.item(i, 0).setText(f"{self.segments[i]['start']:.2f}")

    def apply_sync_edit(self):
        r = self._sync_row
        if r is None or not (0 <= r < len(self.segments)):
            return

        seg = self.segments[r]
        old_start = seg['start']
        new_start = self.sync_start_spin.value()
        delta = new_start - old_start
        if abs(delta) < 0.005:
            self.status_label.setText("변경된 시간이 없습니다.")
            return

        # 선택한 줄: 길이 유지가 켜져 있으면 끝 시간도 같이 이동
        if self.sync_keep_duration_chk.isChecked():
            seg['end'] = seg['end'] + delta
        seg['start'] = new_start
        if seg['end'] < seg['start'] + 0.05:
            seg['end'] = seg['start'] + 0.05

        # 이후 자막도 함께 밀기: 선택한 줄부터 끝까지 전부 같은 만큼 이동시켜서
        # 이 지점 이후 전체가 밀려 있던 싱크를 한 번에 맞춘다.
        if self.sync_cascade_chk.isChecked():
            for i in range(r + 1, len(self.segments)):
                self.segments[i]['start'] += delta
                self.segments[i]['end'] += delta

        # 바로 앞 줄과 겹치면 그 줄의 끝 시간만 당겨서 겹침을 없앤다
        # (앞 줄들의 싱크 자체는 이미 맞아 있다고 보고 건드리지 않음)
        if r > 0:
            prev = self.segments[r - 1]
            if prev['end'] > seg['start']:
                prev['end'] = max(prev['start'] + 0.05, seg['start'])

        self._refresh_start_column(from_row=max(0, r - 1))
        note = f"[{r + 1}번째 줄] 시작 시간을 {old_start:.2f}초 → {new_start:.2f}초로 조정"
        if self.sync_cascade_chk.isChecked():
            note += f" (이후 {len(self.segments) - r - 1}개 줄도 {delta:+.2f}초 이동)"
        self.status_label.setText(note + ". '💾 저장'을 눌러야 반영됩니다.")

    def regen_phonetic(self):
        for idx in {i.row() for i in self.table.selectedIndexes()}:
            jp = self.table.item(idx, 1).text()
            self.table.item(idx, 2).setText(japanese_to_korean_phonetic(jp))

    def retry_failed_translations(self):
        """뜻 칸이 비었거나 '(번역 실패'로 시작하는 줄만 모아서 다시 번역 요청."""
        fail_rows = [r for r in range(self.table.rowCount())
                    if not self.table.item(r, 3).text().strip()
                    or self.table.item(r, 3).text().startswith("(번역 실패")]
        if not fail_rows:
            self.status_label.setText("번역 실패한 줄이 없습니다.")
            return

        self.status_label.setText(f"{len(fail_rows)}개 줄 재번역 요청 중...")
        QApplication.processEvents()

        lines = [f"[{i}] {self.table.item(r, 1).text().strip()}" for i, r in enumerate(fail_rows)]
        prompt = f"""다음은 일본어 애니메이션 대사 목록이야. 각 대사를 자연스러운 한국어 구어체로 번역하고,
문장에 쓰인 주요 단어 최대 5개의 원형과 한국어 뜻도 함께 정리해줘.
반드시 아래 JSON 배열 형태로만 결과를 반환해줘. 목록에 있는 대사는 하나도 빠짐없이 전부 번역해줘.

출력 예시:
[{{"index": 0, "translation": "번역", "vocabulary": [{{"word": "단어", "meaning": "뜻"}}]}}]

번역할 대사 목록:
{chr(10).join(lines)}"""

        key = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""
        raw = gemini_call_with_backoff(prompt, key)
        data_list = parse_json_array(raw)
        if not data_list:
            self.status_label.setText("재번역 실패 (API 응답 없음 — 잠시 후 다시 시도해주세요)")
            return

        done = 0
        for item in data_list:
            rel = item.get("index")
            if rel is None or not (0 <= rel < len(fail_rows)):
                continue
            r = fail_rows[rel]
            trans = item.get("translation", "")
            if trans:
                self.table.item(r, 3).setText(trans)
                self.segments[r]['vocabulary'] = item.get("vocabulary", [])
                done += 1
        self.status_label.setText(f"{done}/{len(fail_rows)}개 재번역 완료. '💾 저장'을 눌러야 반영됩니다.")

    def export_vocabulary_xlsx(self):
        if not HAS_OPENPYXL:
            self.status_label.setText("openpyxl 미설치: pip install openpyxl 실행 후 다시 시도하세요.")
            return
        rows, seen = [], set()
        for s in self.segments:
            for v in s.get('vocabulary', []):
                w, m = v.get('word', '').strip(), v.get('meaning', '').strip()
                if w and (w, m) not in seen:
                    seen.add((w, m))
                    rows.append((w, m, s.get('text', '')))
        if not rows:
            self.status_label.setText("추출된 단어가 없습니다.")
            return

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "단어장"
        ws.append(["일본어 단어", "한국어 뜻", "예문(원문)"])
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append(r)
        widths = [18, 28, 60]
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[chr(64 + i)].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        out_path = os.path.splitext(self.video_path)[0] + "_단어장.xlsx"
        try:
            wb.save(out_path)
            self.status_label.setText(f"저장 완료: {out_path} ({len(rows)}개 단어)")
        except Exception as e:
            self.status_label.setText(f"저장 실패: {e}")
    
    def save_all(self):
        corr = load_corrections()
        for r, s in enumerate(self.segments):
            jp = self.table.item(r, 1).text().strip()
            ph = self.table.item(r, 2).text().strip()
            mn = self.table.item(r, 3).text().strip()
            old = self.original[r]
            if mn != (old.get('meaning') or '') or ph != (old.get('phonetic') or ''):
                record_correction(corr, jp, old.get('meaning', ''), mn, ph, s.get('vocabulary'))
            # 👉 이 부분을 save_all 안의 반복문 속에 넣으라는 의미입니다!
            if jp != (old.get('text') or ''):
                log_stt_correction(self.player.video_path, old.get('text', ''), jp)
                
            s['text'], s['phonetic'], s['meaning'] = jp, ph, mn
        save_corrections(corr)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.segments, f, ensure_ascii=False, indent=4)
        with open(self.srt_path, "w", encoding="utf-8") as f:
            for i, s in enumerate(self.segments, start=1):
                f.write(f"{i}\n{seconds_to_srt_time(s['start'])} --> "
                        f"{seconds_to_srt_time(s['end'])}\n{s['text'].strip()}\n\n")

        self.player.segments = self.segments
        self.player.show_current_segment(jump_player=False)
        self.accept()


# ---------------- 영상 목록 / 설정 저장 / Windows 창 제어 헬퍼 ----------------
VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.ts', '.mpg', '.mpeg')
VIDEO_FILTER = "Video Files (" + " ".join("*" + e for e in VIDEO_EXTS) + ");;All Files (*)"
SIMILAR_NAME_RATIO = 0.6   # 파일명 유사도(0~1)가 이 값 이상이면 '비슷한 영상'으로 본다


def norm_path(p):
    return os.path.normpath(os.path.abspath(p))


def natural_key(s):
    """'ep2' < 'ep10' 처럼 숫자를 숫자로 비교하는 정렬 키."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def _name_key(filename):
    """유사도 비교용 파일명: 확장자/CRC 태그 제거, 숫자(화수)는 '#'로 통일."""
    base = os.path.splitext(filename)[0].lower()
    base = re.sub(r'\[[0-9a-f]{8}\]', '', base)
    return re.sub(r'\d+', '#', base)


def scan_similar_videos(video_path, show_all=False):
    """video_path와 같은 폴더에서 이름이 비슷한 영상(같은 시리즈의 다른 화 등)을 모아 자연 정렬해 반환.
    show_all=True면 이름과 무관하게 폴더 안의 모든 영상을 반환."""
    video_path = norm_path(video_path)
    folder = os.path.dirname(video_path)
    ref_name = os.path.basename(video_path)
    ref_key = _name_key(ref_name)
    try:
        names = [n for n in os.listdir(folder) if n.lower().endswith(VIDEO_EXTS)]
    except OSError:
        return [video_path]

    result = []
    for n in names:
        p = os.path.join(folder, n)
        if (show_all or n == ref_name
                or SequenceMatcher(None, _name_key(n), ref_key).ratio() >= SIMILAR_NAME_RATIO):
            result.append(p)
    if video_path not in result:
        result.append(video_path)
    result.sort(key=lambda p: natural_key(os.path.basename(p)))
    return result


# ---- 설정 저장 (폰트/색/외곽선/배경/위치/창 동작 등을 다음 실행에도 유지) ----
try:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _APP_DIR = os.getcwd()
API_KEYS_FILE = os.path.join(_APP_DIR, "gemini_api_keys.json")   # ← 여기로 이동
SETTINGS_FILE = os.path.join(_APP_DIR, "subtitle_player_settings.json")

CORRECTIONS_FILE = os.path.join(_APP_DIR, "subtitle_corrections.json")

STT_CORRECTIONS_FILE = os.path.join(_APP_DIR, "stt_text_corrections.json")

def log_stt_correction(video_path, before, after):
    """일본어 원문(STT 결과) 수정 이력만 별도 로그. 번역 캐시와 절대 섞지 않는다."""
    if before == after:
        return
    try:
        try:
            with open(STT_CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
            if not isinstance(log, list):
                log = []
        except Exception:
            log = []
        log.append({"video": os.path.basename(video_path), "before": before, "after": after})
        with open(STT_CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(log[-2000:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"STT 수정 로그 저장 실패: {e}")

def load_corrections():
    try:
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {"phrases": {}}
    except Exception:
        return {"phrases": {}}

def save_corrections(corr):
    try:
        tmp = CORRECTIONS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(corr, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CORRECTIONS_FILE)
    except Exception as e:
        print(f"교정 사전 저장 실패: {e}")

def record_correction(corr, jp, before, after, phonetic=None, vocabulary=None):
    """사용자가 고친 내용을 '오답 노트'로 누적"""
    jp = (jp or "").strip()
    if not jp or before == after:
        return
    e = corr.setdefault("phrases", {}).setdefault(jp, {})
    e["meaning"] = after
    e["wrong"] = before
    e["count"] = e.get("count", 0) + 1
    if phonetic:
        e["phonetic"] = phonetic
    if vocabulary:
        e["vocabulary"] = vocabulary

def apply_corrections(segments, corr):
    """자막 로드 시 교정 사전을 자동 반영 (로컬 규칙 기반 학습)"""
    ph = {_norm_key(k): v for k, v in corr.get("phrases", {}).items()}
    hit = 0
    for s in segments:
        e = ph.get(_norm_key((s.get("text") or "").strip()))
        if e:
            if e.get("meaning"):
                s["meaning"] = e["meaning"]
            if e.get("phonetic"):
                s["phonetic"] = e["phonetic"]
            hit += 1
    return hit


def _s_text(v):
    if not isinstance(v, str):
        raise ValueError
    return v


def _s_bool(v):
    if not isinstance(v, bool):
        raise ValueError
    return v


def _s_color(v):
    if not isinstance(v, str) or not QColor(v).isValid():
        raise ValueError
    return QColor(v).name()


def _s_num(lo, hi, cast):
    def check(v):
        if isinstance(v, bool):
            raise ValueError
        return max(lo, min(hi, cast(v)))
    return check


def _s_choice(*options):
    def check(v):
        if v not in options:
            raise ValueError
        return v
    return check


# 저장할 설정 항목 (속성 이름 → 값 검증 함수). 잘못된 값이 들어 있으면 해당 항목만 무시하고 기본값 사용.
def _s_str_list(v):
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError
    return [x for x in v if os.path.isdir(x)][:20]

def _s_str_list_plain(v):
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError
    return [x.strip() for x in v if x.strip()][:20]

SETTING_SPECS = {
    "kanji_font_family": _s_text, "kanji_font_size": _s_num(6, 120, int),
    "kanji_font_weight": _s_num(0, 99, int),
    "meaning_font_family": _s_text, "meaning_font_size": _s_num(6, 120, int),
    "meaning_font_weight": _s_num(0, 99, int),
    "overlay_kanji_color": _s_color, "overlay_meaning_color": _s_color,
    "overlay_kanji_outline_color": _s_color, "overlay_meaning_outline_color": _s_color,
    "overlay_outline_width": _s_num(0, 30, int),
    "overlay_outline_opacity": _s_num(0, 255, int),
    "overlay_bg_color": _s_color, "overlay_bg_opacity": _s_num(0, 255, int),
    "ov_x_ratio": _s_num(0.0, 1.0, float), "ov_y_ratio": _s_num(0.0, 1.0, float),
    "ov_w_ratio": _s_num(0.08, 1.0, float), "ov_min_h_ratio": _s_num(0.03, 1.0, float),
    "subtitle_display_mode": _s_choice("overlay", "side_left", "side_right"),
    "mode": _s_choice("sentence", "word"),
    "topmost_mode": _s_choice("always", "playing", "off"),
    "hide_console": _s_bool,
    "playlist_visible": _s_bool, "playlist_show_all": _s_bool, "auto_next": _s_bool,
    "last_dir": _s_text,
    "extra_folders": _s_str_list,
    "long_vowel_style": _s_choice("repeat", "dash"),
    "gemini_api_keys": _s_str_list_plain
}

def read_saved_last_dir():
    """프로그램 시작 직후(플레이어 생성 전) 파일 선택창의 시작 폴더를 얻기 위한 간단 로더."""
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f).get("last_dir", "")
        return d if isinstance(d, str) and os.path.isdir(d) else ""
    except Exception:
        return ""


# ---- Windows 전용 창 제어 (ctypes) ----
def win_set_topmost(widget, on):
    """창 재생성 없이(=VLC 임베딩이 풀리지 않게) '항상 위' 상태만 바꾼다."""
    if not IS_WIN:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SetWindowPos.restype = wintypes.BOOL
        HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
        user32.SetWindowPos(int(widget.winId()), HWND_TOPMOST if on else HWND_NOTOPMOST,
                            0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
        return True
    except Exception as e:
        print(f"항상 위 설정 실패: {e}")
        return False


def win_foreground_is(*widgets):
    """현재 맨 앞(포그라운드) 창이 주어진 위젯들 중 하나인지."""
    if not IS_WIN:
        return True
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        fg = user32.GetForegroundWindow() or 0
        return any(int(w.winId()) == fg for w in widgets)
    except Exception:
        return True


def win_console_hwnd():
    if not IS_WIN:
        return 0
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetConsoleWindow.restype = wintypes.HWND
        return k32.GetConsoleWindow() or 0
    except Exception:
        return 0


def win_show_console(show):
    """콘솔(cmd) 창 표시/숨김. 콘솔이 없으면(pythonw 실행 등) False."""
    hwnd = win_console_hwnd()
    if not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        SW_HIDE, SW_SHOW = 0, 5
        user32.ShowWindow(hwnd, SW_SHOW if show else SW_HIDE)
        return True
    except Exception:
        return False


PLAYLIST_STYLE = """
    QLabel { color: #DDDDDD; background: transparent; font-size: 11px; }
    QCheckBox { color: #DDDDDD; background: transparent; font-size: 11px; }
    QCheckBox::indicator {
        width: 13px; height: 13px; border: 1px solid #FFD700;
        border-radius: 3px; background: transparent;
    }
    QCheckBox::indicator:checked { background-color: #FFD700; }
    QListWidget { background-color: #222222; color: #DDDDDD; border: 1px solid #444; font-size: 11px; }
    QListWidget::item { padding: 3px 2px; }
    QListWidget::item:selected { background-color: #007acc; color: white; }
    QPushButton {
        background-color: rgba(255, 255, 255, 15); color: #FFFFFF; border: 1px solid #FFFFFF;
        border-radius: 4px; padding: 3px 6px; font-size: 11px;
    }
    QPushButton:hover { background-color: rgba(255, 255, 255, 50); }
"""

# 맨 윗줄 '항상 위' 버튼: 클릭할 때마다 항상 위 → 재생 시에만 → 해제 → 항상 위 ... 순환
TOPMOST_MODES = ("always", "playing", "off")
TOPMOST_INFO = {
    "always": ("📌 항상 위", "현재: 영상 항상 위에 고정\n클릭 → '재생 시에만 위에 고정'으로 변경"),
    "playing": ("▶ 재생 중에만 위", "현재: 영상 재생 시에만 항상 위에 고정 (일시정지하면 다른 창 뒤로 갈 수 있음)\n"
                                    "클릭 → '위에 고정 해제'로 변경"),
    "off": ("🔓 고정 해제", "현재: 항상 위에 고정 해제 (일반 창처럼 동작)\n클릭 → '항상 위에 고정'으로 변경"),
}


# 2. 학습용 플레이어 창 (VLC 임베딩)
class AnimeSubtitlePlayer(QWidget):

    RESIZE_MARGIN = 8
    MIN_WIDTH = 640
    MIN_HEIGHT = 360
    SIDE_PANEL_WIDTH = 280
    PLAYLIST_PANEL_WIDTH = 300

    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.old_pos = None
        self.resize_edge = None
        self.resize_start_geo = None
        self.resize_start_mouse = None

        self.segments = []
        self.current_index = 0
        self.mode = "sentence"  # "sentence" 또는 "word"
        self.subtitle_display_mode = "overlay"  # "overlay" | "side_left" | "side_right"
        self.is_slider_dragging = False
        self.is_maximized_custom = False
        self.long_vowel_style = QSettings("AnimeSubtitlePlayer", "App").value("long_vowel_style", "repeat")
        # [추가] 모듈 전역 변수도 함께 동기화
        global LONG_VOWEL_STYLE
        LONG_VOWEL_STYLE = self.long_vowel_style
        # --- 자막 스타일 (오버레이 / 좌·우 패널 공용) ---
        # 일본어 줄 = 원문(kanji_*), 한국어 줄 = 발음·뜻(meaning_*)
        # 폰트는 시스템에 설치된 것 중 각 언어를 지원하는 첫 후보로 자동 선택
        self.kanji_font_family = pick_default_font(
            QFontDatabase.Japanese,
            ["Yu Gothic UI", "Yu Gothic", "Meiryo UI", "Meiryo", "MS UI Gothic",
             "MS Gothic", "Malgun Gothic"])
        self.kanji_font_size = 18
        self.kanji_font_weight = 75   # Bold
        self.meaning_font_family = pick_default_font(
            QFontDatabase.Korean,
            ["Malgun Gothic", "맑은 고딕", "Gulim", "Dotum", "NanumGothic"])
        self.meaning_font_size = 13
        self.meaning_font_weight = 50  # Normal
        self.overlay_outline_width = 0    # 0 = 없음 (기본값)
        self.overlay_outline_opacity = 255   # ← 여기로 이동
        self.overlay_bg_color = "#000000"  # 기본: 검정
        self.overlay_bg_opacity = 179      # 0(투명) ~ 255(불투명), 179 ≈ 70%
        self.overlay_kanji_color = "#FFD700"
        self.overlay_meaning_color = "#E8E8E8"
        self.overlay_kanji_outline_color = "#000000"
        self.overlay_meaning_outline_color = "#000000"
        self.overlay_outline_width = 0    # 0 = 없음 (기본값)
        self.overlay_bg_color = "#000000"  # 기본: 검정
        self.overlay_bg_opacity = 179      # 0(투명) ~ 255(불투명), 179 ≈ 70%
        # 메뉴/색상창이 열려 있는 동안 오버레이가 raise_()로 메뉴 위를 덮지 않게 하는 카운터
        self._suspend_raise = 0

        # --- 오버레이 자막(창 안의 창) 위치/크기 ---
        # video_frame 대비 비율로 저장해서 창 크기가 바뀌어도 같은 상대 위치/크기를 유지합니다.
        self.ov_x_ratio = 0.1
        self.ov_w_ratio = 0.8
        self.ov_min_h_ratio = 0.12
        self.ov_y_ratio = 1 - self.ov_min_h_ratio - 0.04  # 기본: 하단 중앙 부근

        # --- 창 동작 / 재생 목록 / 자막 생성 대기열 (저장 대상) ---
        self.topmost_mode = "always"   # "always"(항상 위) | "playing"(재생 중에만) | "off"(해제)
        self.hide_console = False      # 콘솔(cmd) 창 숨김 여부
        self.playlist_visible = True   # 재생 목록 패널 표시
        self.playlist_show_all = False  # True면 폴더 안 모든 영상, False면 비슷한 이름만
        self.auto_next = True          # 영상이 끝나면 목록의 다음 영상 자동 재생
        self.last_dir = ""
        self.extra_folders = []
        self.saved_window_geometry = None
        self.saved_maximized = False

        # --- 런타임 상태 (저장 안 함) ---
        self.playlist_paths = []        # 재생 목록(스캔 결과)
        self._playlist_items = {}       # 경로 -> QListWidgetItem
        self.job_status = {}            # 경로 -> {'state': queued|running|failed, 'text': ..., 'percent': ...}
        self.queue_worker = None
        self._rebuild_workers = []
        # 두 창 모두 WindowStaysOnTopHint로 시작하므로 처음엔 '항상 위' 상태로 본다
        self._is_topmost = True
        self._overlay_topmost_applied = True
        self._ended_handled = False
        self._first_show_done = False
        self._shutting_down = False
        self._settings_ready = False
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(600)
        self._settings_timer.timeout.connect(self.save_settings)
        #제미나이 API키
        self.gemini_api_keys = []
        # 저장된 설정이 있으면 위 기본값을 덮어쓴다
        self.load_settings()

        # --- VLC 인스턴스 및 플레이어 준비 (창 임베딩은 initUI 이후, load_video에서) ---
        self.vlc_instance = vlc.Instance(VLC_INSTANCE_ARGS)
        self.media_player = self.vlc_instance.media_player_new()

        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(250)
        self.sync_timer.timeout.connect(self.on_timer_tick)

        self.subtitle_disable_timer = QTimer(self)
        self.subtitle_disable_timer.setInterval(400)
        self.subtitle_disable_timer.timeout.connect(self._retry_disable_subtitle)
        self._subtitle_disable_retries = 0

        self.initUI()
        self.restore_window_geometry()
        self.video_frame.installEventFilter(self)

        # 오버레이 창은 video_frame이 만들어진 뒤에 생성
        self.subtitle_overlay = SubtitleOverlayWindow(self)

        self.apply_subtitle_layout(self.subtitle_display_mode)

        self._settings_ready = True   # 여기부터 설정 변경이 파일로 저장된다
        if not GEMINI_API_KEYS:
            QTimer.singleShot(300, self.manage_api_keys)
        self.apply_console_setting()

    # ---------------- UI ----------------
    def initUI(self):
        # 타이틀바가 없는 프레임리스 창. 단, Qt.Tool을 쓰면 작업표시줄에
        # 아이콘이 안 남아 최소화해도 되돌릴 방법이 없어지므로 빼둡니다.
        # '항상 위' 플래그로 시작하고, 모드 변경(항상/재생 중에만/해제)은 창을 다시 만들지 않도록
        # Win32 SetWindowPos로 바꾼다 (apply_topmost_state).
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(400, 150, 1000, 650)
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMouseTracking(True)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(self.RESIZE_MARGIN, self.RESIZE_MARGIN,
                                        self.RESIZE_MARGIN, self.RESIZE_MARGIN)
        main_layout.setSpacing(6)

        btn_style = """
            QPushButton {
                background-color: rgba(255, 255, 255, 15);
                color: #FFFFFF;
                border: 1px solid #FFFFFF;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 50);
            }
        """
        
        icon_btn_style = """
            QPushButton {
                background-color: transparent;
                color: #DDDDDD;
                border: none;
                font-family: Consolas;
                font-size: 13px;
                min-width: 32px;
                max-width: 32px;
                min-height: 24px;
                max-height: 24px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 30);
            }
        """
        icon_btn_close_style = """
            QPushButton {
                background-color: transparent;
                color: #DDDDDD;
                border: none;
                font-family: Consolas;
                font-size: 13px;
                min-width: 32px;
                max-width: 32px;
                min-height: 24px;
                max-height: 24px;
            }
            QPushButton:hover {
                background-color: rgba(220, 50, 50, 220);
                color: white;
            }
        """

        # --- 상단 줄: 이전/다음/모드(좌) + 최소화/최대화/닫기(우) ---
        # 이 줄 전체가 Windows 타이틀바 역할을 합니다: 버튼이 없는 빈 부분을
        # 클릭 드래그하면 창이 이동합니다 (mousePressEvent에서 이 위젯의
        # geometry로 판정).
        self.title_bar_widget = QWidget()
        self.title_bar_widget.setFixedHeight(30)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        self.title_bar_widget.setLayout(control_layout)

        # 항상 위 3단계 순환 버튼 (항상 위 → 재생 시에만 → 해제)
        self.btn_topmost = QPushButton()
        self.btn_topmost.setStyleSheet(btn_style)
        self.btn_topmost.setFocusPolicy(Qt.NoFocus)
        self.btn_topmost.clicked.connect(self.cycle_topmost_mode)
        control_layout.addWidget(self.btn_topmost)
        self.update_topmost_button()

        self.btn_open = QPushButton("📂 열기")
        self.btn_open.setStyleSheet(btn_style)
        self.btn_open.setFocusPolicy(Qt.NoFocus)
        self.btn_open.clicked.connect(self.open_video_dialog)
        control_layout.addWidget(self.btn_open)

        self.btn_prev = QPushButton("◀ 이전 문장")
        self.btn_prev.setStyleSheet(btn_style)
        self.btn_prev.setFocusPolicy(Qt.NoFocus)
        self.btn_prev.clicked.connect(self.prev_segment)
        control_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("다음 문장 ▶")
        self.btn_next.setStyleSheet(btn_style)
        self.btn_next.setFocusPolicy(Qt.NoFocus)
        self.btn_next.clicked.connect(self.next_segment)
        control_layout.addWidget(self.btn_next)

        self.btn_mode = QPushButton("🔄 모드")
        self.btn_mode.setStyleSheet(btn_style)
        self.btn_mode.setFocusPolicy(Qt.NoFocus)
        self.btn_mode.clicked.connect(self.toggle_mode)
        control_layout.addWidget(self.btn_mode)

        self.btn_playlist = QPushButton("📋 목록")
        self.btn_playlist.setStyleSheet(btn_style)
        self.btn_playlist.setFocusPolicy(Qt.NoFocus)
        self.btn_playlist.setToolTip("재생 목록 / 자막 생성 대기열 패널 열기·닫기")
        self.btn_playlist.clicked.connect(self.toggle_playlist)
        control_layout.addWidget(self.btn_playlist)

        control_layout.addStretch()

        # 최소화 → 최대화 → 닫기 순서, 아이콘만
        self.btn_minimize = QPushButton("─")
        self.btn_minimize.setStyleSheet(icon_btn_style)
        self.btn_minimize.setFocusPolicy(Qt.NoFocus)
        self.btn_minimize.clicked.connect(self.showMinimized)
        control_layout.addWidget(self.btn_minimize)

        self.btn_maximize = QPushButton("☐")
        self.btn_maximize.setStyleSheet(icon_btn_style)
        self.btn_maximize.setFocusPolicy(Qt.NoFocus)
        self.btn_maximize.clicked.connect(self.toggle_maximize)
        control_layout.addWidget(self.btn_maximize)

        self.btn_close = QPushButton("✕")
        self.btn_close.setStyleSheet(icon_btn_close_style)
        self.btn_close.setFocusPolicy(Qt.NoFocus)
        self.btn_close.clicked.connect(self.close_app)
        control_layout.addWidget(self.btn_close)

        main_layout.addWidget(self.title_bar_widget)

        # --- 콘텐츠 영역: 영상 열 + 자막 패널 (순서/표시는 apply_subtitle_layout이 관리) ---
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(6)
        main_layout.addLayout(self.content_layout, stretch=1)

        # --- 영상 열 (비디오 프레임 + 재생바) ---
        self.video_column_widget = QWidget()
        video_col_layout = QVBoxLayout()
        video_col_layout.setContentsMargins(0, 0, 0, 0)
        video_col_layout.setSpacing(6)

        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: black;")
        self.video_frame.setMinimumHeight(300)
        # 위젯을 강제로 네이티브 윈도우로 만들어 winId()가 유효하게 함
        self.video_frame.setAttribute(Qt.WA_NativeWindow, True)
        self.video_frame.setMouseTracking(True) 
        video_col_layout.addWidget(self.video_frame, stretch=1)

        playback_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setStyleSheet(btn_style)
        self.btn_play.setFocusPolicy(Qt.NoFocus)
        self.btn_play.clicked.connect(self.toggle_play_pause)
        playback_layout.addWidget(self.btn_play)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setFocusPolicy(Qt.NoFocus)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #555; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #FFD700; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #FFD700; width: 14px; height: 14px;
                margin: -6px 0; border-radius: 7px;
            }
        """)
        self.seek_slider.sliderPressed.connect(self.on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self.on_slider_released)
        playback_layout.addWidget(self.seek_slider, stretch=1)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #E0E0E0; font-size: 11px;")
        playback_layout.addWidget(self.time_label)

        video_col_layout.addLayout(playback_layout)
        self.video_column_widget.setLayout(video_col_layout)

        # --- 자막 패널 (좌/우 도킹 모드에서만 보임) ---
        self.subtitle_panel_widget = QWidget()
        self.subtitle_panel_widget.setFixedWidth(self.SIDE_PANEL_WIDTH)
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)

        self.kanji_box = SubtitleBox(self)
        self.kanji_label = self.kanji_box.label
        self.kanji_label.setText("우클릭 메뉴 또는 영상을 선택해주세요.")
        panel_layout.addWidget(self.kanji_box)

        self.meaning_box = SubtitleBox(self)
        self.meaning_label = self.meaning_box.label
        self.meaning_label.setText("상단 버튼을 이용하거나 Tab 키를 누르세요.")
        panel_layout.addWidget(self.meaning_box)
        panel_layout.addStretch()

        self.subtitle_panel_widget.setLayout(panel_layout)

        # --- 재생 목록 패널 (같은 폴더의 비슷한 이름 영상 + 자막 생성 대기열) ---
        self.playlist_panel_widget = QWidget()
        self.playlist_panel_widget.setFixedWidth(self.PLAYLIST_PANEL_WIDTH)
        self.playlist_panel_widget.setStyleSheet(PLAYLIST_STYLE)
        pl_layout = QVBoxLayout()
        pl_layout.setContentsMargins(0, 0, 0, 0)
        pl_layout.setSpacing(6)

        pl_head = QHBoxLayout()
        pl_head.addWidget(QLabel("📋 재생 목록"))
        pl_head.addStretch()
        self.chk_show_all = QCheckBox("폴더 전체 보기")
        self.chk_show_all.setFocusPolicy(Qt.NoFocus)
        self.chk_show_all.setChecked(self.playlist_show_all)
        self.chk_show_all.setToolTip("끄면 지금 영상과 이름이 비슷한 영상(같은 시리즈)만 보여줍니다")
        self.chk_show_all.toggled.connect(self.on_show_all_toggled)
        pl_head.addWidget(self.chk_show_all)
        pl_layout.addLayout(pl_head)

        self.playlist_widget = QListWidget()
        self.playlist_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.playlist_widget.setFocusPolicy(Qt.NoFocus)   # 방향키/스페이스는 플레이어 단축키로 계속 동작
        self.playlist_widget.setTextElideMode(Qt.ElideMiddle)
        self.playlist_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.playlist_widget.customContextMenuRequested.connect(self.show_playlist_menu)
        self.playlist_widget.itemDoubleClicked.connect(self.on_playlist_item_activated)
        pl_layout.addWidget(self.playlist_widget, stretch=1)

        nav_row = QHBoxLayout()
        self.btn_prev_video = QPushButton("⏮ 이전 화")
        self.btn_prev_video.setFocusPolicy(Qt.NoFocus)
        self.btn_prev_video.clicked.connect(lambda _=False: self.play_relative(-1))
        nav_row.addWidget(self.btn_prev_video)
        self.btn_next_video = QPushButton("다음 화 ⏭")
        self.btn_next_video.setFocusPolicy(Qt.NoFocus)
        self.btn_next_video.clicked.connect(lambda _=False: self.play_relative(1))
        nav_row.addWidget(self.btn_next_video)
        pl_layout.addLayout(nav_row)

        self.chk_auto_next = QCheckBox("영상이 끝나면 다음 화 자동 재생")
        self.chk_auto_next.setFocusPolicy(Qt.NoFocus)
        self.chk_auto_next.setChecked(self.auto_next)
        self.chk_auto_next.toggled.connect(self.on_auto_next_toggled)
        pl_layout.addWidget(self.chk_auto_next)

        job_row1 = QHBoxLayout()

        self.btn_gen_selected = QPushButton("🎬 선택 영상 자막 생성")
        self.btn_gen_selected.setFocusPolicy(Qt.NoFocus)
        self.btn_gen_selected.setToolTip("목록에서 Ctrl/Shift로 여러 개를 고른 뒤 누르면\n순서대로 백그라운드에서 자막을 만듭니다")
        self.btn_gen_selected.clicked.connect(lambda _=False: self.queue_selected_videos())

        self.btn_add_videos = QPushButton("➕ 영상 추가")
        self.btn_add_videos.setFocusPolicy(Qt.NoFocus)
        self.btn_add_videos.setToolTip("다른 폴더의 영상도 여러 개 골라서 자막 생성 대기열에 넣습니다")
        self.btn_add_videos.clicked.connect(lambda _=False: self.add_videos_to_queue_dialog())

        self.btn_add_folder = QPushButton("📁 폴더 추가")
        self.btn_add_folder.setFocusPolicy(Qt.NoFocus)
        self.btn_add_folder.setToolTip("폴더 안의 모든 영상을 재생 목록에 추가합니다 (하위 폴더 포함)")
        self.btn_add_folder.clicked.connect(lambda _=False: self.add_folder_dialog())

        job_row1.addWidget(self.btn_gen_selected)
        job_row1.addWidget(self.btn_add_videos)
        job_row1.addWidget(self.btn_add_folder)
        pl_layout.addLayout(job_row1)

        job_row2 = QHBoxLayout()
        self.btn_cancel_job = QPushButton("⏹ 현재 작업 중지")
        self.btn_cancel_job.setFocusPolicy(Qt.NoFocus)
        self.btn_cancel_job.clicked.connect(lambda _=False: self.cancel_current_job())
        job_row2.addWidget(self.btn_cancel_job)
        self.btn_clear_queue = QPushButton("🗑 대기 비우기")
        self.btn_clear_queue.setFocusPolicy(Qt.NoFocus)
        self.btn_clear_queue.clicked.connect(lambda _=False: self.clear_queue())
        job_row2.addWidget(self.btn_clear_queue)
        pl_layout.addLayout(job_row2)

        job_row3 = QHBoxLayout()
        self.btn_retry_failed = QPushButton("❗ 번역 실패 줄만 재요청")
        self.btn_retry_failed.setFocusPolicy(Qt.NoFocus)
        self.btn_retry_failed.setToolTip("현재 재생 중인 영상의 번역 실패 문장만 다시 AI에 요청합니다")
        self.btn_retry_failed.clicked.connect(lambda _=False: self.retry_failed_translations_for_current())
        job_row3.addWidget(self.btn_retry_failed)
        self.btn_export_vocab = QPushButton("📊 단어장 엑셀로 내보내기")
        self.btn_export_vocab.setFocusPolicy(Qt.NoFocus)
        self.btn_export_vocab.clicked.connect(lambda _=False: self.export_vocabulary_xlsx_for_current())
        job_row3.addWidget(self.btn_export_vocab)
        pl_layout.addLayout(job_row3)

        self.queue_status_label = QLabel("자막 생성 대기열: 없음")
        self.queue_status_label.setWordWrap(True)
        self.queue_status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.queue_status_label.setMinimumHeight(56)
        self.queue_status_label.setMaximumHeight(96)
        pl_layout.addWidget(self.queue_status_label)

        self.playlist_panel_widget.setLayout(pl_layout)



        self.setLayout(main_layout)
        self.apply_side_panel_style()

    def close_app(self):
        self.shutdown()
        QApplication.quit()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.btn_maximize.setText("☐")
        else:
            self.showMaximized()
            self.btn_maximize.setText("❐")

    # ---------------- VLC 재생 제어 ----------------
    def embed_video(self):
        """video_frame의 네이티브 윈도우 핸들에 VLC 렌더링을 붙인다.
        위젯이 실제로 화면에 표시된(winId가 유효한) 이후에 호출해야 한다."""
        if sys.platform.startswith("win"):
            self.media_player.set_hwnd(int(self.video_frame.winId()))
        elif sys.platform.startswith("linux"):
            self.media_player.set_xwindow(int(self.video_frame.winId()))
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(int(self.video_frame.winId()))
        try:
            self.media_player.video_set_mouse_input(False)
            self.media_player.video_set_key_input(False)
        except Exception:
            pass

    def load_media(self, video_path):
        media = self.vlc_instance.media_new(video_path)
        self.media_player.set_media(media)
        self.embed_video()
        self._ended_handled = False
        self.media_player.play()
        self.btn_play.setText("⏸")
        self.sync_timer.start()

        # 같은 폴더의 .srt를 VLC가 자체적으로 자동 표시하지 않도록 명시적으로 끔.
        # 트랙 목록은 재생이 시작된 뒤 비동기로 채워지므로, 즉시 한 번 끄고
        # 이후 몇 차례 더 재시도해서 큰 파일에서도 확실히 꺼지게 한다.
        self.disable_native_subtitle_track()
        self._subtitle_disable_retries = 0
        self.subtitle_disable_timer.start()

        # video_frame 크기가 확정된 뒤 오버레이 위치도 다시 계산
        self.apply_subtitle_layout(self.subtitle_display_mode)
        self.apply_topmost_state()

    def disable_native_subtitle_track(self):
        try:
            self.media_player.video_set_spu(-1)
        except Exception:
            pass

    def _retry_disable_subtitle(self):
        self.disable_native_subtitle_track()
        self._subtitle_disable_retries += 1
        if self._subtitle_disable_retries >= 8:
            self.subtitle_disable_timer.stop()

    def toggle_play_pause(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.btn_play.setText("▶")
        else:
            self.media_player.play()
            self.btn_play.setText("⏸")
        self.apply_topmost_state()

    def on_slider_pressed(self):
        self.is_slider_dragging = True

    def on_slider_released(self):
        length = self.media_player.get_length()
        if length and length > 0:
            target_ms = int(self.seek_slider.value() / 1000.0 * length)
            self.media_player.set_time(target_ms)
        self.is_slider_dragging = False

    def on_timer_tick(self):
        """0.25초마다: 재생바 갱신 + 현재 재생 위치에 맞는 자막 갱신"""
        cur_ms = self.media_player.get_time()
        length_ms = self.media_player.get_length()

        if cur_ms is not None and cur_ms >= 0:
            if not self.is_slider_dragging and length_ms and length_ms > 0:
                pos = int(cur_ms / length_ms * 1000)
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(pos)
                self.seek_slider.blockSignals(False)
            self.time_label.setText(f"{ms_to_mmss(cur_ms)} / {ms_to_mmss(length_ms)}")

            self.sync_subtitle_with_time(cur_ms / 1000.0)

        # '재생 중에만 항상 위' 모드: 재생/일시정지 상태가 바뀌면 여기서 반영된다
        self.apply_topmost_state()

        # 영상이 끝나면 (자동 다음 화가 켜져 있을 때) 재생 목록의 다음 영상으로 넘어간다
        try:
            ended = self.media_player.get_state() == vlc.State.Ended
        except Exception:
            ended = False
        if not ended:
            self._ended_handled = False   # 다시 재생되면 다음에 끝날 때도 처리되도록 초기화
        elif not self._ended_handled:
            self._ended_handled = True
            self.btn_play.setText("▶")
            if self.auto_next:
                self.play_relative(1)

        # VLC 네이티브 영상 창이 가끔 스스로를 앞으로 올리는 경우가 있어,
        # 오버레이 자막 창이 항상 그 위에 보이도록 주기적으로 올려준다.
        self.bring_overlay_above()

    def sync_subtitle_with_time(self, current_sec):
        if not self.segments:
            return
        for idx, seg in enumerate(self.segments):
            if seg['start'] <= current_sec <= seg['end']:
                if self.current_index != idx:
                    self.current_index = idx
                    self.show_current_segment(jump_player=False)  # 자막만 갱신
                break

    # ---------------- 자막 표시 모드 (오버레이 / 좌측 패널 / 우측 패널) ----------------
    def apply_subtitle_layout(self, mode):
        self.subtitle_display_mode = mode

        # 기존에 붙어있던 위젯을 content_layout에서 떼어낸다 (삭제 아님)
        self.content_layout.removeWidget(self.video_column_widget)
        self.content_layout.removeWidget(self.subtitle_panel_widget)
        self.content_layout.removeWidget(self.playlist_panel_widget)

        if mode == "overlay":
            self.subtitle_panel_widget.hide()
            self.content_layout.addWidget(self.video_column_widget, stretch=1)
        elif mode == "side_left":
            self.subtitle_panel_widget.show()
            self.content_layout.addWidget(self.subtitle_panel_widget, stretch=0)
            self.content_layout.addWidget(self.video_column_widget, stretch=1)
        elif mode == "side_right":
            self.subtitle_panel_widget.show()
            self.content_layout.addWidget(self.video_column_widget, stretch=1)
            self.content_layout.addWidget(self.subtitle_panel_widget, stretch=0)

        # 재생 목록 패널은 항상 맨 오른쪽 (보이기/숨기기는 '📋 목록' 버튼)
        self.content_layout.addWidget(self.playlist_panel_widget, stretch=0)
        self.playlist_panel_widget.setVisible(self.playlist_visible)

        self.update_overlay_geometry()
        self.schedule_save_settings()

    # ---------------- 오버레이 자막(창 안의 창) 위치/크기 계산 ----------------
    def update_overlay_geometry(self):
        """video_frame의 현재 화면 좌표를 기준으로 오버레이 창의 위치/크기를
        다시 계산한다. 저장된 비율(ov_x_ratio 등)을 그대로 따르되, 자막
        텍스트가 넘칠 경우 높이를 자동으로 늘리고, 그 결과 영상 하단을
        넘어서면 위쪽으로 확장되도록(아래쪽 기준선은 고정) 보정한다."""
        if not hasattr(self, "subtitle_overlay"):
            return

        if (self.subtitle_display_mode != "overlay" or self.isMinimized()
                or not self.isVisible()):
            self.subtitle_overlay.hide()
            return

        frame = self.video_frame
        fw, fh = frame.width(), frame.height()
        if fw <= 0 or fh <= 0:
            return

        origin = frame.mapToGlobal(QPoint(0, 0))

        # 좌우로 잘리지 않도록 폭을 영상 폭 안으로 제한
        w = max(self.subtitle_overlay.MIN_W, int(self.ov_w_ratio * fw))
        w = min(w, fw - 4)
        x = int(self.ov_x_ratio * fw)
        x = max(0, min(x, fw - w))

        # 줄바꿈을 반영한 실제 필요 높이 계산
        inner_w = max(10, w - 24)
        self.subtitle_overlay.kanji_label.setFixedWidth(inner_w)
        self.subtitle_overlay.meaning_label.setFixedWidth(inner_w)
        needed_h = (self.subtitle_overlay.kanji_label.heightForWidth(inner_w)
                    + self.subtitle_overlay.meaning_label.heightForWidth(inner_w) + 40)

        min_h = max(self.subtitle_overlay.MIN_H, int(self.ov_min_h_ratio * fh))
        h = max(min_h, needed_h)
        h = min(h, fh)  # 영상보다 커지지는 않게

        y = int(self.ov_y_ratio * fh)
        # 자막이 넘쳐서 아래쪽으로 커지면 영상 하단을 넘지 않도록 위로 밀어올림
        if y + h > fh:
            y = max(0, fh - h)
        y = max(0, min(y, fh - self.subtitle_overlay.MIN_H))

        self.subtitle_overlay.setGeometry(origin.x() + x, origin.y() + y, w, h)
        self.subtitle_overlay.show()
        if IS_WIN and self._overlay_topmost_applied != self._is_topmost:
            win_set_topmost(self.subtitle_overlay, self._is_topmost)
            self._overlay_topmost_applied = self._is_topmost
        self.bring_overlay_above()

    def on_overlay_user_moved(self, geo):
        """사용자가 오버레이를 드래그로 옮긴 뒤, video_frame 대비 비율로 저장."""
        frame = self.video_frame
        fw, fh = frame.width(), frame.height()
        if fw <= 0 or fh <= 0:
            return
        origin = frame.mapToGlobal(QPoint(0, 0))
        self.ov_x_ratio = max(0.0, min(1.0, (geo.x() - origin.x()) / fw))
        self.ov_y_ratio = max(0.0, min(1.0, (geo.y() - origin.y()) / fh))

    def on_overlay_user_resized(self, geo):
        """사용자가 오버레이 모서리를 드래그로 리사이즈한 뒤, 비율로 저장.
        이후 update_overlay_geometry가 이 값을 최소 크기로 사용하고,
        텍스트가 더 길면 자동으로 더 확장한다."""
        frame = self.video_frame
        fw, fh = frame.width(), frame.height()
        if fw <= 0 or fh <= 0:
            return
        origin = frame.mapToGlobal(QPoint(0, 0))
        self.ov_x_ratio = max(0.0, min(1.0, (geo.x() - origin.x()) / fw))
        self.ov_y_ratio = max(0.0, min(1.0, (geo.y() - origin.y()) / fh))
        self.ov_w_ratio = max(0.08, min(1.0, geo.width() / fw))
        self.ov_min_h_ratio = max(0.03, min(1.0, geo.height() / fh))
        self.update_overlay_geometry()

    def reset_overlay_position(self):
        """우클릭 메뉴: 오버레이를 영상 하단 중앙 기본 위치/크기로 초기화."""
        self.ov_w_ratio = 0.8
        self.ov_x_ratio = (1 - self.ov_w_ratio) / 2
        self.ov_min_h_ratio = 0.12
        self.ov_y_ratio = 1 - self.ov_min_h_ratio - 0.04
        self.update_overlay_geometry()
        self.schedule_save_settings()

    # ---------------- 자막 스타일 (폰트 / 색 / 외곽선 / 배경) ----------------
    def make_subtitle_font(self, prefix):
        """prefix: 'kanji'(일본어 줄) 또는 'meaning'(한국어 발음·뜻 줄)의 현재 설정으로 QFont 생성."""
        font = QFont(getattr(self, f"{prefix}_font_family"),
                     int(getattr(self, f"{prefix}_font_size")))
        font.setWeight(int(getattr(self, f"{prefix}_font_weight")))
        return font

    def apply_side_panel_style(self):
        for prefix, box in (("kanji", self.kanji_box), ("meaning", self.meaning_box)):
            box.label.setFont(self.make_subtitle_font(prefix))
            box.label.set_colors(getattr(self, f"overlay_{prefix}_color"),
                                 getattr(self, f"overlay_{prefix}_outline_color"),
                                 self.overlay_outline_width)
            box.updateGeometry()
            box.update()

    def refresh_subtitle_background(self):
        """배경 색/투명도만 바뀐 경우: 다시 그리기만 하면 된다."""
        if hasattr(self, "subtitle_overlay"):
            self.subtitle_overlay.update()
        self.kanji_box.update()
        self.meaning_box.update()
        self.schedule_save_settings()

    def apply_subtitle_style(self):
        """오버레이 + 좌/우 패널 모두에 현재 스타일을 반영하고 오버레이 크기를 다시 계산."""
        self.apply_side_panel_style()
        if hasattr(self, "subtitle_overlay"):
            self.subtitle_overlay.apply_style()
        self.update_overlay_geometry()
        self.schedule_save_settings()

    def set_subtitle_font(self, prefix, kind, value):
        """prefix: 'kanji' | 'meaning',  kind: 'family' | 'size' | 'weight'"""
        setattr(self, f"{prefix}_font_{kind}", value)
        self.apply_subtitle_style()

    def set_overlay_outline_width(self, value):
        self.overlay_outline_width = max(0, min(30, int(value)))
        self.apply_subtitle_style()
        
    def set_overlay_outline_opacity(self, value):
        self.overlay_outline_opacity = max(0, min(255, int(value)))
        self.apply_subtitle_style()

    def set_overlay_bg_opacity(self, value):
        self.overlay_bg_opacity = max(0, min(255, int(value)))
        self.refresh_subtitle_background()

    def reset_overlay_bg(self):
        self.overlay_bg_color = "#000000"
        self.overlay_bg_opacity = 179
        self.refresh_subtitle_background()

    def manage_api_keys(self):
        from PyQt5.QtWidgets import QInputDialog
        current = "\n".join(GEMINI_API_KEYS)
        self._suspend_raise += 1
        try:
            text, ok = QInputDialog.getMultiLineText(
                self, "Gemini API 키 관리",
                "한 줄에 키 하나씩 입력하세요 (여러 개 등록 시 자동으로 번갈아 사용됩니다).\n"
                "발급: https://aistudio.google.com/apikey",
                current)
        finally:
            self._suspend_raise -= 1
        if not ok:
            return
        new_keys = [line.strip() for line in text.splitlines() if line.strip()]
        self.gemini_api_keys = new_keys
        set_api_keys(new_keys)
        self.schedule_save_settings()
        self.queue_status_label.setText(f"API 키 {len(new_keys)}개 저장됨.")

    def pick_color(self, attr, title):
        """attr에 해당하는 색상 속성을 색상 선택창으로 바꾼다."""
        self._suspend_raise += 1
        try:
            color = QColorDialog.getColor(QColor(getattr(self, attr)), self, title)
        finally:
            self._suspend_raise -= 1
        if color.isValid():
            setattr(self, attr, color.name())
            self.apply_subtitle_style()

    # ---------------- 자막 설정 메뉴 (본체 창 / 오버레이 공용) ----------------
    def exec_subtitle_menu(self, menu, global_pos):
        # 메뉴가 열려 있는 동안에는 오버레이 창이 메뉴 위로 올라오지 않게 한다
        self._suspend_raise += 1
        try:
            menu.exec_(global_pos)
        finally:
            self._suspend_raise -= 1
            menu.deleteLater()

    def _add_language_menu(self, menu, title, lang_name, prefix, writing_system):
        """한 언어(일본어 줄 / 한국어 줄)의 폰트·굵기·크기·색·외곽선색 하위 메뉴."""
        sub = menu.addMenu(title)

        # 폰트: 시스템에 설치된 폰트 중 해당 언어를 지원하는 것만 나열
        # (전체 폰트를 보고 싶다면 아래 setWritingSystem 줄을 지우면 됨)
        font_combo = QFontComboBox()
        font_combo.setFontFilters(QFontComboBox.ScalableFonts)
        font_combo.setWritingSystem(writing_system)
        font_combo.setEditable(False)
        font_combo.setMinimumWidth(230)
        font_combo.setCurrentFont(QFont(getattr(self, f"{prefix}_font_family")))
        font_combo.currentFontChanged.connect(
            lambda f, pf=prefix: self.set_subtitle_font(pf, "family", f.family()))
        sub.addAction(make_menu_widget_action(sub, "폰트", font_combo))

        # 굵기
        weight_combo = QComboBox()
        for label, w in FONT_WEIGHTS:
            weight_combo.addItem(label, w)
        cur_w = int(getattr(self, f"{prefix}_font_weight"))
        weight_combo.setCurrentIndex(
            min(range(len(FONT_WEIGHTS)), key=lambda i: abs(FONT_WEIGHTS[i][1] - cur_w)))
        weight_combo.currentIndexChanged.connect(
            lambda i, pf=prefix, cb=weight_combo: self.set_subtitle_font(pf, "weight", cb.itemData(i)))
        sub.addAction(make_menu_widget_action(sub, "굵기", weight_combo))

        # 크기: − / + / 숫자 직접 입력, 현재 크기가 숫자로 표시됨
        size_ctrl = make_size_control(
            getattr(self, f"{prefix}_font_size"), 6, 120,
            lambda v, pf=prefix: self.set_subtitle_font(pf, "size", v))
        sub.addAction(make_menu_widget_action(sub, "크기", size_ctrl))

        sub.addSeparator()

        color_attr = f"overlay_{prefix}_color"
        color_act = QAction(color_icon(getattr(self, color_attr)), "글자 색상...", sub)
        color_act.triggered.connect(
            lambda checked=False, a=color_attr: self.pick_color(a, f"{lang_name} 글자 색상 선택"))
        sub.addAction(color_act)

        outline_attr = f"overlay_{prefix}_outline_color"
        outline_act = QAction(color_icon(getattr(self, outline_attr)), "외곽선 색상...", sub)
        outline_act.triggered.connect(
            lambda checked=False, a=outline_attr: self.pick_color(a, f"{lang_name} 외곽선 색상 선택"))
        sub.addAction(outline_act)

    def build_subtitle_menu(self, menu, with_exit=False):
        menu.setStyleSheet(MENU_STYLE)

        mode_text = "문장 버전" if self.mode == "sentence" else "단어장 버전"
        mode_action = QAction(f"🔄 보기 모드 전환 (현재: {mode_text})", menu)
        mode_action.triggered.connect(self.toggle_mode)
        menu.addAction(mode_action)

        layout_menu = menu.addMenu("🖼 자막 표시 방식")
        for label, mode in [("영상 위에 오버레이", "overlay"),
                            ("오른쪽 패널로 분리", "side_right"),
                            ("왼쪽 패널로 분리", "side_left")]:
            act = QAction(label, layout_menu)
            act.setCheckable(True)
            act.setChecked(self.subtitle_display_mode == mode)
            act.triggered.connect(lambda checked=False, m=mode: self.apply_subtitle_layout(m))
            layout_menu.addAction(act)

        menu.addSeparator()

        self._add_language_menu(menu, "🔤 일본어 자막 (원문)", "일본어", "kanji", QFontDatabase.Japanese)
        self._add_language_menu(menu, "🔤 한국어 자막 (발음·뜻)", "한국어", "meaning", QFontDatabase.Korean)

        # 외곽선 두께는 일본어/한국어 공통 (색상만 각각 따로)
        outline_widget = make_size_control(
            self.overlay_outline_width, 0, 30, self.set_overlay_outline_width, unit="px")
        menu.addAction(make_menu_widget_action(menu, "✏️ 외곽선 두께 (0 = 없음)", outline_widget))

        outline_op_pct = int(round(self.overlay_outline_opacity * 100 / 255))
        outline_op_slider = QSlider(Qt.Horizontal)
        outline_op_slider.setRange(0, 100)
        outline_op_slider.setValue(outline_op_pct)
        outline_op_slider.setMinimumWidth(150)
        outline_op_label = QLabel(f"{outline_op_pct}%")
        outline_op_label.setMinimumWidth(38)
        
        def on_outline_opacity(v):
            outline_op_label.setText(f"{v}%")
            self.set_overlay_outline_opacity(round(v * 255 / 100))

        outline_op_slider.valueChanged.connect(on_outline_opacity)
        outline_op_row = QWidget()
        outline_op_layout = QHBoxLayout()
        outline_op_layout.setContentsMargins(0, 0, 0, 0)
        outline_op_layout.addWidget(outline_op_slider, stretch=1)
        outline_op_layout.addWidget(outline_op_label)
        outline_op_row.setLayout(outline_op_layout)
        menu.addAction(make_menu_widget_action(menu, "외곽선 투명도", outline_op_row))

        # 배경: 색상 + 불투명도 슬라이더 (기본: 검정 70%)
        bg_menu = menu.addMenu("🌓 자막 배경")
        pct = int(round(self.overlay_bg_opacity * 100 / 255))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(pct)
        slider.setMinimumWidth(150)
        pct_label = QLabel(f"{pct}%")
        pct_label.setMinimumWidth(38)

        def on_opacity(v):
            pct_label.setText(f"{v}%")
            self.set_overlay_bg_opacity(round(v * 255 / 100))

        slider.valueChanged.connect(on_opacity)
        slider_row = QWidget()
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(slider, stretch=1)
        slider_layout.addWidget(pct_label)
        slider_row.setLayout(slider_layout)
        bg_menu.addAction(make_menu_widget_action(bg_menu, "불투명도", slider_row))

        bg_color_act = QAction(color_icon(self.overlay_bg_color), "배경 색상 변경... (기본: 검정)", bg_menu)
        bg_color_act.triggered.connect(
            lambda checked=False: self.pick_color("overlay_bg_color", "자막 배경 색상 선택"))
        bg_menu.addAction(bg_color_act)

        bg_reset_act = QAction("배경 기본값으로 (검정 · 70%)", bg_menu)
        bg_reset_act.triggered.connect(self.reset_overlay_bg)
        bg_menu.addAction(bg_reset_act)

        edit_act = QAction("📝 현재 영상 자막 편집...", menu)
        edit_act.setEnabled(bool(self.segments))
        edit_act.triggered.connect(self.open_subtitle_editor)
        menu.addAction(edit_act)
        menu.addSeparator()

        reset_action = QAction("↩ 오버레이 위치/크기 초기화 (하단 중앙)", menu)
        reset_action.triggered.connect(self.reset_overlay_position)
        menu.addAction(reset_action)

        if IS_WIN:
            console_ok = bool(win_console_hwnd())
            console_act = QAction("🖥 콘솔(cmd) 창 숨기기" if console_ok else "🖥 콘솔 창 없음", menu)
            console_act.setCheckable(True)
            console_act.setChecked(self.hide_console)
            console_act.setEnabled(console_ok)
            console_act.triggered.connect(
                lambda checked=False: self.set_console_hidden(not self.hide_console))
            menu.addAction(console_act)
        
        key_act = QAction(f"🔑 API 키 관리 (현재 {len(GEMINI_API_KEYS)}개 등록됨)", menu)
        key_act.triggered.connect(self.manage_api_keys)
        menu.addAction(key_act)

        if with_exit:
            menu.addSeparator()
            exit_action = QAction("❌ 프로그램 종료", menu)
            exit_action.triggered.connect(self.close_app)
            menu.addAction(exit_action)
    
    # ---------------- 키보드 ----------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close_app()
            return

        if event.key() == Qt.Key_Space:
            self.toggle_play_pause()
            event.accept()
            return

        if event.key() == Qt.Key_Tab:
            self.mode = "word" if self.mode == "sentence" else "sentence"
            self.show_current_segment()
            self.schedule_save_settings()
            event.accept()
            return

        if not self.segments:
            return

        if event.key() == Qt.Key_Right:
            if self.current_index < len(self.segments) - 1:
                self.current_index += 1
                self.show_current_segment(jump_player=True)
        elif event.key() == Qt.Key_Left:
            if self.current_index > 0:
                self.current_index -= 1
                self.show_current_segment(jump_player=True)

    def show_current_segment(self, jump_player=False):
        if 0 <= self.current_index < len(self.segments):
            seg = self.segments[self.current_index]
            start_sec = seg['start']
            text = seg['text'].strip()

            mode_str = "📜 [문장 버전]" if self.mode == "sentence" else "📚 [단어장 버전]"
            self.setWindowTitle(
                f"애니 학습기 {mode_str} - [{self.current_index + 1} / {len(self.segments)}] ({int(start_sec)}초)")

            if self.mode == "sentence":
                self.set_kanji_text(text)
                phonetic = seg.get('phonetic', '')
                ai_meaning = seg.get('meaning', '')
                self.set_meaning_text(f"🗣️ 발음: {phonetic}\n💡 뜻: {ai_meaning}")
            else:
                vocab_list = seg.get('vocabulary', [])
                if vocab_list:
                    vocab_texts = [f"{v.get('word')} ({v.get('meaning')})" for v in vocab_list]
                    vocab_display = " | ".join(vocab_texts)
                else:
                    vocab_display = "핵심 단어 정보 없음"

                self.set_kanji_text(f"단어장: {vocab_display}")
                ai_meaning = seg.get('meaning', '번역 정보 없음')
                self.set_meaning_text(f"💡 문장 전체 뜻: {ai_meaning}")

            # 수동 이동(버튼 클릭, 방향키 등) 시 실제 재생 위치도 함께 이동
            if jump_player:
                self.media_player.set_time(int(start_sec * 1000))

    def prev_segment(self):
        if self.segments and self.current_index > 0:
            self.current_index -= 1
            self.show_current_segment(jump_player=True)

    def next_segment(self):
        if self.segments and self.current_index < len(self.segments) - 1:
            self.current_index += 1
            self.show_current_segment(jump_player=True)

    def open_subtitle_editor(self):
        if not self.segments or not self.video_path:
            return
        self._suspend_raise += 1
        try:
            dlg = SubtitleEditorDialog(self, self.segments,
                                       self.video_path + ".json",
                                       os.path.splitext(self.video_path)[0] + ".srt", self)
            dlg.exec_()
        finally:
            self._suspend_raise -= 1

    
    def retry_failed_translations_for_current(self):
        w = getattr(self, "_retry_worker", None)
        if w is not None and w.isRunning():
            return
        if not self.segments or not self.video_path:
            self.queue_status_label.setText("먼저 자막이 있는 영상을 재생하세요.")
            return
        fail_idx = [i for i, s in enumerate(self.segments)
                    if not s.get('meaning') or s['meaning'].startswith("(번역 실패")]
        if not fail_idx:
            self.queue_status_label.setText("번역 실패한 문장이 없습니다.")
            return
        self._retry_video = self.video_path
        self.queue_status_label.setText(f"{len(fail_idx)}개 문장 재번역 요청 중...")
        w = RetryTranslateWorker([(i, self.segments[i]['text']) for i in fail_idx])
        w.progress.connect(self.queue_status_label.setText)
        w.done.connect(lambda res, n=len(fail_idx): self._on_retry_done(res, n))
        self._retry_worker = w
        w.start()

    def _on_retry_done(self, results, total):
        if norm_path(self._retry_video) != norm_path(self.video_path):
            return   # 그 사이 다른 영상으로 바뀌면 반영하지 않음
        if not results:
            self.queue_status_label.setText("재번역 실패 (쿼터 초과 가능 — 1~2분 뒤 다시 시도)")
            return
        corr = load_corrections()
        for idx, trans, vocab in results:
            seg = self.segments[idx]
            seg['meaning'], seg['vocabulary'] = trans, vocab
            record_correction(corr, seg['text'], "", trans, seg.get('phonetic'), vocab)
        save_corrections(corr)
        with open(self.video_path + ".json", "w", encoding="utf-8") as f:
            json.dump(self.segments, f, ensure_ascii=False, indent=4)
        with open(os.path.splitext(self.video_path)[0] + ".srt", "w", encoding="utf-8") as f:
            for i, s in enumerate(self.segments, start=1):
                f.write(f"{i}\n{seconds_to_srt_time(s['start'])} --> "
                        f"{seconds_to_srt_time(s['end'])}\n{s['text'].strip()}\n\n")
        self.show_current_segment(jump_player=False)
        self.queue_status_label.setText(f"{len(results)}/{total}개 재번역 완료 및 저장됨.")
        
    def export_vocabulary_xlsx_for_current(self):
        """현재 재생 중인 영상의 단어장을 엑셀(.xlsx)로 저장."""
        if not HAS_OPENPYXL:
            self.queue_status_label.setText("openpyxl 미설치: pip install openpyxl 실행 후 다시 시도하세요.")
            return
        if not self.segments or not self.video_path:
            self.queue_status_label.setText("먼저 자막이 있는 영상을 재생하세요.")
            return

        rows, seen = [], set()
        for s in self.segments:
            for v in s.get('vocabulary', []):
                w, m = v.get('word', '').strip(), v.get('meaning', '').strip()
                if w and (w, m) not in seen:
                    seen.add((w, m))
                    rows.append((w, m, s.get('text', '')))
        if not rows:
            self.queue_status_label.setText("추출된 단어가 없습니다.")
            return

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "단어장"
        ws.append(["일본어 단어", "한국어 뜻", "예문(원문)"])
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in rows:
            ws.append(r)
        for i, w in enumerate([18, 28, 60], start=1):
            ws.column_dimensions[chr(64 + i)].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        out_path = os.path.splitext(self.player.video_path)[0] + "_단어장.xlsx"
        try:
            wb.save(out_path)
            self.queue_status_label.setText(f"저장 완료: {out_path} ({len(rows)}개 단어)")
        except Exception as e:
            self.queue_status_label.setText(f"저장 실패: {e}")

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        self.build_subtitle_menu(menu, with_exit=True)
        self.exec_subtitle_menu(menu, QCursor.pos())

    def toggle_mode(self):
        self.mode = "word" if self.mode == "sentence" else "sentence"
        self.show_current_segment()
        self.schedule_save_settings()

    # 👇 여기에 추가해주시면 됩니다!
    def change_long_vowel_style(self, new_style):
        self.long_vowel_style = new_style
        # 전역 변수 업데이트
        global LONG_VOWEL_STYLE
        LONG_VOWEL_STYLE = new_style
        
        self.schedule_save_settings()
        # 필요시 현재 화면의 발음 표기 갱신
        if self.segments:
            self._refresh_phonetics(self.segments)
            self.show_current_segment(jump_player=False)

    # ---------------- 자막 텍스트 갱신 (좌/우 패널 + 오버레이 동시 반영) ----------------
    def set_kanji_text(self, text):
        self.kanji_label.setText(text)
        self.subtitle_overlay.kanji_label.setText(text)
        if self.subtitle_display_mode == "overlay":
            self.update_overlay_geometry()

    def set_meaning_text(self, text):
        self.meaning_label.setText(text)
        self.subtitle_overlay.meaning_label.setText(text)
        if self.subtitle_display_mode == "overlay":
            self.update_overlay_geometry()

    # ---------------- 마우스: 창 이동 + 테두리 리사이즈 ----------------
    def get_resize_edge(self, pos):
        rect = self.rect()
        x, y = pos.x(), pos.y()
        m = self.RESIZE_MARGIN

        left = x <= m
        right = x >= rect.width() - m
        top = y <= m
        bottom = y >= rect.height() - m

        if top and left:
            return 'top_left'
        if top and right:
            return 'top_right'
        if bottom and left:
            return 'bottom_left'
        if bottom and right:
            return 'bottom_right'
        if left:
            return 'left'
        if right:
            return 'right'
        if top:
            return 'top'
        if bottom:
            return 'bottom'
        return None

    def update_cursor_for_edge(self, edge):
        cursor_map = {
            'left': Qt.SizeHorCursor, 'right': Qt.SizeHorCursor,
            'top': Qt.SizeVerCursor, 'bottom': Qt.SizeVerCursor,
            'top_left': Qt.SizeFDiagCursor, 'bottom_right': Qt.SizeFDiagCursor,
            'top_right': Qt.SizeBDiagCursor, 'bottom_left': Qt.SizeBDiagCursor,
        }
        self.setCursor(cursor_map.get(edge, Qt.ArrowCursor))

    def perform_resize(self, global_pos):
        delta = global_pos - self.resize_start_mouse
        geo = QRect(self.resize_start_geo)

        if 'left' in self.resize_edge:
            new_left = geo.left() + delta.x()
            if geo.right() - new_left >= self.MIN_WIDTH:
                geo.setLeft(new_left)
        if 'right' in self.resize_edge:
            new_right = geo.right() + delta.x()
            if new_right - geo.left() >= self.MIN_WIDTH:
                geo.setRight(new_right)
        if 'top' in self.resize_edge:
            new_top = geo.top() + delta.y()
            if geo.bottom() - new_top >= self.MIN_HEIGHT:
                geo.setTop(new_top)
        if 'bottom' in self.resize_edge:
            new_bottom = geo.bottom() + delta.y()
            if new_bottom - geo.top() >= self.MIN_HEIGHT:
                geo.setBottom(new_bottom)

        self.setGeometry(geo)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            edge = self.get_resize_edge(event.pos())
            if edge and not self.isMaximized():
                self.resize_edge = edge
                self.resize_start_geo = self.geometry()
                self.resize_start_mouse = event.globalPos()
                return
            # 타이틀바(최소화/최대화/닫기가 있는 상단 줄)의 빈 공간을 클릭했을
            # 때만 창 드래그 이동을 시작합니다. 버튼 위 클릭은 버튼이 먼저
            # 이벤트를 가로채므로 여기까지 오지 않습니다.
            if self.title_bar_widget.geometry().contains(event.pos()):
                self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.resize_edge and (event.buttons() & Qt.LeftButton):
            self.perform_resize(event.globalPos())
            return
        if self.old_pos and (event.buttons() & Qt.LeftButton):
            if self.isMaximized():
                # 최대화 상태에서 타이틀바를 드래그하면 창을 복원하고,
                # 복원된 창이 마우스 커서 아래에 오도록 위치를 맞춥니다.
                self.showNormal()
                self.btn_maximize.setText("☐")
                self.move(int(event.globalPos().x() - self.width() / 2), 0)
                self.old_pos = event.globalPos()
                return
            delta = event.globalPos() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()
            return

        # 버튼이 안 눌린 상태에서는 테두리 근처일 때만 커서 모양 갱신
        if not self.isMaximized():
            edge = self.get_resize_edge(event.pos())
            self.update_cursor_for_edge(edge)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = None
            self.resize_edge = None

    def moveEvent(self, event):
        super().moveEvent(event)
        self.update_overlay_geometry()
        self.schedule_save_settings()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_overlay_geometry()
        self.schedule_save_settings()
        
    def _update_window_mask(self):
        """창 전체 모서리를 아주 살짝 둥글게 (OS 창 자체를 이 모양으로 잘라냄)."""
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 8, 8)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self.update_overlay_geometry()
            # 최소화 → 복원 후에도 현재 모드의 '항상 위' 상태가 유지되도록 다시 적용
            QTimer.singleShot(0, lambda: self.apply_topmost_state(force=True))
        elif event.type() == QEvent.ActivationChange and self.isActiveWindow():
            # 영상 창을 클릭해 앞으로 나오면 자막이 그 뒤로 가려지지 않게 즉시 올린다
            self.bring_overlay_above()

    # ---------------- 설정 저장 / 불러오기 ----------------
    def load_settings(self):
        """subtitle_player_settings.json 에서 저장된 설정을 읽어 반영. 파일이 없거나
        일부 값이 잘못돼 있으면 그 항목만 기본값을 유지한다."""
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        installed_fonts = set(QFontDatabase().families())
        for key, check in SETTING_SPECS.items():
            if key not in data:
                continue
            try:
                value = check(data[key])
            except Exception:
                continue
            if key.endswith("_font_family") and value not in installed_fonts:
                continue   # 지금 PC에 없는 폰트는 무시 (기본 폰트 유지)
            setattr(self, key, value)

        g = data.get("window_geometry")
        if (isinstance(g, list) and len(g) == 4
                and all(isinstance(v, int) and not isinstance(v, bool) for v in g)):
            self.saved_window_geometry = g

        legacy_keys_file = os.path.join(_APP_DIR, "gemini_api_keys.json")
        if os.path.exists(legacy_keys_file):
            try:
                with open(legacy_keys_file, "r", encoding="utf-8") as f:
                    legacy = json.load(f)
                if isinstance(legacy, list):
                    self.gemini_api_keys = list(dict.fromkeys(
                        self.gemini_api_keys + [k for k in legacy if isinstance(k, str)]))
                os.remove(legacy_keys_file)   # 예전 파일은 이제 필요 없으니 삭제
            except Exception:
                pass
        set_api_keys(self.gemini_api_keys)
        
        self.saved_maximized = data.get("window_maximized") is True

    def save_settings(self):
        if not self._settings_ready:
            return
        data = {key: getattr(self, key, "") for key in SETTING_SPECS if hasattr(self, key)}
        try:
            g = self.normalGeometry()
            data["window_geometry"] = [g.x(), g.y(), g.width(), g.height()]
            data["window_maximized"] = self.isMaximized()
            tmp = SETTINGS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, SETTINGS_FILE)   # 쓰는 도중 꺼져도 기존 파일이 깨지지 않게 교체 방식으로 저장
        except Exception as e:
            print(f"설정 저장 실패: {e}")

    def schedule_save_settings(self):
        """슬라이더/드래그처럼 값이 연속으로 바뀌는 경우를 위해 0.6초 뒤 한 번만 저장."""
        if self._settings_ready:
            self._settings_timer.start()

    def restore_window_geometry(self):
        g = self.saved_window_geometry
        if not g:
            return
        x, y, w, h = g
        rect = QRect(x, y, max(w, self.MIN_WIDTH), max(h, self.MIN_HEIGHT))
        for screen in QApplication.screens():
            inter = rect.intersected(screen.availableGeometry())
            if inter.width() >= 120 and inter.height() >= 60:   # 지금 연결된 모니터 안에 있을 때만 복원
                self.setGeometry(rect)
                return

    # ---------------- 항상 위 (3단계 순환) ----------------
    def update_topmost_button(self):
        text, tip = TOPMOST_INFO[self.topmost_mode]
        self.btn_topmost.setText(text)
        self.btn_topmost.setToolTip(tip)

    def cycle_topmost_mode(self):
        i = TOPMOST_MODES.index(self.topmost_mode)
        self.topmost_mode = TOPMOST_MODES[(i + 1) % len(TOPMOST_MODES)]
        self.update_topmost_button()
        self.apply_topmost_state(force=True)
        self.schedule_save_settings()

    def _want_topmost(self):
        if self.topmost_mode == "always":
            return True
        if self.topmost_mode == "playing":
            try:
                return bool(self.media_player.is_playing())
            except Exception:
                return False
        return False

    def apply_topmost_state(self, force=False):
        """현재 모드(항상/재생 중에만/해제)와 재생 상태에 맞춰 창의 '항상 위'를 적용.
        타이머에서 계속 호출되지만 상태가 바뀔 때만 실제로 창 설정을 건드린다."""
        want = self._want_topmost()
        if not force and want == self._is_topmost:
            return
        self._is_topmost = want
        if not IS_WIN:
            return
        win_set_topmost(self, want)                      # 영상 창 먼저
        if hasattr(self, "subtitle_overlay") and self.subtitle_overlay.isVisible():
            win_set_topmost(self.subtitle_overlay, want)  # 오버레이는 나중에 → 영상 창 위에 놓인다
            self._overlay_topmost_applied = want
        self.bring_overlay_above()

    def bring_overlay_above(self):
        """오버레이 자막 창을 영상 창 바로 위로 올린다.
        - 항상 위 상태: 두 창 모두 최상위 그룹이므로 raise 하면 된다.
        - 항상 위가 아닐 때: 영상 창(또는 오버레이)이 맨 앞 창일 때만 올린다.
          (다른 프로그램을 쓰고 있을 때 자막만 그 위로 튀어나오지 않게)"""
        if (self.subtitle_display_mode != "overlay" or not self.subtitle_overlay.isVisible()
                or self._suspend_raise or self.isMinimized()):
            return
        if self._is_topmost or not IS_WIN or win_foreground_is(self, self.subtitle_overlay):
            self.subtitle_overlay.raise_()

    # ---------------- 콘솔(cmd) 창 숨기기/보이기 ----------------
    def set_console_hidden(self, hidden):
        if win_show_console(not hidden):
            self.hide_console = hidden
            self.schedule_save_settings()

    def apply_console_setting(self):
        if self.hide_console:
            win_show_console(False)

    # ---------------- 재생 목록 (같은 폴더의 비슷한 이름 영상) ----------------
    def _playlist_item_text(self, path):
        name = os.path.basename(path)
        if self.extra_folders:
            name = os.path.join(os.path.basename(os.path.dirname(path)), name)
        playing = self.video_path and norm_path(path) == norm_path(self.video_path)
        st = self.job_status.get(path)
        if st and st.get("state") == "running":
            pct = st.get("percent")
            icon = f"⚙ {pct}%" if pct is not None else "⚙"
        elif st and st.get("state") == "queued":
            icon = "⏳"
        elif st and st.get("state") == "failed":
            icon = "❌"
        elif os.path.exists(path + ".json"):
            icon = "✅"
        elif os.path.exists(os.path.splitext(path)[0] + ".srt"):
            icon = "📄"
        else:
            icon = "▫"
        return f"{'▶ ' if playing else '   '}{icon}  {name}"

    def refresh_playlist_item(self, path):
        item = self._playlist_items.get(path)
        if item is None:
            return
        item.setText(self._playlist_item_text(path))
        font = item.font()
        font.setBold(bool(self.video_path and norm_path(path) == norm_path(self.video_path)))
        item.setFont(font)
        st = self.job_status.get(path)
        item.setToolTip(f"{path}\n{st.get('text', '')}" if st else path)
    
    def _folder_videos(self):
        found = []
        for d in self.extra_folders:
            for root, _dirs, files in os.walk(d):
                for n in files:
                    if n.lower().endswith(VIDEO_EXTS):
                        found.append(norm_path(os.path.join(root, n)))
        return sorted(set(found), key=lambda p: natural_key(p))

    def add_folder_dialog(self):
        self._suspend_raise += 1
        try:
            d = QFileDialog.getExistingDirectory(self, "영상 폴더 선택", self.last_dir)
        finally:
            self._suspend_raise -= 1
        if not d:
            return
        d = norm_path(d)
        if d not in self.extra_folders:
            self.extra_folders.append(d)
        self.rebuild_playlist()
        self.schedule_save_settings()

    def rebuild_playlist(self):
        paths = scan_similar_videos(self.video_path, self.playlist_show_all) if self.video_path else []
        for p in self._folder_videos():          # 추가
            if p not in paths:                    # 추가
                paths.append(p)                   # 추가
        self.playlist_paths = list(paths)
        # 다른 폴더에서 자막 생성 대기열에 넣었거나 작업 중인 영상도 목록에 계속 보이게 한다
        for p, st in self.job_status.items():
            if p not in paths and st.get("state") in ("queued", "running", "failed"):
                paths.append(p)
        self.playlist_widget.clear()
        self._playlist_items = {}
        for p in paths:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, p)
            self.playlist_widget.addItem(item)
            self._playlist_items[p] = item
            self.refresh_playlist_item(p)

    def on_show_all_toggled(self, checked):
        self.playlist_show_all = checked
        self.rebuild_playlist()
        self.schedule_save_settings()

    def on_auto_next_toggled(self, checked):
        self.auto_next = checked
        self.schedule_save_settings()

    def toggle_playlist(self):
        self.playlist_visible = not self.playlist_visible
        self.apply_subtitle_layout(self.subtitle_display_mode)

    def on_playlist_item_activated(self, item):
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            self.load_video(path)

    def play_relative(self, delta):
        """재생 목록에서 이전(-1)/다음(+1) 영상으로 이동. 이동했으면 True."""
        paths = self.playlist_paths
        cur = norm_path(self.video_path) if self.video_path else ""
        normed = [norm_path(p) for p in paths]
        if cur not in normed:
            return False
        new_idx = normed.index(cur) + delta
        if 0 <= new_idx < len(paths):
            self.load_video(paths[new_idx])
            return True
        return False

    def open_video_dialog(self):
        self._suspend_raise += 1
        try:
            path, _ = QFileDialog.getOpenFileName(self, "영상 선택", self.last_dir, VIDEO_FILTER)
        finally:
            self._suspend_raise -= 1
        if path:
            self.load_video(path)

    def show_playlist_menu(self, pos):
        item = self.playlist_widget.itemAt(pos)
        if item is not None and not item.isSelected():
            self.playlist_widget.clearSelection()
            item.setSelected(True)
        selected = [i.data(Qt.UserRole) for i in self.playlist_widget.selectedItems()]
        if not selected:
            return

        menu = QMenu(self)
        menu.setStyleSheet(MENU_STYLE)
        if len(selected) == 1:
            play_act = QAction("▶ 재생", menu)
            play_act.triggered.connect(lambda checked=False, p=selected[0]: self.load_video(p))
            menu.addAction(play_act)
        gen_act = QAction(f"🎬 자막 생성 대기열에 추가 ({len(selected)}개)", menu)
        gen_act.triggered.connect(lambda checked=False, ps=selected: self.enqueue_subtitle_jobs(ps))
        menu.addAction(gen_act)
        rem_act = QAction("❌ 대기열에서 제거", menu)
        rem_act.triggered.connect(lambda checked=False, ps=selected: self.remove_from_queue(ps))
        menu.addAction(rem_act)
        self.exec_subtitle_menu(menu, self.playlist_widget.viewport().mapToGlobal(pos))

    # ---------------- 자막 생성 대기열 (백그라운드) ----------------
    def ensure_queue_worker(self):
        if self.queue_worker is None:
            w = SubtitleQueueWorker()
            w.job_started.connect(self.on_job_started)
            w.job_progress.connect(self.on_job_progress)
            w.job_finished.connect(self.on_job_finished)
            w.job_failed.connect(self.on_job_failed)
            w.queue_idle.connect(self.update_queue_status_label)
            self.queue_worker = w
            w.start(QThread.LowPriority)   # 영상 재생에 방해되지 않도록 낮은 우선순위
        return self.queue_worker

    def enqueue_subtitle_jobs(self, paths, front=False):
        """영상들을 자막 생성 대기열에 추가. 이미 자막(.json)이 있는 영상은 건너뛴다.
        front=True면 대기열 맨 앞(지금 보는 영상 등)에 넣는다."""
        worker = self.ensure_queue_worker()
        added, already = 0, 0
        need_rebuild = False
        ordered = list(reversed(paths)) if front else list(paths)
        for p in ordered:
            p = norm_path(p)
            if os.path.exists(p + ".json"):
                already += 1
                continue
            if front:
                worker.remove_pending(p)   # 이미 대기 중이면 맨 앞으로 옮기기 위해 한 번 뺀다
            if worker.enqueue(p, front=front):
                self.job_status[p] = {"state": "queued", "text": "대기 중"}
                added += 1
                if p in self._playlist_items:
                    self.refresh_playlist_item(p)
                else:
                    need_rebuild = True
        if need_rebuild:
            self.rebuild_playlist()
        self.update_queue_status_label()
        if not front:
            note = f"{added}개를 대기열에 추가했습니다."
            if already:
                note += f" (이미 자막이 있는 {already}개는 건너뜀)"
            self.queue_status_label.setText(note + "\n" + self.queue_status_label.text())
        return added, already

    def queue_selected_videos(self):
        selected = [i.data(Qt.UserRole) for i in self.playlist_widget.selectedItems()]
        if not selected:
            self.queue_status_label.setText("목록에서 영상을 선택하세요 (Ctrl / Shift 클릭으로 여러 개 선택)")
            return
        self.enqueue_subtitle_jobs(selected)

    def add_videos_to_queue_dialog(self):
        self._suspend_raise += 1
        try:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "자막을 생성할 영상 선택 (여러 개 선택 가능)", self.last_dir, VIDEO_FILTER)
        finally:
            self._suspend_raise -= 1
        if paths:
            self.last_dir = os.path.dirname(paths[0])
            self.enqueue_subtitle_jobs(sorted(paths, key=lambda p: natural_key(os.path.basename(p))))

    def remove_from_queue(self, paths):
        if self.queue_worker is None:
            return
        for p in paths:
            p = norm_path(p)
            if self.queue_worker.remove_pending(p):
                self.job_status.pop(p, None)
                self.refresh_playlist_item(p)
        self.update_queue_status_label()

    def clear_queue(self):
        if self.queue_worker is None:
            return
        for p in self.queue_worker.clear_pending():
            self.job_status.pop(p, None)
            self.refresh_playlist_item(p)
        self.update_queue_status_label()

    def cancel_current_job(self):
        if self.queue_worker is not None and self.queue_worker.current_path:
            self.queue_worker.cancel_current()
            self.queue_status_label.setText("현재 작업을 중지하는 중... (현재 처리 중인 단계가 끝나면 멈춥니다)")

    def update_queue_status_label(self):
        w = self.queue_worker
        lines = []
        if w is not None:
            cur = w.current_path
            if cur:
                st = self.job_status.get(cur, {})
                lines.append(f"⚙ 생성 중: {os.path.basename(cur)}")
                if st.get("text"):
                    lines.append(st["text"])
            pending = len(w.pending_paths())
            if pending:
                lines.append(f"⏳ 대기 {pending}개")
        self.queue_status_label.setText("\n".join(lines) if lines else "자막 생성 대기열: 없음")

    def _is_current_video(self, path):
        return bool(self.video_path) and norm_path(path) == norm_path(self.video_path)

    def on_job_started(self, path):
        st = self.job_status.setdefault(path, {})
        st.update(state="running", text="시작...", percent=None)
        self.refresh_playlist_item(path)
        self.update_queue_status_label()

    def on_job_progress(self, path, text):
        st = self.job_status.setdefault(path, {})
        st.update(state="running", text=text)
        m = re.search(r'(\d+)%', text)
        if m:
            st["percent"] = int(m.group(1))
        self.refresh_playlist_item(path)
        self.update_queue_status_label()
        # 지금 보고 있는 영상의 자막이 아직 없을 때만 진행 상황을 화면 자막 자리에 보여준다
        # (다른 영상 작업 진행 상황이 지금 영상의 자막을 덮어쓰지 않게)
        if self._is_current_video(path) and not self.segments:
            self.set_kanji_text(text)

    def on_job_finished(self, path, segments):
        self.job_status.pop(path, None)
        self.refresh_playlist_item(path)
        self.update_queue_status_label()
        if self._is_current_video(path) and not self.segments:
            self.on_transcription_finished(segments)

    def on_job_failed(self, path, message):
        if message == "취소됨":
            self.job_status.pop(path, None)
        else:
            self.job_status[path] = {"state": "failed", "text": message}
        self.refresh_playlist_item(path)
        self.update_queue_status_label()
        if self._is_current_video(path) and not self.segments:
            self.set_kanji_text(f"자막 생성 {'취소' if message == '취소됨' else '실패'}: {message}")

    # ---------------- 창 이벤트 / 종료 ----------------
    def showEvent(self, event):
        super().showEvent(event)
        self._update_window_mask()
        if not self._first_show_done:
            self._first_show_done = True
            QTimer.singleShot(0, lambda: self.apply_topmost_state(force=True))
            if self.saved_maximized:
                QTimer.singleShot(0, self.toggle_maximize)

    def eventFilter(self, obj, event):
        if obj is self.video_frame:
            if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self.toggle_play_pause()
                return True
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self.contextMenuEvent(event)   # 영상 위 우클릭 메뉴도 함께 살아납니다
                return True
            if event.type() in (QEvent.Resize, QEvent.Move):
                self.update_overlay_geometry()
        return super().eventFilter(obj, event)

    def shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.sync_timer.stop()
        self.subtitle_disable_timer.stop()
        self._settings_timer.stop()
        self.save_settings()
        try:
            self.subtitle_overlay.close()
        except Exception:
            pass
        try:
            self.media_player.stop()
        except Exception:
            pass
        if self.hide_console:
            win_show_console(True)   # 숨겨둔 콘솔은 종료 전에 되살려서, 안 보이는 cmd가 남지 않게 한다
        w = self.queue_worker
        if w is not None and w.isRunning():
            w.stop()
            # 작업이 바로 멈추지 않는 단계(모델 추론 중 등)에 있으면 3초 뒤 강제 종료
            t = threading.Timer(3.0, lambda: os._exit(0))
            t.daemon = True
            t.start()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()

    # ---------------- 자막/영상 로딩 ----------------
    def _refresh_phonetics(self, segments):
        """저장된 JSON의 발음 표기를 최신 변환기로 다시 만든다 (이전 버전에서 만든 자막도 바로 교정됨)."""
        for seg in segments:
            try:
                seg['phonetic'] = japanese_to_korean_phonetic(seg.get('text', ''))
            except Exception:
                pass

    def _on_rebuild_text(self, video_path, text):
        if self._is_current_video(video_path) and not self.segments:
            self.set_kanji_text(text)

    def _on_rebuild_done(self, video_path, segments):
        self.refresh_playlist_item(video_path)
        # 영상을 바꿨다면 이전 영상용 결과를 지금 영상에 넣지 않는다
        if self._is_current_video(video_path) and not self.segments:
            self.on_transcription_finished(segments)

    def load_video(self, video_path):
        video_path = norm_path(video_path)
        self.video_path = video_path
        self.last_dir = os.path.dirname(video_path)
        srt_path = os.path.splitext(video_path)[0] + ".srt"
        json_path = video_path + ".json"

        # 이전 영상의 자막이 남아 보이지 않도록 초기화
        self.segments = []
        self.current_index = 0
        self.set_kanji_text("")
        self.set_meaning_text("")

        # 자막 생성 여부와 무관하게 영상은 바로 재생 시작
        self.load_media(video_path)
        self.rebuild_playlist()
        self.schedule_save_settings()

        if os.path.exists(srt_path) and not os.path.exists(json_path):
            self.set_kanji_text("기존 SRT 자막을 바탕으로 JSON 데이터를 생성 중...")
            self._rebuild_workers = [w for w in self._rebuild_workers if w.isRunning()]
            worker = SrtRebuilderWorker(srt_path, json_path)
            worker.progress.connect(lambda t, vp=video_path: self._on_rebuild_text(vp, t))
            worker.finished.connect(lambda segs, vp=video_path: self._on_rebuild_done(vp, segs))
            self._rebuild_workers.append(worker)   # 실행 중 스레드가 사라지지 않게 참조 유지
            worker.start()
            return

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    segments = json.load(f)
                self._refresh_phonetics(segments)
                apply_corrections(segments, load_corrections())
                self.segments = segments
                self.current_index = 0
                if self.segments:
                    self.show_current_segment(jump_player=False)
                else:
                    self.set_kanji_text("자막 데이터가 비어 있습니다.")
            except Exception as e:
                self.set_kanji_text(f"JSON 로딩 실패: {e}")
            return

        # 자막이 전혀 없으면 지금 보는 영상을 대기열 맨 앞에 넣어 자동으로 생성한다
        self.set_kanji_text("자막 생성 대기 중... (영상은 바로 재생됩니다)")
        self.enqueue_subtitle_jobs([video_path], front=True)

    def update_status(self, text):
        self.set_kanji_text(text)

    def on_transcription_finished(self, segments):
        self.segments = segments
        self.current_index = 0
        total_segs = len(self.segments) if self.segments else 0

        if total_segs > 0:
            self.set_kanji_text(f"총 {total_segs}개 문장 로드 완료!")
            self.show_current_segment(jump_player=False)
        else:
            self.set_kanji_text("추출되거나 복구된 대사가 없습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "일본어 애니메이션 영상 선택",
        read_saved_last_dir(),
        VIDEO_FILTER
    )

    if file_path:
        player = AnimeSubtitlePlayer()
        # 반드시 show() 이후에 load_video를 호출해야 video_frame의
        # winId()가 유효한 네이티브 윈도우 핸들로 생성됩니다.
        player.show()
        player.load_video(file_path)
        sys.exit(app.exec_())
    else:
        print("영상이 선택되지 않았습니다.")