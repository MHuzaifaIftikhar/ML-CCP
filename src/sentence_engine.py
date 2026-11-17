"""
=============================================================================
SENTENCE SUGGESTION & TEXT-TO-SPEECH ENGINE (Lab 14 CCA Enhancement)
Project: ASL Sign Language to English Translator
Course: Machine Learning (AI-3202) - Complex Computing Activity
Dawood University of Engineering & Technology, Batch 2023
=============================================================================
Provides:
  1. SpeechEngine: Non-blocking, offline Windows Text-to-Speech synthesizer
     running on a background worker thread so webcam FPS stays smooth.
  2. SentenceSuggestionEngine: Smart context-aware, shorthand-aware, and
     fuzzy-matched sentence suggestion engine with optional async Gemini API
     integration (activated automatically if GEMINI_API_KEY is configured).
"""

import difflib
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 1. NON-BLOCKING OFFLINE TEXT-TO-SPEECH ENGINE
# =============================================================================
class SpeechEngine:
    """
    Asynchronous Text-to-Speech engine using Windows built-in System.Speech
    (zero extra pip dependencies, completely offline, non-blocking).
    """

    def __init__(self, rate: int = 0):
        self.rate = rate
        self._queue: queue.Queue[str] = queue.Queue()
        self.is_speaking: bool = False
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def speak(self, text: str) -> None:
        """Queues text to be spoken aloud without blocking the UI loop."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._queue.put(cleaned)

    def _speak_sync(self, text: str) -> None:
        """Speaks text synchronously inside the background worker thread."""
        safe_text = text.replace("'", "''").replace('"', " ")
        if sys.platform.startswith("win"):
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$synth.Rate = {int(self.rate)}; "
                f"$synth.Speak('{safe_text}')"
            )
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                creationflags=creation_flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.is_speaking = True
                self._speak_sync(text)
            except Exception as exc:
                print(f"[!] Speech synthesis warning: {exc}")
            finally:
                self.is_speaking = False
                self._queue.task_done()

    def close(self) -> None:
        self._stop_event.set()


# =============================================================================
# 2. SMART CONVERSATIONAL SENTENCE BANK & ASL SHORTHAND EXPANSIONS
# =============================================================================
SENTENCE_BANK: Dict[str, List[str]] = {
    "A": ["Assalam-o-Alaikum!", "Are you okay?", "All is well, thank you."],
    "B": ["Be right back.", "Bye, take care!", "Bring some water please."],
    "C": ["Can you help me please?", "Call me later.", "Can we start now?"],
    "D": ["Do you understand?", "Don't worry about it.", "Drink some water please."],
    "E": ["Excuse me, please.", "Everything is fine.", "Excuse me, where is the lab?"],
    "F": ["Fine, thank you!", "Follow me please.", "Feel better soon."],
    "G": ["Good morning!", "Good job!", "Go ahead please."],
    "H": ["Hello, how are you?", "Help me please!", "Have a nice day!"],
    "I": ["I need help please.", "I am fine, thank you.", "I understand completely."],
    "J": ["Just a moment please.", "Join us!", "Job well done."],
    "K": ["Keep going!", "Kindly help me.", "Know what I mean?"],
    "L": ["Let me think.", "Look at this.", "Listen carefully please."],
    "M": ["My name is Huzaifa.", "May I ask a question?", "Meet me outside."],
    "N": ["Nice to meet you!", "No problem at all.", "Need any help?"],
    "O": ["Okay, I got it.", "Of course!", "Open the door please."],
    "P": ["Please wait a moment.", "Please speak slowly.", "Perfect, thank you!"],
    "Q": ["Quick question please.", "Quiet please.", "Quite good!"],
    "R": ["Really appreciate it!", "Read this please.", "Right away!"],
    "S": ["Sorry, my mistake.", "See you soon!", "Stay safe!"],
    "T": ["Thank you so much!", "Take care of yourself.", "Tell me more."],
    "U": ["Understand completely.", "Use this one please.", "Up to you."],
    "V": ["Very good!", "Very nice to meet you.", "Verify this please."],
    "W": ["Welcome!", "What is your name?", "Where are you going?"],
    "X": ["Excellent work!", "Excuse me, one second.", "Extra help is needed."],
    "Y": ["Yes, I agree.", "You are welcome!", "Yes, please!"],
    "Z": ["Zero worries!", "Zoom in please.", "Zest for life!"],
}

# Common ASL shorthand / consonant-skeleton glosses -> full English sentences
SHORTHAND_MAP: Dict[str, List[str]] = {
    "HW": ["How are you doing?", "How can I help you?", "How was your day?"],
    "TY": ["Thank you so much!", "Thank you for listening.", "Thank you for your help."],
    "THX": ["Thank you so much!", "Thanks a lot!", "Thank you very much!"],
    "GM": ["Good morning!", "Good morning everyone!", "Good morning, how are you?"],
    "GA": ["Good afternoon!", "Good afternoon everyone!"],
    "GE": ["Good evening!", "Good evening, how are you?"],
    "GN": ["Good night, take care!", "Good night, see you tomorrow."],
    "GB": ["Goodbye, see you soon!", "Goodbye, take care!"],
    "PLZ": ["Please help me.", "Please wait a moment.", "Please speak slowly."],
    "PLS": ["Please help me.", "Please wait a moment.", "Please repeat that."],
    "WTR": ["Can I have some water please?", "I need water please.", "Bring some water please."],
    "HLP": ["Please help me!", "I need help right now.", "Can you help me please?"],
    "SRY": ["Sorry about that!", "Sorry I am late.", "Sorry, my mistake."],
    "NCE": ["Nice to meet you!", "Nice work!", "Have a nice day!"],
    "OK": ["Okay, I got it.", "Okay, no problem!", "Are you okay?"],
    "NM": ["My name is Huzaifa.", "What is your name?", "Nice to meet you!"],
    "IDK": ["I don't know yet.", "I don't understand, please repeat."],
    "BRB": ["Be right back.", "Just a moment, I'll be right back."],
    "NP": ["No problem at all!", "No problem, you are welcome."],
    "YW": ["You are welcome!", "You are very welcome!"],
    "DUET": ["I am a student at DUET.", "Welcome to DUET AI department!"],
    "ML": ["This is our Machine Learning project.", "I love Machine Learning!"],
}

CONVERSATIONAL_PHRASES: List[str] = [
    "Assalam-o-Alaikum, how are you?",
    "Hello everyone!",
    "Hello, nice to meet you!",
    "How are you doing today?",
    "Help is on the way.",
    "I am doing great, thank you!",
    "I am a student at DUET.",
    "I love machine learning.",
    "I need a glass of water please.",
    "I don't understand, could you repeat?",
    "This is our Machine Learning project.",
    "Thank you for listening to our presentation.",
    "Thank you for your time!",
    "That is awesome!",
    "Good afternoon everyone!",
    "Good evening!",
    "Good night, see you tomorrow!",
    "Goodbye, take care!",
    "Please repeat that.",
    "Please give me a moment.",
    "Project demo is ready.",
    "Can we start the presentation now?",
    "Can I ask you a question?",
    "Can you show me the way?",
    "See you tomorrow!",
    "Sorry I am late.",
    "What time is it?",
    "What do you think?",
    "Where is the classroom?",
    "Where is the nearest hospital?",
    "Yes, that is completely correct.",
    "No, thank you very much.",
    "My name is Huzaifa Iftikhar.",
    "We built this with SVM and KNN.",
    "Water please, I am thirsty.",
    "Call a doctor please.",
    "I am hungry, let's eat.",
    "Let's go together.",
    "Wait for me please.",
    "Excuse me, may I come in?",
]


def _load_gemini_key_from_env_file() -> Optional[str]:
    """Loads GEMINI_API_KEY from project-root .env if not already in os.environ."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "GEMINI_API_KEY" and v.strip():
                            key_val = v.strip().strip('"').strip("'")
                            os.environ["GEMINI_API_KEY"] = key_val
                            return key_val
        except Exception:
            pass
    return None


def _is_subsequence(short: str, full: str) -> bool:
    """Returns True if every char of `short` appears in `full` in order."""
    it = iter(full)
    return all(ch in it for ch in short)


class SentenceSuggestionEngine:
    """
    Context-aware, shorthand-aware, and fuzzy-matched sentence suggester
    with optional background Gemini API enhancement when GEMINI_API_KEY is set.
    """

    def __init__(self, use_gemini: bool = True):
        self.committed_words: List[str] = []
        self.current_word: str = ""

        # Build deduplicated phrase corpus
        self._all_phrases: List[str] = []
        seen = set()
        for letter in sorted(SENTENCE_BANK.keys()):
            for phrase in SENTENCE_BANK[letter]:
                if phrase not in seen:
                    self._all_phrases.append(phrase)
                    seen.add(phrase)
        for group in SHORTHAND_MAP.values():
            for phrase in group:
                if phrase not in seen:
                    self._all_phrases.append(phrase)
                    seen.add(phrase)
        for phrase in CONVERSATIONAL_PHRASES:
            if phrase not in seen:
                self._all_phrases.append(phrase)
                seen.add(phrase)

        # Optional Gemini API async enhancement (only if package & key exist)
        self.gemini_enabled: bool = False
        self._gemini_client = None
        self._gemini_cache: Dict[str, List[str]] = {}
        self._pending_key: Optional[str] = None
        self._last_gemini_call: float = 0.0
        self._lock = threading.Lock()

        api_key = _load_gemini_key_from_env_file() if use_gemini else None
        if use_gemini and api_key:
            try:
                from google import genai  # type: ignore
                self._gemini_client = genai.Client(api_key=api_key)
                self.gemini_enabled = True
                print("[+] Gemini API enabled for smart background sentence suggestions.")
            except Exception as exc:
                print(f"[*] Gemini API not active ({exc}); using smart offline engine.")

    def add_letter(self, letter: str) -> None:
        if letter and letter.isalpha() and len(letter) == 1:
            self.current_word += letter.upper()

    def add_space(self) -> None:
        if self.current_word:
            self.committed_words.append(self.current_word)
            self.current_word = ""

    def backspace(self) -> None:
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.committed_words:
            self.current_word = self.committed_words.pop()

    def clear(self) -> None:
        self.committed_words.clear()
        self.current_word = ""

    def get_full_text(self) -> str:
        parts = list(self.committed_words)
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)

    def _trigger_gemini_async(self, query_key: str, full_text: str, active_letter: Optional[str]) -> None:
        if not self.gemini_enabled or self._gemini_client is None:
            return
        with self._lock:
            if query_key in self._gemini_cache or self._pending_key == query_key:
                return
            self._pending_key = query_key

        def _worker():
            try:
                prompt = (
                    "You are an assistive ASL (American Sign Language) communicator. "
                    f"The user has signed: '{full_text}' and is currently showing sign '{active_letter or ''}'. "
                    "Suggest exactly 3 short, natural, everyday English sentences they likely mean. "
                    "Return ONLY 3 lines, no numbering, no bullets, max 8 words each."
                )
                resp = self._gemini_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                )
                lines = [ln.strip(" -•1234567890.\t") for ln in (resp.text or "").splitlines() if ln.strip()]
                if lines:
                    with self._lock:
                        self._gemini_cache[query_key] = lines[:3]
            except Exception:
                pass
            finally:
                with self._lock:
                    if self._pending_key == query_key:
                        self._pending_key = None

        threading.Thread(target=_worker, daemon=True).start()

    def get_suggestions(self, active_letter: Optional[str] = None) -> List[str]:
        """
        Returns 3 ranked sentence suggestions using:
          1. Gemini API cache (if enabled & available),
          2. ASL shorthand / gloss expansion (e.g., TY, WTR, HW, GM),
          3. Multi-word context & prefix matching,
          4. Subsequence / fuzzy matching over the conversational corpus,
          5. Symbol-triggered bank for `active_letter`.
        """
        full_text = self.get_full_text().strip().upper()
        prefix = self.current_word.strip().upper()
        query_key = f"{full_text}|{active_letter or ''}"

        # Check Gemini cache / trigger async refresh for multi-letter or multi-word input
        if self.gemini_enabled and (len(full_text) >= 2 or active_letter):
            with self._lock:
                cached = self._gemini_cache.get(query_key)
            if cached:
                return cached[:3]
            self._trigger_gemini_async(query_key, full_text, active_letter)

        scored: List[Tuple[float, str]] = []

        # 1. Exact shorthand match (e.g., "TY", "WTR", "HW")
        if prefix in SHORTHAND_MAP:
            for idx, phrase in enumerate(SHORTHAND_MAP[prefix]):
                scored.append((100.0 - idx, phrase))

        # 2. Score all phrases in corpus if user has typed any letters/words
        if full_text:
            tokens = full_text.split()
            for phrase in self._all_phrases:
                p_up = phrase.upper()
                score = 0.0

                # Exact full-sentence prefix match
                if p_up.startswith(full_text):
                    score += 80.0
                # Active word prefix match at start of any word in phrase
                elif prefix and any(w.startswith(prefix) for w in p_up.split()):
                    score += 55.0
                # Multi-word context overlap
                if len(tokens) > 1:
                    matched_tokens = sum(1 for tok in tokens[:-1] if tok in p_up.split())
                    score += matched_tokens * 25.0
                # Consonant/letter subsequence match (e.g. "HLP" inside "HELP ME PLEASE")
                if prefix and len(prefix) >= 2 and _is_subsequence(prefix, p_up.replace(" ", "")):
                    score += 32.0
                # Fuzzy similarity on the first word
                if prefix and len(prefix) >= 2:
                    first_word = p_up.split()[0].strip(".,!?")
                    ratio = difflib.SequenceMatcher(None, prefix, first_word).ratio()
                    if ratio >= 0.55:
                        score += ratio * 30.0

                if score > 0:
                    scored.append((score, phrase))

        # 3. Symbol-triggered suggestions for the currently held hand sign
        if active_letter and active_letter.upper() in SENTENCE_BANK:
            for idx, phrase in enumerate(SENTENCE_BANK[active_letter.upper()]):
                base = 20.0 - idx if full_text else 60.0 - idx
                scored.append((base, phrase))

        # Sort descending by score and deduplicate
        scored.sort(key=lambda x: x[0], reverse=True)
        result: List[str] = []
        seen = set()
        for _, phrase in scored:
            if phrase not in seen:
                result.append(phrase)
                seen.add(phrase)
                if len(result) == 3:
                    return result

        for fallback in ["Hello, how are you?", "Thank you so much!", "My name is Huzaifa."]:
            if fallback not in seen:
                result.append(fallback)
                seen.add(fallback)
                if len(result) == 3:
                    break
        return result[:3]

    def select_suggestion(self, index: int, active_letter: Optional[str] = None) -> Optional[str]:
        suggestions = self.get_suggestions(active_letter)
        if 0 <= index < len(suggestions):
            chosen = suggestions[index]
            self.committed_words = [chosen]
            self.current_word = ""
            return chosen
        return None
