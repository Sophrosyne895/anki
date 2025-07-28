"""
YouTube Playlist → Anki Flashcard Automation
===========================================

Watches a YouTube playlist for new videos, generates 5‑10 flashcards **plus a
summary card**, and uploads them to Anki via AnkiConnect.

* Python ≥ 3.9
* openai‑python ≥ 1.0   → `pip install "openai>=1.0"`
* youtube‑transcript‑api ≥ 1.2
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from openai import OpenAI
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ──────────────────────────────────────────────────────────────────────────────
# 🔧  Configuration
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
print("[debug] .env loaded → starting configuration")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
PLAYLIST_ID = os.getenv("PLAYLIST_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DECK_NAME = os.getenv("DECK_NAME", "YouTube Notes")
MIN_CARDS = int(os.getenv("MIN_CARDS", 5))
MAX_CARDS = int(os.getenv("MAX_CARDS", 10))
FETCH_INTERVAL = int(os.getenv("FETCH_INTERVAL", 3600))

print(
    "[debug] Config → "
    f"PLAYLIST_ID={PLAYLIST_ID}, DECK_NAME={DECK_NAME}, "
    f"MIN_CARDS={MIN_CARDS}, MAX_CARDS={MAX_CARDS}, FETCH_INTERVAL={FETCH_INTERVAL}s"
)

if not all([YOUTUBE_API_KEY, PLAYLIST_ID, OPENAI_API_KEY]):
    raise RuntimeError("Missing YOUTUBE_API_KEY, PLAYLIST_ID, or OPENAI_API_KEY in .env")

SEEN_FILE = Path("seen.json")
ANKI_URL = "http://localhost:8765"

# Create API clients
client = OpenAI(api_key=OPENAI_API_KEY)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
print("[debug] OpenAI and YouTube clients ready")

# ──────────────────────────────────────────────────────────────────────────────
# 🗂️  Persistence
# ──────────────────────────────────────────────────────────────────────────────

def load_seen() -> Set[str]:
    if SEEN_FILE.exists():
        seen = set(json.loads(SEEN_FILE.read_text()))
        print(f"[debug] Loaded {len(seen)} seen IDs")
        return seen
    return set()


def save_seen(seen: Set[str]) -> None:
    SEEN_FILE.write_text(json.dumps(sorted(seen)))
    print(f"[debug] Saved {len(seen)} seen IDs")

# ──────────────────────────────────────────────────────────────────────────────
# 🏷️  YouTube helpers
# ──────────────────────────────────────────────────────────────────────────────

def playlist_video_ids(pid: str) -> List[str]:
    vids: List[str] = []
    token: str | None = None
    while True:
        resp = (
            youtube.playlistItems()
            .list(part="contentDetails", playlistId=pid, maxResults=50, pageToken=token)
            .execute()
        )
        vids.extend(item["contentDetails"]["videoId"] for item in resp["items"])
        token = resp.get("nextPageToken")
        if not token:
            break
    print(f"[debug] playlist_video_ids → {len(vids)} IDs")
    return vids


def get_video_title(vid: str) -> str:
    resp = youtube.videos().list(part="snippet", id=vid, maxResults=1).execute()
    items = resp.get("items", [])
    return items[0]["snippet"]["title"] if items else "(unknown title)"

# ──────────────────────────────────────────────────────────────────────────────
# 📜  Transcript
# ──────────────────────────────────────────────────────────────────────────────

def fetch_transcript(vid: str) -> str | None:
    try:
        api = YouTubeTranscriptApi()
        segs = api.fetch(vid, languages=["en", "en-US", "en-GB"])
        text = " ".join(s.text for s in segs)
        print(f"[debug] Transcript fetched ({len(text.split())} words)")
        return text
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as err:
        print("[warn] Transcript unavailable:", err)
        return None

# ──────────────────────────────────────────────────────────────────────────────
# 🧠  GPT helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_flashcards(text: str, n: int) -> List[Dict[str, str]]:
    """Generate detailed flash‑cards.

    * Cuts the transcript to ~6,000 characters to stay within token limits.
    * Asks GPT for rich answers (3–6 sentences).
    * Sanitises markdown fences and extracts the JSON array.
    """
    # Trim huge transcripts so the prompt + completion fit comfortably.
    text_snippet = text[:6000]

    system_msg = (
        "You are a helpful assistant that converts transcripts into rich, detailed Anki flashcards. "
        "Each *question* should stand on its own and not reference 'the speaker' or 'the video'. For example instead of the question being: what did the video say about Lithuania? have the question be: What is unique about Lithuania in the 1400s?"
        "Each *answer* should be 3–6 sentences long with context or examples. "
        "Respond with a pure JSON array — no code fences."
    )
    user_msg = (
        f"Create {n} Q&A flashcards from the text below. Each object must have 'question' and 'answer' keys."
        f"TEXT:{text_snippet}"
    )

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3,
            max_tokens=1200,
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
        import re, json as _json
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            raise ValueError("No JSON array found in model output")
        cards = _json.loads(match.group(0))
        print(f"[debug] Parsed {len(cards)} flashcards from model output")
        return cards
    except Exception as err:
        print("[warn] Failed to generate flashcards:", err)
        return []


def generate_summary(text: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Summarize in one concise paragraph."}, {"role": "user", "content": text}],
        temperature=0.3,
        max_tokens=150,
    )
    return res.choices[0].message.content.strip()

# ──────────────────────────────────────────────────────────────────────────────
# 🃏  Push to Anki
# ──────────────────────────────────────────────────────────────────────────────

def anki_add_notes(deck: str, cards: List[Dict[str, str]]) -> None:
    """Add *new* notes to Anki, skipping duplicates deterministically.

    We first call AnkiConnect's `canAddNotes` to let Anki itself tell us which
    cards are duplicates. Only notes that return *True* are sent in a second
    `addNotes` request.  This prevents a single‑duplicate from failing the
    entire batch and keeps your deck clean without manual intervention.
    """
    # Build full note payload once so we can reuse it for both calls.
    notes_payload = [
        {
            "deckName": deck,
            "modelName": "Basic",
            "fields": {"Front": c["question"], "Back": c["answer"]},
            "options": {"allowDuplicate": False},
            "tags": ["youtube-auto"],
        }
        for c in cards
    ]

    # Ask Anki which ones are safe to add
    can_req = {
        "action": "canAddNotes",
        "version": 6,
        "params": {"notes": notes_payload},
    }
    can_resp = requests.post(ANKI_URL, json=can_req, timeout=30).json()
    if can_resp.get("error"):
        print("[anki] canAddNotes error:", can_resp)
        return

    mask = can_resp["result"]  # list[bool]
    to_add = [n for n, ok in zip(notes_payload, mask) if ok]
    skipped = len(notes_payload) - len(to_add)

    if not to_add:
        print(f"[info] All {skipped} notes were duplicates; nothing to add")
        return

    add_req = {
        "action": "addNotes",
        "version": 6,
        "params": {"notes": to_add},
    }
    add_resp = requests.post(ANKI_URL, json=add_req, timeout=30).json()
    if add_resp.get("error"):
        print("[anki] addNotes error:", add_resp)
    else:
        print(f"[debug] Added {len(to_add)} notes, skipped {skipped} duplicates")

# ──────────────────────────────────────────────────────────────────────────────
# 🔁  Core processing
# ──────────────────────────────────────────────────────────────────────────────

def process_new_videos() -> None:
    seen = load_seen()
    ids = playlist_video_ids(PLAYLIST_ID)
    new_ids = [v for v in ids if v not in seen]
    print(f"[debug] Found {len(new_ids)} new video(s)")
    if not new_ids:
        return

    for vid in new_ids:
        print(f"[info] Processing {vid} …")
        transcript = fetch_transcript(vid)
        if transcript is None:
            continue

        n_cards = max(MIN_CARDS, min(MAX_CARDS, len(transcript.split()) // 150))
        qa_cards = generate_flashcards(transcript, n_cards)

        title = get_video_title(vid)
        summary = generate_summary(transcript)
        link = f"https://www.youtube.com/watch?v={vid}"
        summary_card = {
            "question": f"{title} – Summary",
            "answer": f"{summary}<br><br><a href=\"{link}\">{link}</a>",
        }

        anki_add_notes(DECK_NAME, qa_cards + [summary_card])
        seen.add(vid)
        save_seen(seen)

# ──────────────────────────────────────────────────────────────────────────────
# 🚀  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Continuous watcher loop with Ctrl+C exit."""
    print("▶ YouTube → Anki watcher started. Press Ctrl+C to stop.")
    cycle = 0
    try:
        while True:
            cycle += 1
            ts = datetime.now().isoformat(timespec="seconds")
            print(f"[debug] ===== Cycle {cycle} • {ts} =====")
            try:
                process_new_videos()
            except Exception as exc:  # noqa: BLE001
                print("[error] Unhandled exception:", exc)
            print(f"[debug] Sleeping {FETCH_INTERVAL}s…")
            time.sleep(FETCH_INTERVAL)
    except KeyboardInterrupt:
        print("[info] Stopped by user. Goodbye!")


if __name__ == "__main__":
    main()
