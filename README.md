# YouTube Playlist → Anki Flashcard Automation
Turn any playlist into spaced-repetition notes in one click.

I wanted every new video in my research playlist to land in Anki with GPT-generated flashcards. This script watches a playlist, pulls the transcript, drafts 5-10 rich Q&A cards plus a summary card, and uploads them to Anki via AnkiConnect

Flow:
Polls the playlist on a schedule (FETCH_INTERVAL).

Uses youtube-transcript-api for captions (falls back to Whisper optional).

GPT-4o-mini writes detailed answers (3–6 sentences).

Duplicate cards automatically skipped via canAddNotes.

One summary card per video (paragraph + title + link).

| Var                     | What it does                         |
| ----------------------- | ------------------------------------ |
| `YOUTUBE_API_KEY`       | Google API key for YouTube Data v3   |
| `OPENAI_API_KEY`        | OpenAI token                         |
| `PLAYLIST_ID`           | ID after `list=` in playlist URL     |
| `DECK_NAME`             | Destination Anki deck (auto-created) |
| `MIN_CARDS / MAX_CARDS` | Boundaries for GPT                   |
| `FETCH_INTERVAL`        | Seconds between checks               |


Roadmap / TODO

Whisper fallback for no-caption videos

Multiple playlists mapped to different decks

Dockerfile / systemd unit

Slack webhook when new cards added

