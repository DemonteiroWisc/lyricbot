# LyricBot

Automated lyric-video pipeline: finds a trending/manually-picked song, downloads audio, aligns lyrics with Whisper, renders a synced lyric video (via MoviePy), generates a thumbnail, and optionally uploads to YouTube and/or Google Drive.

The pipeline runs independently for three channels, each with its own look and destination:

| Channel | Entry point | Assets folder |
|---|---|---|
| Labs | `lyricbot_labs.py` / `RUN_LYRICBOT_LABS.bat` | `1_labs_assets/` |
| Vivid | `lyricbot_vivid.py` / `RUN_LYRICBOT_VIVID.bat` | `2_vivid_assets/` |
| Solara | `lyricbot_solara.py` / `RUN_LYRICBOT_SOLARA.bat` | `3_solara_assets/` |

Each channel script builds a `config` dict and calls into the shared workflow in `lyricbot_core.py`.

## Project layout

- `lyricbot_core.py` — shared workflow: YouTube auth/upload, run history, weekly recap, orchestration.
- `music_fetch.py` — song discovery (Billboard/Spotify), audio download (yt-dlp/ytmusicapi), lyrics fetch/sync, AI-upscaled backgrounds.
- `video_generator.py` — renders the synced lyric video with MoviePy.
- `thumbnail.py` — generates the YouTube thumbnail.
- `google_drive_handler.py` — Google Drive upload helper.
- `<n>_<channel>_assets/` — per-channel working directory: rendered output, credentials, run history (gitignored — see below).
- `assets/` — shared static assets (fonts, `used_songs.json` history, Google OAuth client secret).

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   `torch` is CUDA-build specific — install the right build for your GPU from [pytorch.org](https://pytorch.org/get-started/locally/) if the pinned version doesn't match your setup.

2. Copy `.env.example` to `.env` and fill in your API credentials (Spotify, Genius, Leonardo, HuggingFace):
   ```
   cp .env.example .env
   ```

3. Google OAuth: place your Google Cloud OAuth client secret at `assets/client_secrets.json`. The first run of each channel will open a browser to authorize and will cache the resulting token as `<channel>_assets/youtube_credentials_<channel>.json`.

## Running

```
python lyricbot_labs.py
python lyricbot_vivid.py
python lyricbot_solara.py
```

or use the corresponding `RUN_LYRICBOT_*.bat` on Windows.

Per-channel behavior (manual vs. auto song selection, upload targets, playlist IDs, etc.) is configured at the top of each `lyricbot_<channel>.py` file.

## Notes

- Rendered videos, lower-third overlays, run history, and credential files live under the per-channel asset folders and are intentionally **not** version-controlled (see `.gitignore`) — they're large binaries and per-run/per-machine state, not source.
- Never commit `.env`, `assets/client_secrets.json`, or any `youtube_credentials_*.json` — they contain live API credentials.
