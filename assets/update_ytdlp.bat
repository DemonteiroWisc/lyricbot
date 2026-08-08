@echo off
echo "Checking for package updates..."
python -m pip install --upgrade yt-dlp lyricsgenius spotipy syncedlyrics ytmusicapi
echo "Update check finished."