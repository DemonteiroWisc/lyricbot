import json
import logging
import os

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from thefuzz import fuzz

import music_fetch
from thumbnail import _clean_song_name

logger = logging.getLogger("lyricbot")

# How many of a channel's most recent uploads to check each run. Cheap (one
# playlistItems page or two), and comfortably covers normal daily publish
# cadence even if a run gets skipped for a while.
RECENT_VIDEOS_TO_CHECK = 30

# Only exact title matches are applied automatically -- this runs
# unattended with no human review, so anything less than a perfect match is
# left alone rather than risking a wrong youtube_id getting attached.
MATCH_SCORE_THRESHOLD = 100


def _get_service_non_interactive(credentials_file, scopes, service_name, service_version):
    """
    Loads and, if needed, silently refreshes existing credentials for a
    read-only reconciliation call. Deliberately never falls back to the
    interactive browser consent flow (unlike lyricbot_core's
    get_authenticated_service) -- this runs unattended at the very start of
    every scheduled pipeline run, so a token that can't be refreshed
    silently must be skipped, not block forever waiting for a browser that
    will never come. Returns None if a working service can't be built.
    """
    if not os.path.exists(credentials_file):
        return None
    try:
        credentials = Credentials.from_authorized_user_file(credentials_file, scopes)
    except (ValueError, OSError):
        return None

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                # Persist the refreshed token, same as lyricbot_core's
                # get_authenticated_service -- otherwise the Drive
                # verification step later in this same run re-reads the
                # still-stale token from disk and redundantly refreshes it
                # again against Google's OAuth endpoint.
                with open(credentials_file, 'w') as token:
                    token.write(credentials.to_json())
            except Exception:
                return None
        else:
            return None

    try:
        return build(service_name, service_version, credentials=credentials)
    except Exception:
        return None


def _fetch_recent_uploads(youtube, handle, limit):
    ch_resp = youtube.channels().list(forHandle=handle, part='contentDetails').execute()
    items = ch_resp.get('items', [])
    if not items:
        return None
    uploads_playlist_id = items[0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    page_token = None
    while len(videos) < limit:
        resp = youtube.playlistItems().list(
            playlistId=uploads_playlist_id, part='snippet,status',
            maxResults=min(50, limit - len(videos)), pageToken=page_token,
        ).execute()
        for item in resp.get('items', []):
            # Only videos actually public count as "confirmed published" --
            # a scheduled-for-later or still-private upload sitting in the
            # uploads playlist must not permanently lock the song out via
            # music_fetch's "youtube_id set -> excluded forever" rule.
            if item.get('status', {}).get('privacyStatus') != 'public':
                continue
            snippet = item['snippet']
            videos.append({'title': snippet['title'], 'video_id': snippet['resourceId']['videoId']})
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return videos


def _best_match(video_title, candidates):
    video_title_lower = video_title.lower()
    best_score, best_entry = 0, None
    for entry in candidates:
        artist_name = entry.get('artist_name', '')
        song_name = entry.get('song_name', '')
        # Two variants on purpose: newly-generated titles always use the
        # cleaned name (thumbnail.create_youtube_title), but real uploads
        # can still carry the raw, uncleaned name -- e.g. older videos
        # titled with a "- 2012 Remaster" suffix that current title
        # generation would strip. When there's nothing to clean the two
        # strings are identical and the set collapses them for free.
        variants = {
            f"{artist_name} - {_clean_song_name(song_name)} (Lyrics)",
            f"{artist_name} - {song_name} (Lyrics)",
        }
        score = max(fuzz.ratio(video_title_lower, v.lower()) for v in variants)
        if score > best_score:
            best_score, best_entry = score, entry
    return best_entry, best_score


def backfill_youtube_ids(config):
    """
    Reconciles the shared used_songs.json log against a channel's real
    public YouTube uploads: any song logged as Drive-only that has since
    actually gone public gets its youtube_id/youtube_url filled in (via the
    same update-in-place path as music_fetch.log_used_song), which also
    restarts -- correctly ends -- its 90-day cooldown.

    Best-effort: any failure here (network, auth, missing channel) is
    caught and logged, never allowed to block the main pipeline run. Uses
    only a non-interactive credential load (see _get_service_non_interactive)
    so a stale/unrefreshable token is skipped rather than hanging the whole
    unattended run.
    """
    handle = config.get('YOUTUBE_HANDLE')
    if not handle:
        return

    print(f"\n--- Reconciling YouTube uploads for @{handle} against the used-songs log ---")
    try:
        youtube = _get_service_non_interactive(
            credentials_file=config['CREDENTIALS_FILE'],
            scopes=config['SCOPES'],
            service_name='youtube',
            service_version='v3',
        )
        if youtube is None:
            print(f"  [WARNING] No usable (non-interactive) YouTube credentials for @{handle}. Skipping reconciliation.")
            logger.warning("No usable non-interactive YouTube credentials for @%s; skipped reconciliation", handle)
            return

        videos = _fetch_recent_uploads(youtube, handle, RECENT_VIDEOS_TO_CHECK)
        if videos is None:
            print(f"  [WARNING] No channel found for handle @{handle}. Skipping reconciliation.")
            logger.warning("No channel found for handle @%s; skipped reconciliation", handle)
            return
    except Exception as e:
        print(f"  [WARNING] Could not fetch YouTube uploads for reconciliation: {e}")
        logger.warning("Could not fetch YouTube uploads for @%s: %s", handle, e)
        return

    try:
        with open(config['USED_SONGS_LOG'], 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    drive_only = [s for s in log_data if not s.get('youtube_id') and s.get('spotify_id')]

    updated = 0
    for video in videos:
        entry, score = _best_match(video['title'], drive_only)
        if entry and score >= MATCH_SCORE_THRESHOLD:
            music_fetch.log_used_song({'spotify_id': entry['spotify_id']}, video['video_id'], config['USED_SONGS_LOG'])
            drive_only.remove(entry)
            updated += 1

    print(f"  -> Reconciliation complete: {updated} song(s) confirmed newly published on YouTube.")
    logger.info("YouTube reconciliation for @%s: %d song(s) confirmed newly published", handle, updated)
