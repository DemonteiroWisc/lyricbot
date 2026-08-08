import sys
import time
import traceback
import logging
from datetime import datetime, timedelta, timezone
import os
import glob
import json
import re
import shutil
import torch
import music_fetch
import video_generator
import google_drive_handler
import thumbnail
import pipeline_logging
import youtube_backfill
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

logger = logging.getLogger("lyricbot")

def get_authenticated_service(credentials_file, client_secrets_file, scopes, service_name, service_version):
    """
    Authenticates with a Google API and returns the service object.
    Handles token refresh and creation.
    """
    credentials = None
    if os.path.exists(credentials_file):
        credentials = Credentials.from_authorized_user_file(credentials_file, scopes)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            credentials = flow.run_local_server(port=0)
        with open(credentials_file, 'w') as token:
            token.write(credentials.to_json())
    
    return build(service_name, service_version, credentials=credentials)

def add_video_to_playlist(service, video_id, playlist_id):
    """Adds a video to a specified YouTube playlist."""
    if not playlist_id:
        print("  -> No YOUTUBE_PLAYLIST_ID is set. Skipping.")
        return
        
    print(f"  Adding video to playlist: {playlist_id}")
    try:
        request_body = {
            'snippet': {
                'playlistId': playlist_id,
                'resourceId': {
                    'kind': 'youtube#video',
                    'videoId': video_id
                }
            }
        }
        
        response = service.playlistItems().insert(
            part='snippet',
            body=request_body
        ).execute()
        
        print(f"  -> Video successfully added to playlist.")

    except HttpError as e:
        if e.resp.status == 403:
             print("  [ERROR] Could not add video to playlist: Permission denied.")
             print("  -> Please ensure the authenticated user owns or has manager access to the playlist.")
        else:
            print(f"  [ERROR] An HTTP error occurred while adding to playlist: {e.resp.status} - {e.content}")
    except Exception as e:
        print(f"  [ERROR] An unexpected error occurred while adding to playlist: {e}")

def upload_video_to_youtube(service, video_path, description_path, thumbnail_path, title, privacy_status='private', publish_at_datetime=None):
    """
    Uploads a video to YouTube, with an option to schedule it for a future date.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    if not os.path.exists(description_path):
        print(f"[ERROR] Description file not found: {description_path}. Using empty description.")
        video_description = ""
    else:
        with open(description_path, 'r', encoding='utf-8') as f:
            video_description = f.read()

    print(f"\n--- Preparing to upload '{title}' ---")
    
    body = {
        'snippet': {
            'title': title,
            'description': video_description,
            'tags': title.split() + ['lyrics', 'lyric video', 'music'],
            'categoryId': '10' # Music category
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }

    if publish_at_datetime:
        body['status']['privacyStatus'] = 'private'
        publish_time_iso = publish_at_datetime.isoformat()
        body['status']['publishAt'] = publish_time_iso
        print(f"  -> This video will be scheduled to go public at: {publish_time_iso}")
    else:
        print(f"  -> This video will be uploaded with status: {privacy_status}")

    try:
        media_body = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        insert_request = service.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media_body
        )

        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status:
                print(f"  Uploaded {int(status.progress() * 100)}%")

        video_id = response['id']
        print(f"  Video uploaded successfully! Video ID: {video_id}")

        if os.path.exists(thumbnail_path):
            print(f"  Setting thumbnail: {thumbnail_path}")
            # --- FIX: Add a retry loop for thumbnail upload to handle processing delays ---
            for i in range(3): # Try up to 3 times
                try:
                    time.sleep(5 * i) # Wait 0s, then 5s, then 10s
                    media_thumbnail = MediaFileUpload(thumbnail_path, mimetype='image/png')
                    service.thumbnails().set(
                        videoId=video_id,
                        media_body=media_thumbnail
                    ).execute()
                    print("  -> Thumbnail set successfully.")
                    break # Exit loop on success
                except HttpError as e:
                    print(f"  [WARNING] Attempt {i+1} to set thumbnail failed: {e.resp.status}")
                    if i == 2: # If it's the last attempt
                        print("  [ERROR] Final attempt to set thumbnail failed. Please check if your YouTube channel is verified: https://www.youtube.com/verify")
                        print(f"  Full error: {e.content.decode('utf-8')}")
            # --- END OF FIX ---
        else:
            print(f"  [WARNING] Thumbnail file not found: {thumbnail_path}. Skipping thumbnail upload.")

        print(f"  YouTube URL: https://www.youtube.com/watch?v={video_id}")
        return video_id

    except HttpError as e:
        print(f"  An HTTP error occurred: {e.resp.status} - {e.content}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during YouTube upload: {e}")
        return None

def cleanup_temp_files(config):
    """
    Removes ALL temporary assets from previous runs to ensure a clean slate.
    """
    print("--- Cleaning up all temporary files from previous runs ---")
    files_deleted = 0
    folders_deleted = 0

    patterns_to_delete = [
        os.path.join(config['ASSET_FOLDER'], "test_song*.*"),
        os.path.join(config['ASSET_FOLDER'], "lyrics_candidate_*.lrc"),
        config['VIDEO_FILE'],
        config['DESCRIPTION_FILE'],
        config['THUMBNAIL_FILE'],
        os.path.join(config['ASSET_FOLDER'], "demucs_separated_*")
    ]

    for pattern in patterns_to_delete:
        items = glob.glob(pattern)
        for item_path in items:
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"  Deleted file: {os.path.basename(item_path)}")
                    files_deleted += 1
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  Deleted folder: {os.path.basename(item_path)}")
                    folders_deleted += 1
            except OSError as e:
                print(f"  [ERROR] Could not delete {item_path}: {e}")
    
    if files_deleted == 0 and folders_deleted == 0:
        print("  -> No temporary files or folders to clean up.")
    print("-" * 40)

# ── Run history / weekly recap ───────────────────────────────────────────────
# Mirrors PinterestBot's scripts/run_pipeline.py: each run is logged, and the
# first run of the day on Sunday prints a 7-day recap.

HISTORY_KEEP_DAYS = 60


def _history_file(config):
    # Test runs (IS_TEST_RUN) get their own history file so they don't
    # pollute the prod weekly recap even though they share ASSET_FOLDER.
    name = 'run_history.test.json' if config.get('IS_TEST_RUN') else 'run_history.json'
    return os.path.join(config['ASSET_FOLDER'], name)


def _load_history(config):
    path = _history_file(config)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(config, records):
    with open(_history_file(config), 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _append_record(config, record):
    records = _load_history(config)
    records.append(record)
    cutoff = (datetime.now() - timedelta(days=HISTORY_KEEP_DAYS)).strftime("%Y-%m-%d")
    records = [r for r in records if r.get("date", "") >= cutoff]
    _save_history(config, records)


def _count_today_records(config):
    today = datetime.now().strftime("%Y-%m-%d")
    return sum(1 for r in _load_history(config) if r.get("date") == today)


def _print_weekly_recap(config):
    channel = config.get('GOOGLE_DRIVE_FOLDER_SUFFIX') or config.get('ASSET_FOLDER', 'LyricBot')
    records = _load_history(config)
    cutoff  = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    recent  = [r for r in records if r.get("date", "") >= cutoff]

    dates      = sorted({r["date"] for r in recent}) if recent else []
    date_range = f"{dates[0]}  ->  {dates[-1]}" if dates else "no data"

    print("\n" + "=" * 58)
    print(f"  WEEKLY RECAP — {channel}  |  {date_range}")
    print("=" * 58)

    if not recent:
        print("  No runs in the past 7 days.")
    else:
        by_date = {}
        for r in recent:
            by_date.setdefault(r["date"], []).append(r)

        for date in sorted(by_date.keys(), reverse=True):
            runs  = by_date[date]
            ok    = sum(1 for r in runs if r.get("status") == "success")
            fail  = len(runs) - ok
            label = f"{ok} uploaded" + (f"  {fail} failed" if fail else "")
            _dt = datetime.strptime(date, "%Y-%m-%d")
            day = _dt.strftime("%a %b ") + str(_dt.day)
            print(f"\n  {day}  --  {label}")
            for r in runs:
                ts       = r.get("timestamp", "")
                time_str = ts[11:16] if len(ts) >= 16 else "??:??"
                if r.get("status") == "success":
                    print(f"    {time_str}  {r.get('song') or '—'}")
                else:
                    err = (r.get("error") or "unknown error")[:50]
                    print(f"    {time_str}  FAILED        {err}")

    total_ok   = sum(1 for r in recent if r.get("status") == "success")
    total_fail = len(recent) - total_ok
    print("\n" + "-" * 58)
    fail_str = f"  |  {total_fail} failed" if total_fail else ""
    print(f"  Total:   {total_ok} uploaded{fail_str}")
    print("=" * 58 + "\n")


# ── Resume / checkpoint ───────────────────────────────────────────────────
# Protects the two costliest failure points: the paid Leonardo background
# generation and the network-dependent Drive/YouTube upload. Once a song has
# passed its quality check and its background image is generated, a
# checkpoint is saved; if the process dies before the upload finishes, the
# next run resumes straight into rendering + uploading instead of redoing
# song discovery, audio download, and the Whisper bake-off. A resume attempt
# is one-shot: the checkpoint is cleared after it's attempted (success or
# failure) so a bad checkpoint can't wedge future runs indefinitely.

CHECKPOINT_MAX_AGE_HOURS = 24


def _checkpoint_file(config):
    # Test runs (IS_TEST_RUN) get their own checkpoint file so a stale prod
    # checkpoint can't get resumed by a test run (wrong scopes/credentials)
    # or vice versa, even though both share ASSET_FOLDER.
    name = 'pipeline_state.test.json' if config.get('IS_TEST_RUN') else 'pipeline_state.json'
    return os.path.join(config['ASSET_FOLDER'], name)


def _save_checkpoint(config, song_info, best_overall_result, background_image_path, audio_paths):
    try:
        data = {
            'saved_at': datetime.now().isoformat(),
            'song_info': song_info,
            'best_overall_result': best_overall_result,
            'background_image_path': background_image_path,
            'audio_paths': audio_paths,
        }
        with open(_checkpoint_file(config), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Checkpoint saved for %s - %s", song_info.get('artist'), song_info.get('name'))
    except (TypeError, OSError) as e:
        print(f"  [WARNING] Could not save resume checkpoint: {e}")
        logger.warning("Could not save resume checkpoint: %s", e)


def _load_checkpoint(config):
    path = _checkpoint_file(config)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        saved_at = datetime.fromisoformat(data['saved_at'])
        if datetime.now() - saved_at > timedelta(hours=CHECKPOINT_MAX_AGE_HOURS):
            print(f"  [INFO] Ignoring resume checkpoint from {saved_at} (older than {CHECKPOINT_MAX_AGE_HOURS}h).")
            _clear_checkpoint(config)
            return None
        result = data['best_overall_result']
        referenced_paths = [
            data['background_image_path'],
            result['lrc_path'],
            result['original_audio_path'],
            result['winning_audio_path'],
            result['winning_vocal_path'],
        ] + data['audio_paths']
        if not all(os.path.exists(p) for p in referenced_paths):
            print("  [INFO] Ignoring resume checkpoint: referenced files no longer exist.")
            _clear_checkpoint(config)
            return None
        return data
    except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError):
        _clear_checkpoint(config)
        return None


def _clear_checkpoint(config):
    path = _checkpoint_file(config)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _finish_and_upload_song(config, song_info, best_overall_result, background_image_path,
                             audio_paths, drive_service, drive_parent_folder_name,
                             schedule_start_time, song_index, whisper_model):
    """
    Renders the final video for a confirmed song, generates the thumbnail and
    description, uploads to Drive/YouTube, logs the song as used, and cleans
    up bake-off assets. Returns (upload_successful, schedule_start_time).
    """
    run_folder_id, drive_run_folder_name = None, None
    i = song_index

    print("\nSTEP C: Generating the lyric video...")
    logger.info("Rendering video for %s - %s", song_info.get('artist'), song_info.get('name'))
    video_generator.generate_video_from_winner(best_overall_result, background_image_path, config['VIDEO_FILE'], whisper_model, lower_third_video_path=config['LOWER_THIRD_VIDEO'])

    winning_lrc_path = best_overall_result['lrc_path']
    destination_lrc_path = os.path.join(config['ASSET_FOLDER'], "test_lyrics.lrc")
    # --- FIX: Only copy the file if the source and destination are not the same ---
    if os.path.abspath(winning_lrc_path) != os.path.abspath(destination_lrc_path):
        shutil.copyfile(winning_lrc_path, destination_lrc_path)
        print(f"\n--- Copied winning lyric file to '{destination_lrc_path}' for description generation. ---")
    # --- END OF FIX ---

    print("\nSTEP D: Generating the thumbnail and YouTube description...")
    thumbnail.generate_thumbnail(song_info['name'], song_info['artist'], asset_folder=config['ASSET_FOLDER'], channel_name=config.get('GOOGLE_DRIVE_FOLDER_SUFFIX', 'Lyric Labs'))
    logger.info("Thumbnail and description generated")

    # --- Read the final, cleaned title from the generated text file ---
    title_file_path = os.path.join(config['ASSET_FOLDER'], "youtube_title.txt")
    full_title = "Generated Lyric Video" # Fallback title
    if os.path.exists(title_file_path):
        with open(title_file_path, 'r', encoding='utf-8') as f:
            full_title = f.read().strip()

    # --- Rename the local video file to match the YouTube title ---
    # The video was generated as 'lyric_video_FINAL.mp4'; now that the title
    # exists, rename it so it matches youtube_title.txt as closely as possible
    # (minimal editing when uploading manually). We use a local variable rather
    # than mutating config['VIDEO_FILE'], so the next song in an ALBUM loop still
    # generates into the stable 'lyric_video_FINAL.mp4' path.
    final_video_path = config['VIDEO_FILE']
    if os.path.exists(config['VIDEO_FILE']):
        title_based_name = google_drive_handler.title_to_video_filename(full_title)
        title_based_path = os.path.join(config['ASSET_FOLDER'], title_based_name)
        if os.path.abspath(title_based_path) != os.path.abspath(config['VIDEO_FILE']):
            try:
                os.replace(config['VIDEO_FILE'], title_based_path)
                final_video_path = title_based_path
                print(f"\n--- Renamed video file to '{title_based_name}' ---")
            except OSError as e:
                print(f"\n[WARNING] Could not rename video file to '{title_based_name}': {e}")

    # --- Trigger Google Drive folder creation only if access was verified and it's the first upload ---
    if drive_service and drive_parent_folder_name:
        # Check if the folder has been created for this run yet.
        if not run_folder_id:
            print("\n--- Preparing Google Drive Run Folder (First Upload) ---")
            parent_folder_id = config.get('GOOGLE_DRIVE_FOLDER_ID')
            # --- FIX: Pass the folder suffix from the config to the creation function ---
            folder_suffix = config.get('GOOGLE_DRIVE_FOLDER_SUFFIX')
            run_folder_id, drive_run_folder_name = google_drive_handler.create_run_folder(
                drive_service, parent_folder_id, folder_suffix
            )
            if not run_folder_id:
                # Folder creation failed, so run the archive cleanup as a fallback task
                google_drive_handler.cleanup_archive(drive_service, parent_folder_id)

    drive_upload_successful = False # Initialize flag
    # Now, with the folder confirmed (or failed), attempt the upload.
    if run_folder_id:
        print("\nSTEP E: Uploading assets to Google Drive...")
        drive_upload_successful = google_drive_handler.upload_assets_to_drive(
            service=drive_service, video_file=final_video_path, thumbnail_file=config['THUMBNAIL_FILE'],
            description_file=config['DESCRIPTION_FILE'], title_file=title_file_path,
            run_folder_id=run_folder_id, parent_folder_name=drive_parent_folder_name,
            run_folder_name=drive_run_folder_name
        )
        logger.info("Google Drive upload %s", "succeeded" if drive_upload_successful else "failed")

    youtube_upload_successful = False
    youtube_video_id = None # Initialize to None
    if config['UPLOAD_TO_YOUTUBE']:
        print("\nSTEP E: Uploading video to YouTube...")
        if schedule_start_time is None: schedule_start_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        publish_time = schedule_start_time + timedelta(minutes=i * 30)

        youtube_service = get_authenticated_service(
            credentials_file=config['CREDENTIALS_FILE'],
            client_secrets_file=config['CLIENT_SECRETS_FILE'],
            scopes=config['SCOPES'],
            service_name='youtube',
            service_version='v3'
        )
        youtube_video_id = upload_video_to_youtube( # The 'title' argument is now read from the file
            service=youtube_service, video_path=final_video_path,
            description_path=config['DESCRIPTION_FILE'], thumbnail_path=config['THUMBNAIL_FILE'],
            title=full_title, publish_at_datetime=publish_time
        )
        if youtube_video_id: youtube_upload_successful = True
        logger.info("YouTube upload %s", "succeeded" if youtube_upload_successful else "failed")

    # --- Log the song only if an upload was fully successful ---
    # Wrapped so a bookkeeping failure here (e.g. a locked/full-disk write to
    # USED_SONGS_LOG) can't propagate and prevent the checkpoint from being
    # cleared below — that would make the next run re-upload a song that has
    # already gone out.
    if youtube_upload_successful or drive_upload_successful:
        try:
            music_fetch.log_used_song(song_info, youtube_video_id, config['USED_SONGS_LOG'])
        except Exception as e:
            print(f"  [WARNING] Upload succeeded but failed to log the song as used: {e}")
            logger.warning("Upload succeeded but log_used_song failed: %s", e)

        # The video is now safely stored on Drive/YouTube -- delete the local
        # copy so successfully-uploaded videos don't pile up in the asset
        # folder indefinitely.
        try:
            if os.path.exists(final_video_path):
                os.remove(final_video_path)
                print(f"  -> Deleted local video file after successful upload: {os.path.basename(final_video_path)}")
                logger.info("Deleted local video file after successful upload: %s", os.path.basename(final_video_path))
        except OSError as e:
            print(f"  [WARNING] Could not delete local video file after upload: {e}")
            logger.warning("Could not delete local video file after upload: %s", e)

    # --- Add to playlist only if it was a successful YouTube upload ---
    if youtube_video_id:
        add_video_to_playlist(youtube_service, youtube_video_id, config['YOUTUBE_PLAYLIST_ID'])
    elif config['UPLOAD_TO_YOUTUBE']: # This means upload was enabled but failed
        print("\nYouTube upload failed. Skipping playlist addition.")
    else: # This means YouTube upload was never attempted
        print("\nSTEP E: YouTube upload skipped (UPLOAD_TO_YOUTUBE is False).")

    video_generator.cleanup_bakeoff_assets(all_audio_paths=audio_paths, winning_audio_path=best_overall_result['original_audio_path'], asset_folder=config['ASSET_FOLDER'])

    return (youtube_upload_successful or drive_upload_successful), schedule_start_time


def run_workflow_and_exit(config):
    """
    Runs the workflow and records it for the weekly recap.

    Returns a process exit code:
        0 = success
        1 = failure (keeps the .bat window open so the error stays visible)
        2 = success on the first run of Sunday (recap printed — keeps the
            .bat window open to review it)
    """
    pipeline_logging.setup_logging(config)
    logger.info("=== Run started (%s) ===", config.get('GOOGLE_DRIVE_FOLDER_SUFFIX') or config.get('ASSET_FOLDER'))

    try:
        youtube_backfill.backfill_youtube_ids(config)
    except Exception as e:
        print(f"  [WARNING] YouTube reconciliation step failed: {e}")
        logger.warning("YouTube reconciliation step failed: %s", e)

    run_started_at    = datetime.now()
    is_sunday          = run_started_at.weekday() == 6
    is_first_run_today = _count_today_records(config) == 0

    record = {
        "timestamp": run_started_at.isoformat(),
        "date":      run_started_at.strftime("%Y-%m-%d"),
    }

    try:
        success, detail = run_workflow(config)
    except Exception as exc:
        traceback.print_exc()
        logger.exception("Run failed with an unhandled exception")
        record["status"] = "failed"
        record["error"]  = str(exc)
        _append_record(config, record)
        if is_sunday and is_first_run_today:
            _print_weekly_recap(config)
        return 1

    if success:
        record["status"] = "success"
        record["song"]   = detail
        logger.info("Run finished successfully: %s", detail)
    else:
        record["status"] = "failed"
        record["error"]  = detail
        logger.warning("Run finished without success: %s", detail)
    _append_record(config, record)

    if is_sunday and is_first_run_today:
        _print_weekly_recap(config)
        return 2 if success else 1

    return 0 if success else 1


def _verify_drive_access(config):
    """
    Verifies Google Drive access at the start of a run, but does not create
    any folders yet. Returns (drive_service, drive_parent_folder_name),
    either of which may be None if Drive uploads are disabled or access
    could not be verified.
    """
    drive_service = None
    drive_parent_folder_name = None
    if config.get('UPLOAD_TO_GOOGLE_DRIVE', False):
        print("\n--- Verifying Google Drive Access ---")
        parent_folder_id = config.get('GOOGLE_DRIVE_FOLDER_ID')
        if not parent_folder_id or parent_folder_id == "YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE":
            print("  [ERROR] GOOGLE_DRIVE_FOLDER_ID is not set. Google Drive uploads will be skipped.")
        else:
            drive_service = google_drive_handler.get_drive_service(config, get_authenticated_service)
            if drive_service:
                drive_parent_folder_name = google_drive_handler.verify_drive_access(drive_service, parent_folder_id)
        print("-" * 40)
    return drive_service, drive_parent_folder_name


def _resume_from_checkpoint(config, checkpoint):
    """
    Resumes an interrupted run: skips song discovery, audio download, and the
    Whisper bake-off, and jumps straight to rendering + uploading using the
    song/background/audio captured in the checkpoint. One-shot: the
    checkpoint is cleared once this attempt finishes, win or lose.
    """
    song_info = checkpoint['song_info']
    print(f"\n--- RESUMING interrupted run for: {song_info['artist']} - {song_info['name']} ---")
    logger.info("Resuming checkpoint for %s - %s", song_info['artist'], song_info['name'])

    drive_service, drive_parent_folder_name = _verify_drive_access(config)

    device = "cuda" if torch.cuda.is_available() and video_generator.USE_GPU else "cpu"
    print(f"\n--- Loading Whisper Model ({video_generator.WHISPER_MODEL}) to {device.upper()} ---")
    whisper_model = video_generator.whisper.load_model(video_generator.WHISPER_MODEL, device=device)
    print("--- Whisper Model Loaded ---")

    try:
        _finish_and_upload_song(
            config, song_info, checkpoint['best_overall_result'],
            checkpoint['background_image_path'], checkpoint['audio_paths'],
            drive_service, drive_parent_folder_name,
            schedule_start_time=None, song_index=0, whisper_model=whisper_model
        )
    finally:
        # One-shot resume: clear regardless of outcome so a bad checkpoint
        # can't wedge every future run.
        _clear_checkpoint(config)

    return True, f"{song_info['artist']} - {song_info['name']}"


def run_workflow(config):
    """
    The main workflow, now driven by a configuration dictionary.
    Returns (success: bool, detail: str | None) — detail is the winning
    song's "Artist - Title" on success, or a failure reason on failure.
    """
    print("--- Starting LyricBot Master Workflow ---")

    checkpoint = _load_checkpoint(config)
    if checkpoint:
        return _resume_from_checkpoint(config, checkpoint)

    cleanup_temp_files(config)

    drive_service, drive_parent_folder_name = _verify_drive_access(config)

    MAX_QUALITY_CHECK_ATTEMPTS = 5
    successful_song_found = False
    
    for attempt in range(1, MAX_QUALITY_CHECK_ATTEMPTS + 1):
        print("\n" + "="*60)
        print(f"--- Main Loop Attempt {attempt}/{MAX_QUALITY_CHECK_ATTEMPTS} ---")
        print("="*60)

        songs_to_process = []
        
        if config['PROCESSING_MODE'] == 'ALBUM' and config['USE_EXISTING_LRC']:
            print("\n[CONFIG WARNING] USE_EXISTING_LRC is set to True, but it is not supported in 'ALBUM' mode. It will be ignored.")

        if config['PROCESSING_MODE'] == 'SINGLE':
            music_fetch.GENERATE_NEW_BACKGROUND = config['GENERATE_NEW_BACKGROUND']
            song_info = None
            if config['USE_VIRAL_SONG_FINDER']:
                print(f"\n--- Finding Viral Song ---")
                song_info = music_fetch.get_viral_ytmusic_song(config['USED_SONGS_LOG'], only_find_english_songs=config['ONLY_FIND_ENGLISH_SONGS'])
            else: # Manual mode
                song_info = music_fetch.get_song_details_from_spotify(
                    config['MANUAL_SPOTIFY_URL'],
                    music_fetch.SPOTIFY_CLIENT_ID,
                    music_fetch.SPOTIFY_CLIENT_SECRET
                )

            if not song_info:
                print("Could not find a song to process. Ending attempt.")
                time.sleep(10)
                continue 

            print(f"\n--- Validating assets for: {song_info['artist']} - {song_info['name']} ---")
            logger.info("Candidate song: %s - %s", song_info['artist'], song_info['name'])

            if config['USE_EXISTING_LRC']:
                local_lrc_path = os.path.join(config['ASSET_FOLDER'], "test_lyrics.lrc")
                if not os.path.exists(local_lrc_path): print(f"  [ERROR] The file '{local_lrc_path}' was not found. Aborting."); raise RuntimeError(f"Required file not found: {local_lrc_path}")
                
                # --- FIX: Only download audio and lyrics. Defer background generation. ---
                music_fetch.GENERATE_NEW_BACKGROUND = False # Temporarily disable
                audio_paths = music_fetch.download_song_audio(f"{song_info['name']} {song_info['artist']}", config['ASSET_FOLDER'], num_to_download=config['NUM_AUDIO_CANDIDATES'], is_explicit=song_info.get('is_explicit', False))
                lrc_paths = [local_lrc_path] if audio_paths else []
            else:
                 # --- FIX: Defer background generation until after quality check ---
                 music_fetch.GENERATE_NEW_BACKGROUND = False # Temporarily disable
                 lrc_paths, audio_paths = music_fetch.fetch_all_assets(song_info, config['NUM_AUDIO_CANDIDATES'], asset_folder=config['ASSET_FOLDER'])

            if not lrc_paths or not audio_paths:
                print(f"\n[VALIDATION FAILED] Could not fetch all required assets. Skipping song.")
                continue
            
            songs_to_process.append((song_info, lrc_paths, audio_paths))

        elif config['PROCESSING_MODE'] == 'ALBUM':
            album_tracks = music_fetch.get_album_tracks_from_spotify(config['MANUAL_ALBUM_URL'], music_fetch.SPOTIFY_CLIENT_ID, music_fetch.SPOTIFY_CLIENT_SECRET)
            for track_info in album_tracks:
                songs_to_process.append((track_info, None, None))
            if not album_tracks: break

        schedule_start_time = None
        total_songs = len(songs_to_process)
        for i, (song_info, pre_fetched_lrcs, pre_fetched_audios) in enumerate(songs_to_process):
            print("\n" + "="*60); print(f"--- Processing Song {i+1}/{total_songs}: {song_info['artist']} - {song_info['name']} ---"); print("="*60)
            
            lrc_paths, audio_paths = pre_fetched_lrcs, pre_fetched_audios
            if config['PROCESSING_MODE'] == 'ALBUM':
                music_fetch.GENERATE_NEW_BACKGROUND = False # Defer background generation
                if not lrc_paths and not audio_paths:
                    lrc_paths, audio_paths = music_fetch.fetch_all_assets(song_info, config['NUM_AUDIO_CANDIDATES'], asset_folder=config['ASSET_FOLDER'])
                    if not lrc_paths or not audio_paths:
                        print(f"\nAsset fetching failed for '{song_info['name']}'. Skipping."); continue

            background_image_path = os.path.join(config['ASSET_FOLDER'], "test_background.jpg")

            # --- FIX: Load the Whisper model ONCE before the bake-off ---
            device = "cuda" if torch.cuda.is_available() and video_generator.USE_GPU else "cpu"
            print(f"\n--- Loading Whisper Model ({video_generator.WHISPER_MODEL}) to {device.upper()} ---")
            whisper_model = video_generator.whisper.load_model(video_generator.WHISPER_MODEL, device=device)
            print("--- Whisper Model Loaded ---")

            print("\nSTEP B: Starting Master Bake-Off Across All Audio Sources...")
            best_overall_result = None
            lowest_overall_score = 101.0
            for audio_path in audio_paths:
                current_result = video_generator.find_best_alignment(song_info, lrc_paths, audio_path, asset_folder=config['ASSET_FOLDER'], model=whisper_model)
                if current_result and current_result['fallback_percentage'] < lowest_overall_score:
                    lowest_overall_score = current_result['fallback_percentage']
                    best_overall_result = current_result

            if not best_overall_result:
                print("\n[CRITICAL] Master bake-off failed to find any valid alignment. Skipping song."); continue

            fallback_percentage = best_overall_result['fallback_percentage']
            quality_check_passed = not (config['PROCESSING_MODE'] == 'SINGLE' and config['USE_VIRAL_SONG_FINDER'] and fallback_percentage > 40.0)

            # --- NEW: FALLBACK AUDIO DOWNLOAD LOGIC ---
            if not quality_check_passed:
                print(f"\n--- QUALITY CHECK FAILED (Fallback: {fallback_percentage:.2f}%) ---")
                print("--- Attempting fallback: Downloading one more audio candidate. ---")

                # Determine the index for the next audio file to download
                next_audio_index = len(audio_paths) + 1
                
                # Download just the next search result from YouTube
                fallback_audio_paths = music_fetch.download_song_audio(
                    f"{song_info['name']} {song_info['artist']}",
                    config['ASSET_FOLDER'],
                    num_to_download=next_audio_index,
                    start_index=next_audio_index,
                    is_explicit=song_info.get('is_explicit', False)
                )

                if fallback_audio_paths:
                    new_audio_path = fallback_audio_paths[0]
                    print(f"\n--- Running bake-off on new audio candidate: {os.path.basename(new_audio_path)} ---")
                    fallback_result = video_generator.find_best_alignment(song_info, lrc_paths, new_audio_path, asset_folder=config['ASSET_FOLDER'], model=whisper_model)
                    
                    # If the new result is better than the previous best, it becomes the winner
                    if fallback_result and fallback_result['fallback_percentage'] < best_overall_result['fallback_percentage']:
                        print("  -> Fallback audio provided a better result. Promoting it to winner.")
                        best_overall_result = fallback_result
                        # Add the new path to the list for cleanup purposes
                        audio_paths.append(new_audio_path)

            # Re-evaluate the quality check after the potential fallback
            fallback_percentage = best_overall_result['fallback_percentage']
            if config['PROCESSING_MODE'] == 'SINGLE' and config['USE_VIRAL_SONG_FINDER'] and fallback_percentage > 40.0:
                print(f"\n--- QUALITY CHECK FAILED (Fallback: {fallback_percentage:.2f}%) ---")
                print("--- Finding a new song. ---")
                video_generator.cleanup_bakeoff_assets(all_audio_paths=audio_paths, winning_audio_path=best_overall_result['original_audio_path'], asset_folder=config['ASSET_FOLDER'])
                break
            
            print("\n--- QUALITY CHECK PASSED ---")
            logger.info("Quality check passed for %s - %s (fallback %.2f%%)", song_info['artist'], song_info['name'], fallback_percentage)
            successful_song_found = True

            # --- FIX: Generate the background image HERE, only after a song is confirmed. ---
            background_image_path = os.path.join(config['ASSET_FOLDER'], "test_background.jpg")
            if config['GENERATE_NEW_BACKGROUND']:
                print("\n--- Generating background image for confirmed song... ---")
                if not music_fetch.generate_background_image(background_image_path):
                    print("\n[CRITICAL] Background generation failed. Cannot proceed with video creation.")
                    logger.error("Background generation failed for %s - %s", song_info['artist'], song_info['name'])
                    successful_song_found = False
                    break # Exit the song processing loop
            # --- END OF FIX ---

            # Everything expensive up to here (song discovery, audio download,
            # Whisper bake-off, and — most importantly — the paid Leonardo
            # background generation) is done. Checkpoint before the
            # render/upload steps so a crash there doesn't waste it.
            _save_checkpoint(config, song_info, best_overall_result, background_image_path, audio_paths)

            _, schedule_start_time = _finish_and_upload_song(
                config, song_info, best_overall_result, background_image_path, audio_paths,
                drive_service, drive_parent_folder_name, schedule_start_time, i, whisper_model
            )
            _clear_checkpoint(config)

        if successful_song_found:
            break
            
    if not successful_song_found:
        detail = f"Could not find a suitable song after {MAX_QUALITY_CHECK_ATTEMPTS} attempts."
        print(f"\n--- LyricBot workflow failed: {detail} ---")
        return False, detail
    else:
        print("\n--- LyricBot Workflow Finished Successfully! ---")
        song_label = f"{song_info['artist']} - {song_info['name']}" if 'song_info' in locals() and song_info else None
        return True, song_label