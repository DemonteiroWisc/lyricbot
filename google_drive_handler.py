import os
import re
import time
from datetime import datetime, timedelta, timezone
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

def title_to_video_filename(video_title, fallback="lyric_video_FINAL.mp4"):
    """
    Turns a YouTube title into an .mp4 filename that matches the title as closely
    as possible, so it needs as little editing as possible when uploading manually.
    Keeps the full title (including any "(Lyrics)" suffix) and only substitutes
    characters that are invalid in Windows/Drive filenames: < > : " / \\ | ? *
    """
    video_title = (video_title or "").strip()
    if not video_title:
        return fallback
    sanitized_title = re.sub(r'[\\/*?:"<>|]', '_', video_title)
    # Windows drops trailing dots/spaces before the extension, so strip them.
    sanitized_title = sanitized_title.rstrip(' .')
    if not sanitized_title:
        return fallback
    return f"{sanitized_title}.mp4"

def get_drive_service(config, get_authenticated_service_func):
    """Authenticates with Google Drive and returns the service object."""
    try:
        service = get_authenticated_service_func(
            credentials_file=config['CREDENTIALS_FILE'],
            client_secrets_file=config['CLIENT_SECRETS_FILE'],
            scopes=config['SCOPES'],
            service_name='drive',
            service_version='v3'
        )
        return service
    except Exception as e:
        print(f"  [CRITICAL AUTH ERROR] Failed to get Google Drive service: {e}")
        return None

def verify_drive_access(drive_service, parent_folder_id):
    """
    Verifies that the Google Drive service can access the parent folder.
    Returns the parent folder name on success, None on failure.
    """
    MAX_DRIVE_ATTEMPTS = 3
    for attempt in range(1, MAX_DRIVE_ATTEMPTS + 1):
        try:
            print(f"  Attempting to connect... (Attempt {attempt}/{MAX_DRIVE_ATTEMPTS})")
            parent_folder_file = drive_service.files().get(fileId=parent_folder_id, fields='name').execute()
            parent_folder_name = parent_folder_file.get('name', 'Unknown Folder')
            print(f"  -> Successfully connected to Google Drive.")
            print(f"  -> Parent folder '{parent_folder_name}' is accessible.")
            return parent_folder_name
        except HttpError as e:
            print(f"  [CRITICAL ACCESS ERROR] Could not access the specified Google Drive folder.")
            print(f"  -> Reason: {e.reason}")
            print(f"  -> Please ensure the GOOGLE_DRIVE_FOLDER_ID is correct and that you have granted the correct permissions.")
        except Exception as e:
            print(f"  [CRITICAL ACCESS ERROR] An unexpected error occurred during Google Drive verification: {e}")
        
        if attempt < MAX_DRIVE_ATTEMPTS:
            print("  -> Connection failed. Waiting 5 minutes before retrying...")
            time.sleep(300)
    return None

def create_run_folder(drive_service, parent_folder_id, folder_suffix=None):
    """
    Creates a new subfolder for the current run, named with the date.
    Avoids name collisions by adding a suffix if a folder for today already exists.
    Returns the new folder's ID and name, or (None, None) on failure.
    """
    try:
        base_name = datetime.now().strftime("%m-%d-%y")
        if folder_suffix:
            base_name = f"{base_name}_{folder_suffix}"

        # Query for folders that start with the base name to handle collisions
        query = f"'{parent_folder_id}' in parents and name starts with '{base_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        response = drive_service.files().list(q=query, fields="files(name)").execute()
        existing_folders = response.get('files', [])
        
        next_suffix = 1
        folder_name_to_create = base_name
        existing_names = {f['name'] for f in existing_folders}
        while folder_name_to_create in existing_names:
            next_suffix += 1
            folder_name_to_create = f"{base_name}_{next_suffix}"

        print(f"  -> Creating new run subfolder: '{folder_name_to_create}'")
        folder_metadata = {
            'name': folder_name_to_create, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_folder_id]
        }
        new_folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
        run_folder_id = new_folder.get('id')
        print(f"  -> Successfully created run subfolder (ID: {run_folder_id})")
        return run_folder_id, folder_name_to_create

    except HttpError as e:
        print(f"  [CRITICAL ERROR] Could not create Google Drive folder: {e}")
        print("  -> Please ensure the GOOGLE_DRIVE_FOLDER_ID is correct and you have permissions.")
        return None, None

def cleanup_archive(drive_service, parent_folder_id):
    """Deletes folders in the 'Archive' subfolder that are older than 30 days."""
    print("\n--- Checking Google Drive Archive for old folders ---")
    try:
        # 1. Find the 'Archive' folder within the parent folder.
        archive_query = f"'{parent_folder_id}' in parents and name='Archive' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        archive_response = drive_service.files().list(q=archive_query, fields="files(id, name)").execute()
        archive_folders = archive_response.get('files', [])

        if not archive_folders:
            print("  -> 'Archive' folder not found. Skipping cleanup.")
            return

        archive_folder_id = archive_folders[0]['id']
        print(f"  -> Found 'Archive' folder (ID: {archive_folder_id}).")

        # 2. List all folders inside the 'Archive' folder.
        subfolder_query = f"'{archive_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        subfolders_response = drive_service.files().list(q=subfolder_query, fields="files(id, name, createdTime)").execute()
        subfolders = subfolders_response.get('files', [])

        if not subfolders:
            print("  -> Archive is empty. No folders to clean up.")
            return

        # 3. Check each folder's age and delete if older than one month.
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        folders_deleted_count = 0
        for folder in subfolders:
            created_time_str = folder['createdTime']
            created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))

            if created_time < one_month_ago:
                print(f"     - Deleting old folder '{folder['name']}' (created on {created_time.date()})...")
                try:
                    drive_service.files().delete(fileId=folder['id']).execute()
                    folders_deleted_count += 1
                except HttpError as delete_error:
                    print(f"       [ERROR] Failed to delete folder '{folder['name']}': {delete_error}")
        
        if folders_deleted_count > 0:
            print(f"  -> Successfully deleted {folders_deleted_count} old folder(s) from the Archive.")
        else:
            print("  -> No folders in the Archive were old enough to be deleted.")
    except Exception as archive_exc:
        print(f"  [WARNING] An error occurred during archive cleanup: {archive_exc}")

def upload_assets_to_drive(service, video_file, thumbnail_file, description_file, title_file, run_folder_id, parent_folder_name, run_folder_name):
    """
    Uploads the generated video assets to a specified Google Drive folder.
    Returns True if all files are uploaded successfully, False otherwise.
    """
    print(f"  -> Uploading assets to Google Drive folder: '{parent_folder_name}/{run_folder_name}'")
    
    # 1. Name the video exactly like the YouTube title (from youtube_title.txt) so it
    #    needs as little editing as possible when uploading to YouTube manually.
    video_upload_name = "lyric_video_FINAL.mp4"  # Fallback name
    if os.path.exists(title_file):
        with open(title_file, 'r', encoding='utf-8') as f:
            video_upload_name = title_to_video_filename(f.read())
    
    files_to_upload = {
        "Video": (video_file, video_upload_name),
        "Thumbnail": (thumbnail_file, os.path.basename(thumbnail_file)),
        "Description": (description_file, os.path.basename(description_file)),
        "Title": (title_file, os.path.basename(title_file)),
    }

    all_successful = True

    for name, (path, upload_name) in files_to_upload.items():
        if os.path.exists(path):
            print(f"     Uploading {name} as '{upload_name}'...")
            file_metadata = {'name': upload_name, 'parents': [run_folder_id]}
            try:
                media = MediaFileUpload(path, resumable=True)
                service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                print(f"       ...Success.")
            except HttpError as e:
                print(f"       [ERROR] Failed to upload {name}: {e}")
                all_successful = False
        else:
            print(f"     Skipping {name} (file not found at '{path}')")
            all_successful = False # A missing asset means the upload is not complete.

    if not all_successful: print("  [WARNING] Not all assets were uploaded successfully to Google Drive.")
    return all_successful