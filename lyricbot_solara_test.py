import os
import lyricbot_core

def main():
    # --- Define all settings for this specific controller ---
    ASSET_FOLDER = "3_solara_assets"
    
    config = {
        # --- Channel-Specific Paths ---
        "ASSET_FOLDER": ASSET_FOLDER,
        "LOWER_THIRD_VIDEO": os.path.join(ASSET_FOLDER, "sub_lower_third_solara.mov"),
        
        # --- Processing Mode ---
        "PROCESSING_MODE": 'SINGLE',  # 'SINGLE' or 'ALBUM'
        
        # --- Download & Bake-off Options ---
        "NUM_AUDIO_CANDIDATES": 2,
        
        # --- Single Mode Options ---
        "USE_VIRAL_SONG_FINDER": False,
        "USE_EXISTING_LRC": False,
        "MANUAL_SPOTIFY_URL": "https://open.spotify.com/track/3uwnnTQcHM1rDqSfA4gQNz?si=4394d4ed2f6d4bae",
        
        # --- Album Mode Options ---
        "MANUAL_ALBUM_URL": "https://open.spotify.com/album/4a6NzYL1YHRUgx9e3YZI6I",
        
        # --- General Options ---
        "GENERATE_NEW_BACKGROUND": True,
        "ONLY_FIND_ENGLISH_SONGS": True,
        
        # --- Shared Asset Paths ---
        "USED_SONGS_LOG": r"assets\used_songs.json",
        "CLIENT_SECRETS_FILE": 'assets/client_secrets.json',
        
        # --- YouTube Upload Configuration ---
        "CREDENTIALS_FILE": os.path.join(ASSET_FOLDER, 'youtube_credentials_solara.json'),
        "SCOPES": ['https://www.googleapis.com/auth/youtube', 'https://www.googleapis.com/auth/drive.file'],
        "YOUTUBE_PLAYLIST_ID": "PLu9y7VOqxCyqZrNr7NFCUcUqHIDgsOioP",
        "UPLOAD_TO_YOUTUBE": False,
        "UPLOAD_TO_GOOGLE_DRIVE": False, # Set to True to enable Drive upload
        "GOOGLE_DRIVE_FOLDER_ID": "1mVMqSgrHWCzloGccX6XE-xvMyVdBjT3W", # The ID of the parent folder in Drive for Vivid
        
        # --- Output File Names ---
        "VIDEO_FILE": os.path.join(ASSET_FOLDER, 'lyric_video_FINAL.mp4'),
        "DESCRIPTION_FILE": os.path.join(ASSET_FOLDER, 'youtube_description.txt'),
        "TITLE_FILE": os.path.join(ASSET_FOLDER, 'youtube_title.txt'),
        "THUMBNAIL_FILE": os.path.join(ASSET_FOLDER, 'thumbnail.png'),
        "WHISPER_OUTPUT_FILE": os.path.join(ASSET_FOLDER, 'whisper_output.json'),
    }

    # --- Run the main workflow with the defined configuration ---
    lyricbot_core.run_workflow(config)

if __name__ == '__main__':
    main()
