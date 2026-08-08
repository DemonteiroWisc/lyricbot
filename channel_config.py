import os

# --- Settings shared by every channel ---
USED_SONGS_LOG = r"assets\used_songs.json"
CLIENT_SECRETS_FILE = "assets/client_secrets.json"
MANUAL_ALBUM_URL = "https://open.spotify.com/album/4a6NzYL1YHRUgx9e3YZI6I"

# --- Per-channel identity (constant across prod/test) plus prod/test overrides ---
CHANNELS = {
    "labs": {
        "asset_folder": "1_labs_assets",
        "lower_third_filename": "sub_lower_third_labs.mov",
        "credentials_filename": "youtube_credentials_labs.json",
        "youtube_handle": "LyricLabsOfficial",
        "playlist_id": "PLkfnDbhByBDhQP-t6dAAnL-EI54KAIN4B",
        "drive_folder_suffix": "Lyric Labs",
        "drive_folder_id": "1TiouoSPV0Oy90JpoejCvCT_l8hSjAQDL",
        "only_english": False,
        "prod": {
            "use_viral_finder": True,
            "manual_spotify_url": "https://open.spotify.com/track/68APF7rM3bIxKCjqeO1QSZ?si=15186f45e55f4679",
            "drive_scope": "drive",
            "upload_to_drive": True,
        },
        "test": {
            "use_viral_finder": False,
            "manual_spotify_url": "https://open.spotify.com/track/5vNRhkKd0yEAg8suGBpjeY",
            "drive_scope": "drive",
            "upload_to_drive": True,
        },
    },
    "vivid": {
        "asset_folder": "2_vivid_assets",
        "lower_third_filename": "sub_lower_third_vivid.mov",
        "credentials_filename": "youtube_credentials_vivid.json",
        "youtube_handle": "vividmusiclyrics",
        "playlist_id": "PL0K6msmvkKYuTXwGWCOQb01Bnang4vmxG",
        "drive_folder_suffix": "Vivid Music",
        "drive_folder_id": "1z9wQ24BB-xAZ3FKp5ca6SB9Avfsedxbq",
        "only_english": True,
        "prod": {
            "use_viral_finder": True,
            "manual_spotify_url": "https://open.spotify.com/track/6MzofobZt2dm0Kf1hTThFz",
            "drive_scope": "drive",
            "upload_to_drive": True,
        },
        "test": {
            "use_viral_finder": True,
            "manual_spotify_url": "https://open.spotify.com/track/6MzofobZt2dm0Kf1hTThFz",
            "drive_scope": "drive.file",
            "upload_to_drive": True,
        },
    },
    "solara": {
        "asset_folder": "3_solara_assets",
        "lower_third_filename": "sub_lower_third_solara.mov",
        "credentials_filename": "youtube_credentials_solara.json",
        "youtube_handle": None,  # not yet provided -- reconciliation is skipped for this channel until it is
        "playlist_id": "PLu9y7VOqxCyqZrNr7NFCUcUqHIDgsOioP",
        "drive_folder_suffix": "Solara Music",
        "drive_folder_id": "1mVMqSgrHWCzloGccX6XE-xvMyVdBjT3W",
        "only_english": True,
        "prod": {
            "use_viral_finder": True,
            "manual_spotify_url": "https://open.spotify.com/track/003vvx7Niy0yvhvHt4a68B?si=d89e5e1bc2df409d",
            "drive_scope": "drive",
            "upload_to_drive": True,
        },
        "test": {
            "use_viral_finder": False,
            "manual_spotify_url": "https://open.spotify.com/track/3uwnnTQcHM1rDqSfA4gQNz?si=4394d4ed2f6d4bae",
            "drive_scope": "drive.file",
            "upload_to_drive": False,
        },
    },
}


def build_config(channel: str, test_mode: bool = False) -> dict:
    """
    Builds the full config dict for a channel, mirroring what each
    lyricbot_<channel>[_test].py script used to define by hand.
    """
    ch = CHANNELS[channel]
    mode = ch["test"] if test_mode else ch["prod"]
    asset_folder = ch["asset_folder"]

    return {
        "ASSET_FOLDER": asset_folder,
        # Lets run_workflow keep test-run bookkeeping (resume checkpoint,
        # run history) separate from prod's, even though both share the
        # same ASSET_FOLDER for its actual media assets.
        "IS_TEST_RUN": test_mode,
        "LOWER_THIRD_VIDEO": os.path.join(asset_folder, ch["lower_third_filename"]),

        "PROCESSING_MODE": "SINGLE",

        "NUM_AUDIO_CANDIDATES": 2,

        "USE_VIRAL_SONG_FINDER": mode["use_viral_finder"],
        "USE_EXISTING_LRC": False,
        "MANUAL_SPOTIFY_URL": mode["manual_spotify_url"],

        "MANUAL_ALBUM_URL": MANUAL_ALBUM_URL,

        "GENERATE_NEW_BACKGROUND": True,
        "ONLY_FIND_ENGLISH_SONGS": ch["only_english"],

        "USED_SONGS_LOG": USED_SONGS_LOG,
        "CLIENT_SECRETS_FILE": CLIENT_SECRETS_FILE,

        "CREDENTIALS_FILE": os.path.join(asset_folder, ch["credentials_filename"]),
        "YOUTUBE_HANDLE": ch["youtube_handle"],
        "SCOPES": [
            "https://www.googleapis.com/auth/youtube",
            f"https://www.googleapis.com/auth/{mode['drive_scope']}",
        ],
        "YOUTUBE_PLAYLIST_ID": ch["playlist_id"],
        "UPLOAD_TO_YOUTUBE": False,
        "UPLOAD_TO_GOOGLE_DRIVE": mode["upload_to_drive"],
        "GOOGLE_DRIVE_FOLDER_SUFFIX": ch["drive_folder_suffix"],
        "GOOGLE_DRIVE_FOLDER_ID": ch["drive_folder_id"],

        "VIDEO_FILE": os.path.join(asset_folder, "lyric_video_FINAL.mp4"),
        "DESCRIPTION_FILE": os.path.join(asset_folder, "youtube_description.txt"),
        "TITLE_FILE": os.path.join(asset_folder, "youtube_title.txt"),
        "THUMBNAIL_FILE": os.path.join(asset_folder, "thumbnail.png"),
        "WHISPER_OUTPUT_FILE": os.path.join(asset_folder, "whisper_output.json"),
    }
