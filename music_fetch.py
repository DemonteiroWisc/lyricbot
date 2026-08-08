import os
import re
import requests
import yt_dlp
import syncedlyrics
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from ytmusicapi import YTMusic
import random
import time
import json
from urllib.parse import quote_plus
import billboard
import numpy as np
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from datetime import datetime
import glob
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
GENERATE_NEW_BACKGROUND = True
LEONARDO_API_KEY = os.environ.get("LEONARDO_API_KEY")

BLACKLISTED_ARTISTS = ["Taylor Swift"]

# --- AUTHENTICATION CONFIG ---
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

# --- HELPER FUNCTIONS ---
def is_likely_english(sp, spotify_track_object):
    """
    Checks if a song is likely English by analyzing the artist's genres.
    Returns True if likely English, False otherwise.
    """
    # A blocklist of keywords found in non-English genres. This can be expanded.
    NON_ENGLISH_GENRE_KEYWORDS = [
        'k-pop', 'j-pop', 'c-pop', 'pop francais', 'french pop', 'rap francais',
        'latin', 'reggaeton', 'trap latino', 'banda', 'corrido', 'grupera',
        'sertanejo', 'pagode', 'forro', 'axe', 'mpb', 'schlager', 'dansktop',
        'arabic', 'turkish', 'mandopop', 'cantopop', 'pop sunda', 'dangdut',
        'indonesian', 'russian', 'ukrainian', 'polish', 'otacore', 'anime'
    ]
    try:
        # Get the primary artist's ID from the track object
        artist_id = spotify_track_object['artists'][0]['id']
        if not artist_id: return True # Failsafe, assume English

        # Fetch artist details to get genres
        artist_details = sp.artist(artist_id)
        artist_genres = artist_details.get('genres', [])

        if not artist_genres:
            return True # No genre data, assume English as a failsafe

        # Check if any of the artist's genres contain a non-English keyword
        for genre in artist_genres:
            for keyword in NON_ENGLISH_GENRE_KEYWORDS:
                if keyword in genre.lower():
                    print(f"    -> Language Check: Artist genre '{genre}' suggests non-English. Skipping.")
                    return False

        # If no non-English keywords were found, it's likely English
        return True

    except Exception as e:
        print(f"    -> Language Check Warning: Could not verify artist genres: {e}")
        # Failsafe: If the API call fails for any reason, assume it's okay to proceed
        return True

def is_valid_lrc(lrc_text):
    if not lrc_text: return False
    if re.search(r'\[\d{2}:\d{2}\.\d{2,3}\]', lrc_text): return True
    return False

def clean_lrc_text(lrc_data):
    """
    Cleans LRC data by removing non-lyric artifacts and stripping section headers
    from the beginning of lines, preserving any subsequent text.
    """
    if not lrc_data: return None
    cleaned_lines = []
    SECTION_KEYWORDS = ['intro', 'verse', 'pre-chorus', 'chorus', 'post-chorus', 
                        'bridge', 'hook', 'refrain', 'outro', 'solo', 'instrumental']
    
    for line in lrc_data.strip().split('\n'):
        # Match a timestamp and capture all subsequent content
        match = re.match(r'(\[\d{2}:\d{2}\.\d{2,3}\])\s*(.*)', line.strip())
        
        if match:
            timestamp = match.group(1)
            content = match.group(2).strip()
            
            # --- HEADER STRIPPING LOGIC ---
            # This regex looks for a bracketed header ONLY at the start of the content.
            header_match = re.match(r'^\s*\[(.*?)\]\s*', content)
            if header_match:
                text_inside_brackets = header_match.group(1).lower()
                if any(keyword in text_inside_brackets for keyword in SECTION_KEYWORDS):
                    # It's a header. Remove it from the start of the content string.
                    content = content[len(header_match.group(0)):].strip()
            # --- END OF LOGIC ---
            
            # If any text remains after potential stripping, process it.
            if content:
                # Further clean remaining known artifacts like <00:..> tags
                clean_content = re.sub(r'<\d{2}:\d{2}\.\d{2,3}>', '', content)
                clean_content = re.sub(r'<[^>]*>', '', clean_content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                if clean_content:
                    cleaned_lines.append(f"{timestamp} {clean_content}")

    return '\n'.join(cleaned_lines)

def clean_lrc_for_netease(lrc_data):
    """
    Cleans LRC data, stripping headers and specifically removing any line 
    containing Chinese characters.
    """
    if not lrc_data: return None
    cleaned_lines = []
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    SECTION_KEYWORDS = ['intro', 'verse', 'pre-chorus', 'chorus', 'post-chorus', 
                        'bridge', 'hook', 'refrain', 'outro', 'solo', 'instrumental']
                        
    for line in lrc_data.strip().split('\n'):
        # First, check for and reject any line containing Chinese characters.
        if chinese_char_pattern.search(line):
            continue

        # Match a timestamp and capture all subsequent content
        match = re.match(r'(\[\d{2}:\d{2}\.\d{2,3}\])\s*(.*)', line.strip())
        
        if match:
            timestamp = match.group(1)
            content = match.group(2).strip()
            
            # --- HEADER STRIPPING LOGIC (identical to the function above) ---
            header_match = re.match(r'^\s*\[(.*?)\]\s*', content)
            if header_match:
                text_inside_brackets = header_match.group(1).lower()
                if any(keyword in text_inside_brackets for keyword in SECTION_KEYWORDS):
                    content = content[len(header_match.group(0)):].strip()
            # --- END OF LOGIC ---

            # If any text remains after potential stripping, process it.
            if content:
                clean_content = re.sub(r'<\d{2}:\d{2}\.\d{2,3}>', '', content)
                clean_content = re.sub(r'<[^>]*>', '', clean_content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                if clean_content:
                    cleaned_lines.append(f"{timestamp} {clean_content}")
            
    if not cleaned_lines: return None
    return '\n'.join(cleaned_lines)

def clean_youtube_title(title):
    """
    Cleans a YouTube video title to make it a better search query for Spotify.
    """
    if not title: return ""
    title = title.replace('“', '').replace('”', '')
    title = re.sub(r'[\(\[].*?[\)\]]', '', title)
    title = title.split('|')[0]
    keywords = ['official music video', 'official video', 'lyric video', 'lyrics', 'audio', 'official']
    for key in keywords:
        title = re.sub(f'(?i){key}', '', title)
    return title.strip()

def _format_all_artists(spotify_track_object):
    """Combines all artists from a Spotify track object into a single string."""
    artists = [artist['name'] for artist in spotify_track_object.get('artists', [])]
    if len(artists) > 1:
        # Join all but the last with a comma, and the last with an ampersand
        return f"{', '.join(artists[:-1])} & {artists[-1]}"
    elif artists:
        return artists[0]
    return "Unknown Artist"

def get_viral_ytmusic_song(used_songs_log_path, only_find_english_songs=False):
    """
    Fetches tracks from YT Music, performs a biased random selection, and *then*
    checks if the chosen song has been used. This is much faster as it minimizes API calls.
    """
    print("--- Finding a Trending Song from YouTube Music (Optimized Method) ---")

    # --- CONFIGURATION ---
    YT_PLAYLIST_URLS = [
        "https://music.youtube.com/playlist?list=PL4fGSI1pDJn6O1LS0XSdF3RyO0Rq_LDeI", # Viral Hits
        "https://music.youtube.com/playlist?list=RDCLAK5uy_k5n4srrEB1wgvIjPNTXS9G1ufE9WQxhnA", # Today's Hits
        "https://music.youtube.com/playlist?list=PL4fGSI1pDJn4XZgmK_9TcrUoxgcQK7zwu",
        "https://music.youtube.com/playlist?list=PL4fGSI1pDJn77aK7sAW2AT0oOzo5inWY8&si=4C7wAeN0RBm7hEvT" # Top 100
    ]
    TRACKS_TO_FETCH_PER_LIST = 25 # Keep this low for speed, 25 is a good balance.
    MAX_SEARCH_ATTEMPTS = 20 # Safety net to prevent an infinite loop.

    # --- Load used songs from log ---
    try:
        with open(used_songs_log_path, 'r') as f:
            used_songs_data = json.load(f)
            # --- MODIFICATION START ---
            used_spotify_ids = {song['spotify_id'] for song in used_songs_data if 'spotify_id' in song}
            used_song_artist_pairs = {(song['song_name'].lower(), song['artist_name'].lower()) for song in used_songs_data}
            print(f"  Loaded {len(used_spotify_ids)} IDs and {len(used_song_artist_pairs)} name/artist pairs from the log.")
            # --- MODIFICATION END ---
    except (FileNotFoundError, json.JSONDecodeError):
        used_spotify_ids = set()
        # --- MODIFICATION ---
        used_song_artist_pairs = set()
        print("  No previous song log found, or it's empty. Starting fresh.")

    # --- Step 1: Quickly gather all potential tracks without checking them ---
    all_tracks = []
    all_weights = []
    ytmusic = YTMusic()

    print("  -> Quickly fetching track lists from YouTube Music...")
    for url in YT_PLAYLIST_URLS:
        try:
            list_id = url.split("list=")[1].split("&")[0]
            content_data = ytmusic.get_playlist(list_id, limit=TRACKS_TO_FETCH_PER_LIST)
            if not content_data or 'tracks' not in content_data:
                continue
            
            yt_tracks = content_data['tracks']
            all_tracks.extend(yt_tracks)
            
            # Generate weights (inverse rank bias)
            num_tracks = len(yt_tracks)
            weights = list(range(num_tracks, 0, -1))
            all_weights.extend(weights)
        except Exception as e:
            print(f"  -> Warning: Could not process source {url}: {e}")
            continue

    if not all_tracks:
        print("\n[ERROR] No tracks could be fetched from any YouTube Music source. Aborting.")
        return None

    print(f"  -> Collected {len(all_tracks)} total candidates. Starting iterative search...")

    # --- Step 2: Iteratively pick, check, and return the first valid song ---
    auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    attempts = 0
    while attempts < MAX_SEARCH_ATTEMPTS and all_tracks:
        attempts += 1
        print(f"\n  Attempt #{attempts}...")

        # Pick a random track based on weight
        indices = list(range(len(all_tracks)))
        chosen_index = random.choices(indices, weights=all_weights, k=1)[0]
        
        # Get the chosen track and then remove it so we don't pick it again
        chosen_track = all_tracks.pop(chosen_index)
        all_weights.pop(chosen_index)

        yt_title = chosen_track.get('title')
        yt_artist = chosen_track['artists'][0]['name'] if chosen_track.get('artists') else 'Unknown'
        cleaned_title = clean_youtube_title(yt_title)
        
        print(f"    -> Selected '{cleaned_title}' by '{yt_artist}'. Checking Spotify...")

        # Now perform the single, targeted Spotify search
        query = f"track:{cleaned_title} artist:{yt_artist}"
        results = sp.search(q=query, type='track', limit=1, market="US")

        if not results or not results['tracks']['items']:
            print("    -> Not found on Spotify. Trying next song.")
            continue

        spotify_track = results['tracks']['items'][0]
        
        # =======================================================
        # --- NEW: ARTIST BLACKLIST CHECK ---
        current_artist_name = spotify_track['artists'][0]['name']
        if any(blacklisted_artist.lower() in current_artist_name.lower() for blacklisted_artist in BLACKLISTED_ARTISTS):
            print(f"    -> Artist '{current_artist_name}' is on the blacklist. Skipping.")
            continue
        # --- END OF NEW CODE ---
        # =======================================================
        
        # =======================================================
        # --- LANGUAGE FILTER ---
        if only_find_english_songs and not is_likely_english(sp, spotify_track):
            continue
        # --- END OF LANGUAGE FILTER ---
        # =======================================================
        
        spotify_id = spotify_track['id']
        # --- MODIFICATION START ---
        song_name = spotify_track['name']
        artist_name = spotify_track['artists'][0]['name']

        if spotify_id in used_spotify_ids:
            print(f"    -> Found on Spotify, but it has already been used (ID: {spotify_id}). Trying next song.")
            continue
        
        if (song_name.lower(), artist_name.lower()) in used_song_artist_pairs:
            print(f"    -> Found on Spotify, but it has already been used (Name/Artist Match). Trying next song.")
            continue
        # --- MODIFICATION END ---

        # --- SUCCESS: We found a new, valid song ---
        print(f"    -> SUCCESS: Found a new song to process!")
        song_info = {
            'name': spotify_track['name'],
            'artist': _format_all_artists(spotify_track),
            'duration': round(spotify_track['duration_ms'] / 1000),
            'is_explicit': spotify_track['explicit'],
            'spotify_id': spotify_id
        }
        return song_info

    print("\n[WARNING] Could not find a new, unused song after multiple attempts.")
    return None

def get_viral_billboard_song(used_songs_log_path, only_find_english_songs=False):
    print("--- Finding a Trending Song from Billboard Hot 100 ---")
    try:
        with open(used_songs_log_path, 'r') as f:
            used_songs_data = json.load(f)
            # --- MODIFICATION START ---
            used_spotify_ids = {song['spotify_id'] for song in used_songs_data if 'spotify_id' in song}
            used_song_artist_pairs = {(song['song_name'].lower(), song['artist_name'].lower()) for song in used_songs_data}
            print(f"  Loaded {len(used_spotify_ids)} IDs and {len(used_song_artist_pairs)} name/artist pairs from the log.")
            # --- MODIFICATION END ---
    except (FileNotFoundError, json.JSONDecodeError):
        used_spotify_ids = set()
        # --- MODIFICATION ---
        used_song_artist_pairs = set()
        print("  No previous song log found, or it's empty. Starting fresh.")

    try:
        print("  Fetching current Billboard Hot 100 chart...")
        chart = billboard.ChartData('hot-100')
        
        shuffled_chart = list(chart.entries)
        random.shuffle(shuffled_chart)
        
        auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
        sp = spotipy.Spotify(auth_manager=auth_manager)

        for song_entry in shuffled_chart:
            print(f"\n  Checking Billboard Rank #{song_entry.rank}: '{song_entry.title}' by '{song_entry.artist}'...")
            primary_artist = re.split(r' featuring | & | with | x | vs\. |, ', song_entry.artist, 1, flags=re.IGNORECASE)[0].strip()
            query = f"track:{song_entry.title} artist:{primary_artist}"
            results = sp.search(q=query, type='track', limit=1)

            if not results or not results['tracks']['items']:
                print("    -> Could not find a match on Spotify. Skipping.")
                continue

            top_result = results['tracks']['items'][0]

            current_artist_name = top_result['artists'][0]['name']
            if any(blacklisted_artist.lower() in current_artist_name.lower() for blacklisted_artist in BLACKLISTED_ARTISTS):
                print(f"    -> Artist '{current_artist_name}' is on the blacklist. Skipping.")
                continue
            
            if only_find_english_songs and not is_likely_english(sp, top_result):
                continue

            # --- MODIFICATION START ---
            spotify_id = top_result['id']
            song_name = top_result['name']
            artist_name = top_result['artists'][0]['name']

            if spotify_id in used_spotify_ids:
                print(f"    -> Found on Spotify (ID: {spotify_id}), but it has already been used. Skipping.")
                continue

            if (song_name.lower(), artist_name.lower()) in used_song_artist_pairs:
                print(f"    -> Found on Spotify (Name/Artist Match), but it has already been used. Skipping.")
                continue
            # --- MODIFICATION END ---

            # --- Found a new, unused song ---
            print("    -> SUCCESS: Found a new song to process!")
            song_info = {
                'name': top_result['name'],
                'artist': _format_all_artists(top_result),
                'duration': round(top_result['duration_ms'] / 1000),
                'is_explicit': top_result['explicit'],
                'spotify_id': spotify_id
            }
            return song_info

        print("\n[WARNING] All available songs from the chart have already been used.")
        return None

    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        return None

# --- LYRIC FETCHING FUNCTIONS ---
def _lrclib_get(api_url, max_retries=3, timeout=15):
    """GET with retries for transient SSL/connection errors to lrclib.net.
    Uses a fresh session per call and Connection: close to avoid reusing
    a connection the server may have closed (which can cause SSLEOFError).
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            session.headers["Connection"] = "close"
            response = session.get(api_url, timeout=timeout)
            session.close()
            return response
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** attempt  # 2, 4, 8 seconds
                print(f"    -> Lrclib request failed (attempt {attempt}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
    raise last_error

def get_lrclib_api_search_candidates(song_name, artist_name, output_folder, unique_lrcs, start_index, limit=3):
    print(f"  > Checking Lrclib API (/search) for '{song_name}' by '{artist_name}'...")
    api_url = f"https://lrclib.net/api/search?artist_name={quote_plus(artist_name)}&track_name={quote_plus(song_name)}"
    found_paths = []
    try:
        response = _lrclib_get(api_url); response.raise_for_status()
        data = response.json()
        if not data: print("    -> No results found from Lrclib API."); return []
        print(f"    -> Lrclib API returned {len(data)} potential versions.")
        for item in data:
            if len(found_paths) >= limit: break
            lrc_data = item.get('syncedLyrics')
            final_lrc_content = clean_lrc_text(lrc_data)
            if not final_lrc_content or final_lrc_content in unique_lrcs: continue
            unique_lrcs.add(final_lrc_content)
            output_path = os.path.join(output_folder, f"lyrics_candidate_{start_index + len(found_paths)}.lrc")
            with open(output_path, 'w', encoding='utf-8') as f: f.write(final_lrc_content)
            found_paths.append(output_path)
            print(f"    -> SUCCESS: Saved Lrclib API candidate to '{output_path}'.")
        return found_paths
    except requests.exceptions.RequestException as e:
        print(f"    -> An error occurred calling Lrclib /search API: {e}"); return []

def get_lrclib_api_get_candidate(song_name, artist_name, duration, output_folder, unique_lrcs, start_index):
    print(f"  > Checking Lrclib API (/get) for duration-matched lyric...")
    api_url = f"https://lrclib.net/api/get?artist_name={quote_plus(artist_name)}&track_name={quote_plus(song_name)}&duration={duration}"
    try:
        response = _lrclib_get(api_url)
        if response.status_code == 404: print("    -> No duration-matched lyric found."); return []
        response.raise_for_status()
        item = response.json()
        if not item: return []
        lrc_data = item.get('syncedLyrics')
        final_lrc_content = clean_lrc_text(lrc_data)
        if not final_lrc_content or final_lrc_content in unique_lrcs: return []
        unique_lrcs.add(final_lrc_content)
        output_path = os.path.join(output_folder, f"lyrics_candidate_{start_index}.lrc")
        with open(output_path, 'w', encoding='utf-8') as f: f.write(final_lrc_content)
        print(f"    -> SUCCESS: Saved duration-matched Lrclib candidate to '{output_path}'.")
        return [output_path]
    except requests.exceptions.RequestException as e:
        print(f"    -> An error occurred calling Lrclib /get API: {e}"); return []

# --- CORE ASSET GATHERING ---
def fetch_all_assets(song_info, num_audio_to_download=2, asset_folder="test_assets"):
    print("--- Starting Asset Download Process ---")
    os.makedirs(asset_folder, exist_ok=True)
    
    # Unpack all info from the song_info dictionary
    song_name = song_info['name']
    artist_name = song_info['artist']
    duration = song_info['duration']
    is_explicit = song_info['is_explicit']
    
    audio_search_term = f"{song_name} {artist_name}"
    background_output_path = os.path.join(asset_folder, "test_background.jpg")

    # Download audio first
    downloaded_audio_paths = download_song_audio(audio_search_term, asset_folder, num_to_download=num_audio_to_download)
    audio_success = len(downloaded_audio_paths) > 0

    print(f"\nSearching for lyrics for '{song_name}'...")
    if is_explicit:
        print("  -> Song is explicit. Prioritizing search for explicit versions.")

    candidate_paths, unique_lrcs = [], set()
    MAX_CANDIDATES_FROM_LRCLIB_SEARCH = 3

    # --- Tier 1: Lrclib Search with the FULL, specific song title ---
    print("\n--- Tier 1: Searching Lrclib with specific title ---")
    explicit_song_name = f"{song_name} (Explicit)"
    if is_explicit:
        specific_explicit_paths = get_lrclib_api_search_candidates(
            explicit_song_name, artist_name, asset_folder, unique_lrcs,
            start_index=len(candidate_paths), limit=MAX_CANDIDATES_FROM_LRCLIB_SEARCH
        )
        candidate_paths.extend(specific_explicit_paths)
    
    needed = MAX_CANDIDATES_FROM_LRCLIB_SEARCH - len(candidate_paths)
    if needed > 0:
        specific_normal_paths = get_lrclib_api_search_candidates(
            song_name, artist_name, asset_folder, unique_lrcs,
            start_index=len(candidate_paths), limit=needed
        )
        candidate_paths.extend(specific_normal_paths)

    # --- Tier 2: Lrclib Search with a SIMPLIFIED song title ---
    if len(candidate_paths) < MAX_CANDIDATES_FROM_LRCLIB_SEARCH:
        print("\n--- Tier 2: Broadening search on Lrclib with simplified title ---")
        simplified_song_name = re.sub(r'\s*\([^)]*\)', '', song_name).strip()
        simplified_explicit_name = f"{simplified_song_name} (Explicit)"

        if simplified_song_name.lower() != song_name.lower():
            if is_explicit:
                needed = MAX_CANDIDATES_FROM_LRCLIB_SEARCH - len(candidate_paths)
                if needed > 0:
                    simplified_explicit_paths = get_lrclib_api_search_candidates(
                        simplified_explicit_name, artist_name, asset_folder, unique_lrcs,
                        start_index=len(candidate_paths), limit=needed
                    )
                    candidate_paths.extend(simplified_explicit_paths)

            needed = MAX_CANDIDATES_FROM_LRCLIB_SEARCH - len(candidate_paths)
            if needed > 0:
                simplified_normal_paths = get_lrclib_api_search_candidates(
                    simplified_song_name, artist_name, asset_folder, unique_lrcs,
                    start_index=len(candidate_paths), limit=needed
                )
                candidate_paths.extend(simplified_normal_paths)
        else:
            print("  -> Simplified name is the same as the original, skipping broader search.")

    # --- Tier 3: Musixmatch Multi-Strategy Search ---
    # MODIFICATION: This tier now runs regardless of previous success.
    print("\n--- Tier 3: Searching Musixmatch with multiple strategies ---")
    musixmatch_found = False
    search_terms_to_try = []
    simplified_name = re.sub(r'\s*\([^)]*\)', '', song_name).strip()
    featured_artist = ""
    match = re.search(r'(?:feat|ft|featuring)\.?\s(.*?)\)', song_name, re.IGNORECASE)
    if match: featured_artist = match.group(1).strip()
    
    search_terms_to_try.append(f"{song_name} {artist_name}")
    search_terms_to_try.append(f"{simplified_name} {artist_name} {featured_artist}".strip())
    search_terms_to_try.append(f"{simplified_name} {artist_name}")
    unique_search_terms = list(dict.fromkeys(search_terms_to_try))

    for term in unique_search_terms:
        if musixmatch_found: break
        print(f"  > Checking Musixmatch with term: '{term}'...")
        try:
            lrc_data = syncedlyrics.search(term, save_path=None, enhanced=True, providers=['Musixmatch'])
            final_lrc_content = clean_lrc_text(lrc_data)
            if final_lrc_content and final_lrc_content not in unique_lrcs:
                unique_lrcs.add(final_lrc_content)
                output_path = os.path.join(asset_folder, f"lyrics_candidate_{len(candidate_paths)}.lrc")
                with open(output_path, 'w', encoding='utf-8') as f: f.write(final_lrc_content)
                candidate_paths.append(output_path)
                print(f"    -> SUCCESS: Saved unique Musixmatch candidate.")
                musixmatch_found = True
        except Exception:
            print(f"    -> Failed.")
    if not musixmatch_found:
        print("    -> No valid LRC found from any Musixmatch strategy.")

    # --- Tier 4: Lrclib GET (duration-matched, higher priority) ---
    # MODIFICATION: This tier now runs regardless of previous success.
    print("\n--- Tier 4: Searching Lrclib for duration-matched lyric ---")
    duration_match_found = False
    simplified_song_name_for_get = re.sub(r'\s*\([^)]*\)', '', song_name).strip()
    titles_to_try = [f"{song_name} (Explicit)", song_name]
    if simplified_song_name_for_get.lower() != song_name.lower():
        titles_to_try.extend([f"{simplified_song_name_for_get} (Explicit)", simplified_song_name_for_get])
    
    if not is_explicit:
        titles_to_try = [t for t in titles_to_try if "(Explicit)" not in t]

    for title_attempt in titles_to_try:
        if duration_match_found: break
        get_paths = get_lrclib_api_get_candidate(title_attempt, artist_name, duration, asset_folder, unique_lrcs, start_index=len(candidate_paths))
        if get_paths:
            candidate_paths.extend(get_paths)
            duration_match_found = True
    
    # --- Tier 5: Last Resort Netease Backup ---
    # MODIFICATION: This tier now only runs if less than 2 candidates were found from all above sources.
    if len(candidate_paths) < 2:
        print("\n--- Tier 5: Last Resort Netease Backup Search (fewer than 2 candidates found) ---")
        simplified_name = re.sub(r'\s*\([^)]*\)', '', song_name).strip()
        featured_artist = ""
        match = re.search(r'(?:feat|ft|featuring)\.?\s(.*?)\)', song_name, re.IGNORECASE)
        if match: featured_artist = match.group(1).strip()
        netease_search_term = f"{simplified_name} {artist_name} {featured_artist}".strip()
        
        print(f"  > Checking Netease with term: '{netease_search_term}'...")
        try:
            lrc_data = syncedlyrics.search(netease_search_term, save_path=None, enhanced=True, providers=['Netease'])
            final_lrc_content = clean_lrc_for_netease(lrc_data)
            if final_lrc_content and final_lrc_content not in unique_lrcs:
                unique_lrcs.add(final_lrc_content)
                output_path = os.path.join(asset_folder, f"lyrics_candidate_{len(candidate_paths)}.lrc")
                with open(output_path, 'w', encoding='utf-8') as f: f.write(final_lrc_content)
                candidate_paths.append(output_path)
                print(f"    -> SUCCESS: Saved unique, cleaned Netease candidate.")
        except Exception:
            print(f"    -> Failed. No results from Netease.")
            
    # --- Final Summary ---
    lyrics_success = len(candidate_paths) > 0

    background_success = False
    if GENERATE_NEW_BACKGROUND:
        background_success = generate_background_image(background_output_path)
    else:
        # Deferred: core generates background after quality check, or file already exists
        background_success = True

    print("\n--- Download Summary ---")
    print(f"Audio: {'SUCCESS' if audio_success else 'FAILED'}")
    print(f"Lyrics: {'SUCCESS' if lyrics_success else 'FAILED'} ({len(candidate_paths)} candidates found)")
    print(f"Background: {'SUCCESS' if background_success else 'FAILED'}")

    if all([audio_success, lyrics_success, background_success]):
        return candidate_paths, downloaded_audio_paths
    else:
        return [], []

# --- OTHER FUNCTIONS ---
def get_song_details_from_spotify(url, client_id, client_secret):
    """This function is now ONLY used for the MANUAL_SPOTIFY_URL fallback."""
    print("--- Fetching details for manual Spotify URL ---")
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    try:
        track_info = sp.track(url)
        
        song_info = {
            'name': track_info['name'],
            'artist': _format_all_artists(track_info),
            'duration': round(track_info['duration_ms'] / 1000),
            'is_explicit': track_info['explicit'],
            'spotify_id': track_info['id']  # <-- THIS IS THE FIX
        }
        
        print(f"  Song: {song_info['name']}")
        print(f"  Artist: {song_info['artist']}")
        print(f"  Duration: {song_info['duration']}s")
        print(f"  Explicit: {'Yes' if song_info['is_explicit'] else 'No'}")
        
        return song_info
        
    except Exception as e:
        print(f"  [ERROR] Failed to fetch details from Spotify: {e}")
        return None
    
def get_album_tracks_from_spotify(album_url, client_id, client_secret):
    """
    Fetches all tracks from a given Spotify album URL.

    Returns:
        A list of song_info dictionaries for each track in the album.
    """
    print(f"--- Fetching all tracks from Spotify album: {album_url} ---")
    
    # Authenticate with Spotify
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        # Get the album's tracks
        results = sp.album_tracks(album_url)
        tracks = results['items']
        
        # Handle paginated results for very large albums
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
            
        if not tracks:
            print("  [ERROR] No tracks found for this album.")
            return []

        album_songs = []
        for track in tracks:
            # Create a song_info dictionary that matches the structure used elsewhere
            song_info = {
                'name': track['name'],
                'artist': _format_all_artists(track),
                'duration': round(track['duration_ms'] / 1000),
                'is_explicit': track['explicit'],
                'spotify_id': track['id']
            }
            album_songs.append(song_info)
        
        print(f"  -> Successfully fetched {len(album_songs)} tracks from the album.")
        return album_songs

    except Exception as e:
        print(f"  [CRITICAL ERROR] Failed to fetch album details from Spotify: {e}")
        print("  Please check if the album URL is correct and public.")
        return []

def download_song_audio(search_term, asset_folder, output_base_name="test_song", num_to_download=2, start_index=1):
    """
    Downloads the top N YouTube search results for a song as MP3 files.
    Retries once after a 5-minute pause if the first attempt fails.
    Can specify a start_index to download a specific range of results.
    """
    MAX_ATTEMPTS = 3
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Adjust log message based on whether we're downloading a range or a single item
        if start_index > 1:
            print(f"\nDownloading audio candidate #{start_index} for '{search_term}' from YouTube... (Attempt {attempt}/{MAX_ATTEMPTS})")
            search_query = f'ytsearch{start_index}' # Search for the top N to get the Nth item
        else:
            print(f"\nDownloading top {num_to_download} audio candidate(s) for '{search_term}' from YouTube... (Attempt {attempt}/{MAX_ATTEMPTS})")
            search_query = f'ytsearch{num_to_download}'
        
        # This tells yt-dlp which items from the search results to download (e.g., 1-2, or just 3)
        playlist_items_spec = f"{start_index}-{num_to_download}" if start_index <= num_to_download else str(start_index)

        ydl_opts = {
            'format': 'bestaudio/best',
            'default_search': search_query,
            'outtmpl': os.path.join(asset_folder, f'{output_base_name}_v%(playlist_index)s.%(ext)s'), # <-- Note the added .%(ext)s
            'noplaylist': True,
            'playlist_items': playlist_items_spec,
            # ==============================================================================
            # --- TEMPORARY FIX for YouTube 403 Errors (October 2024) ---
            # This forces yt-dlp to use the "actual" player version, bypassing some
            # of YouTube's recent download restrictions.
            # REMOVE THIS once yt-dlp releases a permanent fix.
            # See: https://github.com/yt-dlp/yt-dlp/issues/14680
            'extractor_args': {
                'youtube': {
                    'player_js_version': 'actual'
                }
            }
            # --- END OF TEMPORARY FIX ---
            # ==============================================================================
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([search_term])
            
            downloaded_paths = glob.glob(os.path.join(asset_folder, f"{output_base_name}_v*.*"))
            
            if len(downloaded_paths) > 0:
                print(f"  -> Successfully downloaded {len(downloaded_paths)} audio file(s).")
                return downloaded_paths
            else:
                print("  -> [ERROR] yt-dlp ran but no output files were found.")

        except Exception as e:
            print(f"  -> [ERROR] An exception occurred during audio download: {e}")

        if attempt < MAX_ATTEMPTS:
            print("  -> Download failed. Waiting for 5 minutes before retrying...")
            time.sleep(300)
        else:
            print("  -> Final download attempt failed.")

    return []

def prepare_media_assets(song_info, asset_folder):
    """
    Prepares assets that are NOT lyrics (audio and background).
    This is used when the user wants to provide their own LRC file.
    """
    print("--- Preparing Media-Only Assets (Audio & Background) ---")
    os.makedirs(asset_folder, exist_ok=True)
    
    search_term = f"{song_info['name']} {song_info['artist']}"
    background_output_path = os.path.join(asset_folder, "test_background.jpg")

    # 1. Download Audio
    audio_success = len(download_song_audio(search_term, asset_folder)) > 0

    # 2. Generate or confirm Background
    background_success = False
    if GENERATE_NEW_BACKGROUND:
        background_success = generate_background_image(background_output_path)
    else:
        # If not generating, success is just the file existing.
        background_success = os.path.exists(background_output_path)

    print("\n--- Media Preparation Summary ---")
    print(f"Audio: {'SUCCESS' if audio_success else 'FAILED'}")
    print(f"Background: {'SUCCESS' if background_success else 'FAILED'}")
    
    # Return True only if both were successful
    return all([audio_success, background_success])

def generate_background_image(output_path):
    MAX_ATTEMPTS = 2
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\nGenerating AI background with Leonardo.AI... (Attempt {attempt}/{MAX_ATTEMPTS})")
        
        FLUX_DEV_MODEL_ID = "b2614463-296c-462a-9586-aafdb8f00e36"
        base_prompts = [
            "A vast, minimalist sky with a smooth, deeply saturated evening gradient. A simple, sharp mountain range silhouette rests directly on the bottom edge of the frame, highly detailed.",
            "A minimalist, sharp-focus sky with a vibrant and warm evening gradient. A dark, crisp mountain silhouette is flush with the absolute bottom of the image, clear and detailed.",
            "Cinematic wide shot of a minimalist abstract gradient with rich and intensely vibrant warm colors. A sharp, minimalist mountain silhouette sits firmly at the very bottom border of the image.",
        ]
        prompt = random.choice(base_prompts)
        negative_prompt = "watermark, text, blurry, soft, out of focus, hazy, grainy, noisy, misty, foggy, dull"
        print(f"  -> Using prompt: \"{prompt[:80]}...\"")
        headers = {"accept": "application/json", "content-type": "application/json", "authorization": f"Bearer {LEONARDO_API_KEY}"}
        payload = {"prompt": prompt, "negative_prompt": negative_prompt, "modelId": FLUX_DEV_MODEL_ID, "width": 1536, "height": 864, "num_images": 1, "presetStyle": "NONE", "guidance_scale": 7}
        start_job_url = "https://cloud.leonardo.ai/api/rest/v1/generations"

        try:
            response = requests.post(start_job_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            generation_id = response.json()['sdGenerationJob']['generationId']
            print(f"  -> Job started successfully. Generation ID: {generation_id}")

            get_job_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
            max_wait_time = 180; poll_interval = 10; start_time = time.time(); image_generated = False
            while time.time() - start_time < max_wait_time:
                print("  -> Checking job status...")
                response = requests.get(get_job_url, headers=headers, timeout=30); response.raise_for_status()
                job_data = response.json().get('generations_by_pk', {}); job_status = job_data.get('status')
                if job_status == 'COMPLETE':
                    print("  -> Generation COMPLETE.")
                    images = job_data.get('generated_images', [])
                    if images:
                        image_url = images[0]['url']; print("  -> Downloading final image...")
                        image_response = requests.get(image_url, timeout=30)
                        if image_response.status_code == 200:
                            with open(output_path, "wb") as f: f.write(image_response.content)
                            print(f"  -> Successfully saved base image to '{output_path}'")
                            image_generated = True
                    break
                elif job_status == 'FAILED':
                    raise Exception("Leonardo generation job failed.")
                else:
                    time.sleep(poll_interval)
            
            if not image_generated:
                raise Exception("Job timed out or failed to generate image URL.")

            print("\n--- Upscaling Background Image for Maximum Quality ---")
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(scale=4, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', model=model, tile=0, half=False)
            img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
            original_height, original_width = img.shape[:2]
            print("  Enhancing image...")
            output, _ = upsampler.enhance(img, outscale=4)
            target_width = original_width * 2; target_height = original_height * 2
            final_image = cv2.resize(output, (target_width, target_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, final_image)
            print(f"  -> SUCCESS: Final high-resolution image saved. New resolution: {final_image.shape[1]}x{final_image.shape[0]}")
            return True # Success, exit function

        except Exception as e:
            print(f"  -> [ERROR] An error occurred during background generation: {e}")

        if attempt < MAX_ATTEMPTS:
            print("  -> Background generation failed. Waiting for 5 minutes before retrying...")
            time.sleep(300)
        else:
            print("  -> Final background generation attempt failed.")
            
    return False

def log_used_song(song_info, youtube_video_id, used_songs_log_path):
    """Logs a song with comprehensive details, adding it to the top of the file."""
    print("\n--- Logging Used Song ---")
    try:
        with open(used_songs_log_path, 'r') as f:
            used_songs_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        used_songs_data = []

    # Use .get() for safety, providing a default value of None if the key is missing
    spotify_id = song_info.get('spotify_id')

    # Final duplicate check using Spotify ID before writing, only if the ID exists
    if spotify_id and any(s.get('spotify_id') == spotify_id for s in used_songs_data):
        print(f"  Song with Spotify ID '{spotify_id}' was already in the log.")
        return

    # Create the new, comprehensive log entry using .get() for all fields
    new_entry = {
        "song_name": song_info.get('name', 'N/A'),
        "artist_name": song_info.get('artist', 'N/A'),
        "is_explicit": song_info.get('is_explicit', False),
        "date_added": datetime.now().isoformat(timespec='seconds'),
        "youtube_id": youtube_video_id if youtube_video_id else None,
        "youtube_url": f"https://www.youtube.com/watch?v={youtube_video_id}" if youtube_video_id else None,
        "spotify_id": spotify_id,
        "spotify_url": f"https://open.spotify.com/track/{spotify_id}" if spotify_id else "N/A"
    }

    # Insert the new entry at the beginning of the list
    used_songs_data.insert(0, new_entry)
    
    with open(used_songs_log_path, 'w') as f:
        json.dump(used_songs_data, f, indent=4)
        
    print(f"  Successfully added '{song_info.get('name', 'N/A')}' to the top of the log.")

# --- MAIN WORKFLOW ---
def main():
    """The main function to orchestrate the asset fetching workflow."""
    print("--- Starting Music Fetch Workflow ---")
    
    used_songs_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "used_songs.json")
    
    # 1. Find a new viral song from YouTube Music
    # In standalone mode, we'll default to not filtering by language
    song_info = get_viral_ytmusic_song(used_songs_log_path, only_find_english_songs=False)
    
    if not song_info:
        print("\nCould not find a new song to process. Exiting.")
        return

    # 2. Fetch all the assets for that song
    candidate_paths, audio_paths = fetch_all_assets(song_info, asset_folder="test_assets")

    if not candidate_paths:
        print("\nFailed to fetch all necessary assets. Exiting.")
        return

    # 3. If everything was successful, log the song so we don't use it again
    log_used_song(song_info, None, used_songs_log_path)

    print("\n--- Workflow Complete ---")
    print(f"Assets are ready in the 'test_assets' folder.")


if __name__ == "__main__":
    main()