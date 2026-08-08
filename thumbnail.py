import os
import re
import spotipy
import lyricsgenius
import numpy as np
from thefuzz import fuzz
from spotipy.oauth2 import SpotifyClientCredentials
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
GENIUS_API_TOKEN = os.environ.get("GENIUS_API_TOKEN")
STANDALONE_SPOTIFY_URL = "https://open.spotify.com/track/5vNRhkKd0yEAg8suGBpjeY"
STANDALONE_ASSET_FOLDER = "1_labs_assets" # The folder to use when running this script directly

# --- STYLE AND FILE CONFIGURATION ---
FONT_FILE = 'assets/BebasNeue-Regular.ttf'
SHADOW_COLOR = 'black'
TEXT_COLOR = 'white'
SHADOW_OFFSET = (6, 6)   # Offset in pixels (x, y)
# --- MODIFICATION: ADDED A SEPARATE SHADOW OFFSET FOR THE ARTIST ---
ARTIST_SHADOW_OFFSET = (4, 4) # Closer shadow for the artist text
# --- END OF MODIFICATION ---
SHADOW_BLUR_RADIUS = 3   # The radius for the Gaussian blur
THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720
FONT_SIZE_REDUCTION_THRESHOLD_FOR_SPLIT = 0.65 # If font size drops below this % of max, force a two-line split.
VERTICAL_OFFSET_PIXELS = 5

# --- DESCRIPTION GENERATOR (UNCHANGED) ---
def _clean_song_name(song_name):
    """Strips version/remaster suffixes (e.g. '- 2012 Remaster', '(Remastered 1999)') and feat./ft. credits from a song title."""
    dash_suffix_pattern = r'\s*-\s*((\d{4}\s*)?remaster(ed)?(\s*\d{4})?|single version|radio edit|album version|explicit.*)\s*$'
    clean_song_name = re.sub(dash_suffix_pattern, '', song_name, flags=re.IGNORECASE).strip()
    paren_suffix_pattern = r'\s*[\(\[]((\d{4}\s*)?remaster(ed)?(\s*\d{4})?|single version|radio edit|album version)[\)\]]\s*$'
    clean_song_name = re.sub(paren_suffix_pattern, '', clean_song_name, flags=re.IGNORECASE).strip()
    clean_song_name = re.sub(r'\s*[\(\[](?:feat|ft)\.?.*?[\)\]]', '', clean_song_name, flags=re.IGNORECASE).strip()
    return clean_song_name

def _format_genius_lyrics_final(raw_lyrics, asset_folder):
    """
    Cleans Genius lyrics by fixing annotations, ensuring section breaks,
    and removing all extraneous internal whitespace for a perfect format.
    """
    if not raw_lyrics: return ""

    # Helper function to clean the content of annotations (brackets and parens)
    def clean_annotation_content(match):
        text = match.group(0)
        # Step 1: Collapse all internal newlines and multiple spaces into a single space.
        cleaned_content = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
        # Step 2: Clean up spacing around commas (e.g., " , " becomes ", ").
        cleaned_content = re.sub(r'\s*,\s*', ', ', cleaned_content)
        # Step 3 (THE FIX): Remove space immediately after an opening bracket/paren.
        cleaned_content = re.sub(r'([(\[])\s+', r'\1', cleaned_content)
        # Step 4 (THE FIX): Remove space immediately before a closing bracket/paren.
        cleaned_content = re.sub(r'\s+([)\]])', r'\1', cleaned_content)
        return cleaned_content

    # First, do a global pass to fix the messy, multi-line annotations.
    lyrics = re.sub(r'\[.*?\]', clean_annotation_content, raw_lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'\(.*?\)', clean_annotation_content, lyrics, flags=re.DOTALL)
    
    # Force a blank line before EVERY section header.
    lyrics = re.sub(r'\n(\[)', r'\n\n\1', lyrics)
    
    # Normalize all newline sequences to preserve stanza breaks perfectly.
    lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)

    return lyrics.strip()

def _clean_text_for_comparison(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _load_font(font_file, font_size):
    """Helper to load a font, with fallback to default."""
    try:
        font = ImageFont.truetype(font_file, font_size)
    except IOError:
        print(f"  [WARNING] Font file not found at '{font_file}'. Using default font.")
        font = ImageFont.load_default()
    return font

def _get_text_width(text, font):
    """A small helper to get the pixel width of a string."""
    # Use multiline_textbbox to correctly measure width even if text contains newlines
    # For single line measurement, it behaves like textbbox.
    bbox = ImageDraw.Draw(Image.new("RGBA", (1,1))).multiline_textbbox((0,0), text, font=font)
    return bbox[2] - bbox[0]

def _force_two_line_split(text, font, max_width):
    """
    Intelligently splits a single line of text into two lines, prioritizing
    punctuation and balanced line lengths.
    """
    words = text.split(' ')
    possible_breaks = []

    for i in range(1, len(words)):
        line1 = " ".join(words[:i])
        line2 = " ".join(words[i:])
        
        if _get_text_width(line1, font) <= max_width and _get_text_width(line2, font) <= max_width:
            balance_score = abs(len(line1) - len(line2))
            is_punctuation_break = words[i-1].endswith((',', '!', '?', '.', ';', ':'))
            if is_punctuation_break:
                balance_score -= 1000 # Make this break extremely attractive
            possible_breaks.append({'index': i, 'score': balance_score})
    
    if not possible_breaks: return text # If no valid split points, return original text
    best_break = min(possible_breaks, key=lambda x: x['score'])
    return " ".join(words[:best_break['index']]) + "\n" + " ".join(words[best_break['index']:])

def _format_lrc_to_plain_text(lrc_file_path):
    if not os.path.exists(lrc_file_path): return None
    with open(lrc_file_path, 'r', encoding='utf-8') as f:
        lrc_content = f.read()
    lyrics_lines = re.findall(r'\[\d{2}:\d{2}\.\d{2,3}\]\s*(.*)', lrc_content)
    return '\n'.join(line for line in lyrics_lines if line.strip())

def create_youtube_description(song_name, artist_name, asset_folder, channel_name="Lyric Labs"):
    print("\n--- Generating YouTube Description ---")
    used_lrc_path = os.path.join(asset_folder, "test_lyrics.lrc")
    local_lyrics = _format_lrc_to_plain_text(used_lrc_path)
    if not local_lyrics:
        print("  [ERROR] Could not read local lyric file. Aborting description generation.")
        return
    
    final_lyrics_to_use = local_lyrics
    
    if GENIUS_API_TOKEN and GENIUS_API_TOKEN != "YOUR_GENIUS_API_ACCESS_TOKEN_HERE":
        print("  -> Attempting to fetch better-formatted lyrics from Genius...")
        genius = lyricsgenius.Genius(GENIUS_API_TOKEN, remove_section_headers=True, timeout=10)
        try:
            song = genius.search_song(song_name, artist_name)
            if song:
                genius_lyrics_raw = song.lyrics.strip()
                clean_local = _clean_text_for_comparison(local_lyrics)
                clean_genius = _clean_text_for_comparison(genius_lyrics_raw)
                similarity_score = fuzz.ratio(clean_local, clean_genius)
                print(f"  -> Genius lyrics found. Similarity score: {similarity_score}%")
                
                if similarity_score > 80:
                    print("  -> Match is strong. Formatting and using the Genius lyrics.")
                    final_lyrics_to_use = _format_genius_lyrics_final(genius_lyrics_raw, asset_folder)
                else:
                    print("  -> Mismatch detected. Falling back to the lyrics from the .lrc file.")
            else:
                print("  -> Song not found on Genius. Using local .lrc lyrics.")
        except Exception as e:
            print(f"  [WARNING] An error occurred while fetching from Genius: {e}")
            print("  -> Defaulting to local .lrc lyrics.")
    else:
        print("  -> Genius API token not set. Using local .lrc lyrics.")

    # --- START OF FIX ---
    # The YouTube API rejects descriptions with '<' or '>' characters.
    # This line removes them to prevent the upload from failing.
    final_lyrics_to_use = final_lyrics_to_use.replace('<', '').replace('>', '')
    # --- END OF FIX ---
    
    # --- FIX: Use the same cleaning logic for the description as for the thumbnail/title ---
    clean_song_name = _clean_song_name(song_name)
    hashtag_song = ''.join(c for c in clean_song_name.lower() if c.isalnum())
    hashtag_artist = ''.join(c for c in artist_name.lower() if c.isalnum())
    
    description_content = f"""{artist_name} - {clean_song_name} (Lyrics)\n\n🔔 Subscribe and turn on notifications to {channel_name} for more of the latest lyric videos!\n\n🎤 Lyrics: {clean_song_name} - {artist_name}\n{final_lyrics_to_use}\n\n#{hashtag_song} #{hashtag_artist} #lyrics"""
    
    output_filename = os.path.join(asset_folder, "youtube_description.txt")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(description_content)
        print(f"  -> Successfully created '{output_filename}'")
    except IOError as e:
        print(f"  [ERROR] Could not write description file: {e}")

def create_youtube_title(song_name, artist_name, asset_folder):
    """Creates a text file with the formatted YouTube video title."""
    print("\n--- Generating YouTube Title File ---")
    clean_song_name = _clean_song_name(song_name)

    title_content = f"{artist_name} - {clean_song_name} (Lyrics)"
    output_filename = os.path.join(asset_folder, "youtube_title.txt")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(title_content)
        print(f"  -> Successfully created '{output_filename}'")
    except IOError as e:
        print(f"  [ERROR] Could not write title file: {e}")

def create_blurred_text_image(text, font_file, font_size, text_color, shadow_color, shadow_offset, blur_radius):
    """
    Creates a PIL Image of text with a blurred drop shadow on a transparent background.
    """
    try:
        font = ImageFont.truetype(font_file, font_size)
    except IOError:
        print(f"  [ERROR] Font file not found at '{font_file}'. Using default font.")
        font = ImageFont.load_default()
    
    # Use multiline_textbbox for accurate dimensions with potentially wrapped text
    interline_spacing = 12 # Consistent interline spacing for thumbnail text
    bbox = ImageDraw.Draw(Image.new("RGBA", (1,1))).multiline_textbbox((0,0), text, font=font, align='center', spacing=interline_spacing)
    padding = int(blur_radius * 4)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    padding = int(blur_radius * 4)
    canvas_w = int(text_w + abs(shadow_offset[0]) + padding)
    canvas_h = int(text_h + abs(shadow_offset[1]) + padding)

    draw_pos = (int(padding / 2 - bbox[0]), int(padding / 2 - bbox[1]))

    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_pos = (draw_pos[0] + shadow_offset[0], draw_pos[1] + shadow_offset[1])
    shadow_draw.multiline_text(shadow_pos, text, font=font, fill=shadow_color, align='center', spacing=interline_spacing)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.multiline_text(draw_pos, text, font=font, fill=text_color, align='center', spacing=interline_spacing)

    final_image = Image.alpha_composite(shadow_layer, text_layer)
    return final_image


def generate_thumbnail(song_title, artist_name, asset_folder="test_assets", channel_name="Lyric Labs"):
    print("\n--- Generating Thumbnail Image ---")

    # Define file paths based on the provided asset_folder
    background_image_path = os.path.join(asset_folder, "test_background.jpg")
    output_filename = os.path.join(asset_folder, "thumbnail.png")
    
    # --- FIX: Expanded regex to remove "Single Version" and other common suffixes ---
    simple_song = _clean_song_name(song_title)
    # --- END OF FIX ---
    
    simple_artist = artist_name.split('feat.')[0].strip()
    print(f"  Using simplified names for thumbnail: '{simple_song}' by '{simple_artist}'")

    if not os.path.exists(background_image_path):
        print(f"  Error: Background image not found at '{background_image_path}'.")
        return

    try:
        bg_pil_img = Image.open(background_image_path)
        original_width, original_height = bg_pil_img.size
        aspect_ratio = original_height / original_width
        new_height = int(THUMBNAIL_WIDTH * aspect_ratio)
        resized_bg_pil_img = bg_pil_img.resize((THUMBNAIL_WIDTH, new_height), Image.Resampling.LANCZOS)
        background_clip = ImageClip(np.array(resized_bg_pil_img))
    except Exception as e:
        print(f"  [FATAL ERROR] Failed to load or resize background image: {e}")
        return

    # --- MODIFICATION: Increased max text width from 90% to 95% ---
    max_text_width = THUMBNAIL_WIDTH * 0.95
    
    # --- MODIFICATION: Title font size calculation is now independent ---
    initial_font_size_candidate = 235
    font_size_title = initial_font_size_candidate
    print("  Calculating optimal font size for the title...")
    while font_size_title > 10:
        # Use PIL's ImageFont to measure text width for accurate calculation
        temp_font = _load_font(FONT_FILE, font_size_title)
        if _get_text_width(simple_song, temp_font) < max_text_width:
            break
        font_size_title -= 5
    print(f"    -> Title font size set to: {font_size_title}px")

    # --- NEW LOGIC: Force two lines if font size was reduced too much ---
    if font_size_title < initial_font_size_candidate * FONT_SIZE_REDUCTION_THRESHOLD_FOR_SPLIT:
        print(f"  -> Title font size ({font_size_title}px) is less than 80% of initial ({initial_font_size_candidate}px). Forcing two lines.")
        # --- THIS IS THE FIX: Use the shrunken font size to find a valid split point ---
        current_font_for_split = _load_font(FONT_FILE, font_size_title)
        simple_song = _force_two_line_split(simple_song, current_font_for_split, max_text_width)
        
        # --- THIS IS THE FIX ---
        # Now that the text is split, recalculate the font size to make it as large as possible.
        # --- NEW: Cap the max font size at 80% of the original max for split titles ---
        new_max_font_size = int(initial_font_size_candidate * 0.80)
        print(f"  -> Recalculating font size for the new two-line title (max size capped at {new_max_font_size}px)...")
        font_size_title = new_max_font_size # Reset to the new, capped max size
        while font_size_title > 10:
            temp_font = _load_font(FONT_FILE, font_size_title)
            if _get_text_width(simple_song, temp_font) < max_text_width:
                break
            font_size_title -= 5
        print(f"    -> New optimal two-line font size: {font_size_title}px")
    # --- END NEW LOGIC ---
    
    # --- MODIFICATION: Artist font size is now calculated independently ---
    font_size_artist = 95 # Start with a large, independent size for the artist
    print("  Calculating optimal font size for the artist name...")
    while font_size_artist > 10:
        temp_font = _load_font(FONT_FILE, font_size_artist)
        if _get_text_width(simple_artist, temp_font) < max_text_width:
            break
        font_size_artist -= 5
    print(f"    -> Artist font size set to: {font_size_artist}px")

    print("  Generating text images with blurred shadows and applying wrapping rules...")
    try:
        # Pass the potentially multi-line simple_song directly
        title_pil_img = create_blurred_text_image(text=simple_song, font_file=FONT_FILE, font_size=font_size_title, text_color=TEXT_COLOR, shadow_color=SHADOW_COLOR, shadow_offset=SHADOW_OFFSET, blur_radius=SHADOW_BLUR_RADIUS)
        title_clip = ImageClip(np.array(title_pil_img))

        # Artist name is not subject to forced two-line split based on font size reduction
        artist_pil_img = create_blurred_text_image(text=simple_artist, font_file=FONT_FILE, font_size=font_size_artist, text_color=TEXT_COLOR, shadow_color=SHADOW_COLOR, shadow_offset=ARTIST_SHADOW_OFFSET, blur_radius=SHADOW_BLUR_RADIUS)
        artist_clip = ImageClip(np.array(artist_pil_img))
    except Exception as e:
        print(f"  [FATAL ERROR] Failed to create text images with PIL: {e}")
        return
    
    spacing_between_clips = 25
    total_text_height = title_clip.h + artist_clip.h + spacing_between_clips
    
    top_of_block_y = (THUMBNAIL_HEIGHT / 2) - (total_text_height / 2) + VERTICAL_OFFSET_PIXELS
    
    title_y_pos = top_of_block_y
    artist_y_pos = title_y_pos + title_clip.h + spacing_between_clips

    final_title_clip = title_clip.set_position(('center', title_y_pos))
    final_artist_clip = artist_clip.set_position(('center', artist_y_pos))

    final_thumbnail = CompositeVideoClip(
        [background_clip, final_title_clip, final_artist_clip],
        size=(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
    )

    final_thumbnail.save_frame(output_filename)
    print(f"\n--- Thumbnail saved successfully to '{output_filename}' ---")
    
    create_youtube_title(song_title, artist_name, asset_folder)
    create_youtube_description(song_title, artist_name, asset_folder, channel_name=channel_name)

def get_song_details_from_spotify(url, client_id, client_secret):
    print("--- Fetching Song Details from Spotify ---")
    try:
        manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=manager)
        track_info = sp.track(url)
        
        # NEW HELPER FUNCTION from music_fetch.py to get all artists.
        # It needs to be copied here if thumbnail.py is truly standalone.
        def _format_all_artists(spotify_track_object):
            artists = [artist['name'] for artist in spotify_track_object.get('artists', [])]
            if len(artists) > 1:
                return f"{', '.join(artists[:-1])} & {artists[-1]}"
            elif artists:
                return artists[0]
            return "Unknown Artist"
            
        song_name = track_info['name']
        # --- MODIFICATION START ---
        artist_name = _format_all_artists(track_info)
        # --- MODIFICATION END ---
        
        return song_name, artist_name
    except Exception as e:
        print(f"  Failed to fetch details from Spotify: {e}")
        return None, None

if __name__ == '__main__':
    print("--- Running thumbnail.py in Standalone Mode ---")
    song_name, artist_name = get_song_details_from_spotify(
        STANDALONE_SPOTIFY_URL, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
    )
    if song_name and artist_name:
        generate_thumbnail(song_name, artist_name, asset_folder=STANDALONE_ASSET_FOLDER)
    else:
        print("Could not fetch song info. Aborting.")