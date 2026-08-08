import os
import re
import sys
import subprocess
import math
import stable_whisper as whisper
import torch
import shutil
import time
import glob
import numpy as np # <-- ADDED IMPORT
from thefuzz import fuzz
from moviepy.editor import *
import moviepy.video.fx.all as vfx
from pydub import AudioSegment
from pydub.effects import strip_silence
from pydub.silence import detect_silence
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter # <-- ADDED IMPORTS
import unicodedata

# --- CONFIGURATION: STYLE AND TIMING ---
FONT_FILE = 'assets/BebasNeue-Regular.ttf'
FADE_DURATION = 0.13
FONT_SIZE = 110
# --- REPLACED OLD SHADOW STYLE WITH NEW ---
TEXT_COLOR = 'white'
SHADOW_COLOR = 'black'
SHADOW_OFFSET = (6, 6)      # Offset in pixels (x, y)
SHADOW_BLUR_RADIUS = 3      # The radius for the Gaussian blur
USE_SNOW_OVERLAY = False
SNOW_OVERLAY_PATH = "assets/snow_overlay.mov"

# --- AI & RENDER CONFIGURATION ---
WHISPER_MODEL = "large-v3"
USE_GPU = True
VOCAL_SILENCE_THRESHOLD_S = 2.0
VOCAL_SILENCE_CUSHION_S = 0.5
NVENC_PRESET = 'p3'
CPU_PRESET = 'fast'
FUZZY_MATCH_THRESHOLD = 85
GROUPING_TIME_THRESHOLD = 2.5
CHARS_PER_SECOND = 10 
MIN_READABILITY_CUSHION_S = 1.00 # The minimum time a lyric will hang after being sung
READABILITY_CUSHION_FACTOR = 0.25 # Adds 15% of the sung duration as extra cushion
MAX_LAST_WORD_DUR = 3.0
MAX_GROUP_LINES = 6
MAX_FALLBACK_GROUP_SECONDS = 5.0
FFMPEG_BINARY_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# (Standard helper functions remain unchanged)
def trim_silence_from_audio(audio_path, silence_db_threshold=-60, min_silence_duration_ms=500):
    print("\n--- Pre-processing: Checking for Silence in Audio File ---")
    
    # Define the output path for the trimmed version
    base, ext = os.path.splitext(audio_path)
    # --- THIS IS THE FIX ---
    # Explicitly create a .wav path for the lossless trimmed file
    trimmed_audio_path = f"{base}_trimmed.wav"

    try:
        audio_segment = AudioSegment.from_file(audio_path)
        original_duration = len(audio_segment) / 1000.0
        print(f"  Original duration: {original_duration:.2f}s")

        trimmed_segment = strip_silence(
            audio_segment, 
            silence_thresh=silence_db_threshold, 
            silence_len=min_silence_duration_ms
        )
        
        trimmed_duration = len(trimmed_segment) / 1000.0

        if trimmed_duration < (original_duration - 0.1):
            print(f"  Trimmed duration: {trimmed_duration:.2f}s. Saving new file.")
            # --- THIS IS THE CRITICAL CHANGE ---
            # Export as a lossless WAV file to preserve 100% of the quality.
            trimmed_segment.export(trimmed_audio_path, format="wav")
            print(f"  -> Saved trimmed audio to: {trimmed_audio_path}")
            return trimmed_audio_path
        else:
            print("  -> No significant leading/trailing silence detected. No new file created.")
            return None
            
    except Exception as e:
        print(f"  [ERROR] Could not process audio for silence trimming: {e}")
        return None
    
def detect_long_silences(audio_path, silence_db_threshold=-60):
    """
    Analyzes an audio file to find periods of silence longer than the configured threshold.
    Returns a list of timestamps where lyrics should be forced to end.
    """
    print("\n--- Analyzing Vocals for Long Pauses ---")
    try:
        audio_segment = AudioSegment.from_file(audio_path)
        # detect_silence returns a list of [start, end] pairs in milliseconds
        silences = detect_silence(
            audio_segment,
            min_silence_len=int(VOCAL_SILENCE_THRESHOLD_S * 1000),
            silence_thresh=silence_db_threshold
        )

        if not silences:
            print("  -> No long vocal silences found. Timing will not be adjusted.")
            return []

        # We only care about the START of a long silence. We convert it to seconds
        # and add a small cushion to avoid cutting off lyrics abruptly.
        forced_end_times = [(s[0] / 1000.0) + VOCAL_SILENCE_CUSHION_S for s in silences]
        print(f"  -> Found {len(forced_end_times)} long pauses. Lyrics will be cut short before these points.")
        return forced_end_times

    except Exception as e:
        print(f"  [ERROR] Could not analyze vocal track for silences: {e}")
        return []

def apply_silence_timing(timed_lyrics, forced_end_times):
    """
    Adjusts the end time of lyrics that extend into a detected silent period
    and ensures the 'status' element is always removed for the final video render.
    """
    print("\n--- Applying Vocal Silence Timing Adjustments ---")
    
    adjusted_lyrics = []
    # The tuple now includes status, so we unpack all three parts.
    for ((start, end), text, status) in timed_lyrics:
        new_end = end
        
        # --- THIS IS THE FIX ('Do Not Disturb' Rule) ---
        # Only apply silence cuts to cues that were individually successful.
        # This prevents the function from incorrectly shortening 'grouped' cues
        # that are intentionally designed to span silent gaps.
        if forced_end_times and status == 'success':
            # Find the earliest forced end time that occurs after the lyric starts
            for forced_end in forced_end_times:
                if forced_end > start and forced_end < new_end:
                    # This lyric extends into a silent period. Cap its duration.
                    new_end = forced_end
                    print(f"  -> Applied silence cut to cue: '{text.replace(chr(10), ' ')}'")
                    break # We only need to apply the first relevant silence cut

        # Append the (potentially adjusted) cue back to the list as a 2-element tuple.
        adjusted_lyrics.append(((start, new_end), text))

    print("  -> Silence timing adjustments applied.")
    return adjusted_lyrics

def is_lrc_explicit(lrc_file_path):
    """
    Checks if the content of an LRC file contains common explicit words.
    """
    # A basic list of words to check for. You can expand this list.
    EXPLICIT_KEYWORDS = {'fuck', 'shit', 'bitch', 'cunt', 'asshole', 'ass', 'dick', 'pussy', 'cock', 'nigger', 'faggot'}
    try:
        # Use utf-8-sig to correctly handle a potential Byte Order Mark (BOM)
        with open(lrc_file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read().lower()
            # Use a generator expression for an efficient check
            if any(word in content for word in EXPLICIT_KEYWORDS):
                return True
    except Exception:
        # If file reading fails for any reason, assume it's not explicit
        return False
    return False
    
def sanitize_for_rendering(text):
    """
    Removes characters that cannot be encoded in 'latin-1', effectively stripping
    out most non-Latin scripts to prevent font rendering errors ("tofu").
    """
    # Encode the string into a basic character set, ignoring any characters that fail.
    # Then, decode it back into a clean Python string.
    return text.encode('latin-1', 'ignore').decode('latin-1')

def check_nvenc_support():
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        # --- FIX HERE: 'get_setting' is removed. Let the system find 'ffmpeg'. ---
        # Your script already sets the correct path via an environment variable.
        ffmpeg_binary = "ffmpeg"
        # --- END FIX ---
        
        cmd = [ffmpeg_binary, '-encoders']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo)
        return 'h264_nvenc' in result.stdout
    except Exception: 
        return False

def separate_vocals(audio_path, variant_name, asset_folder):
    """
    Separates vocals, placing the output in a uniquely named directory
    to avoid conflicts between trimmed/untrimmed versions.
    """
    print(f"--- 1. Separating Vocals for '{variant_name}' version ---")
    
    # Create a unique output directory for this audio variant
    demucs_output_dir = os.path.join(asset_folder, f"demucs_separated_{variant_name}")
    
    if os.path.exists(demucs_output_dir):
        print(f"  Previous demucs output found. Deleting '{demucs_output_dir}' to ensure a fresh separation.")
        shutil.rmtree(demucs_output_dir)
    os.makedirs(demucs_output_dir, exist_ok=True)
    
    demucs_model = "htdemucs_ft" 
    command = [sys.executable, "-m", "demucs.separate", "--mp3", "--two-stems", "vocals", "-n", demucs_model, "-o", demucs_output_dir, audio_path]
    
    try:
        print(f"  Executing Demucs command for '{os.path.basename(audio_path)}'")
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("  Demucs separation completed.")
        
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        vocal_track_path = os.path.join(demucs_output_dir, demucs_model, audio_filename, "vocals.mp3")
        
        if not os.path.exists(vocal_track_path):
            print(f"  [ERROR] Vocal track not found after separation. Looked for: {vocal_track_path}")
            if process.stderr: print(f"  Demucs stderr output:\n{process.stderr.decode()}")
            return None, None
            
        # Return both the path to the vocals and the directory they are in for later cleanup
        return vocal_track_path, demucs_output_dir

    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Error during Demucs execution:", e.stderr.decode())
        return None, None
    except Exception as e:
        print(f"  [CRITICAL ERROR] An unexpected error occurred during Demucs processing: {e}")
        return None, None

def align_lyrics_to_vocals(vocal_track_path, lrc_path, whisper_model):
    print(f"--- 2. Performing Forced Alignment for '{os.path.basename(lrc_path)}' ---")
    try:
        with open(lrc_path, 'r', encoding='utf-8-sig') as f:
            full_lyrics_text = "\n".join([re.sub(r'\[.*?\]', '', line).strip() for line in f])
        
        # --- THE FIX: ADDED max_expand_ms PARAMETER ---
        result = whisper_model.align(
            vocal_track_path, 
            full_lyrics_text, 
            language='en',
            regroup=False,
            max_word_dur=3.0  # Allow timestamps to stretch by up to 1 second
        )
        # --- END OF FIX ---

        print("  Forced alignment complete for this candidate.")
        return [word.to_dict() for segment in result.segments for word in segment.words]

    except Exception as e:
        print(f"  [ERROR] Error during Forced Alignment for this candidate: {e}")
        return []

def clean_text_for_matching(text, keep_parentheses=False):
    if not keep_parentheses: text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r"[^\w\s'()]", '', text)
    return text.lower().strip()

def get_precise_timings_for_cues(display_cues, aligned_words, verbose_logging=False):
    # --- CONFIGURATION FOR TIMING BIAS ---
    # These values determine the weighting between the original LRC file's timing
    # and Whisper's detected timing. They should add up to 1.0.
    # Example: 0.5/0.5 is an even split. 0.7/0.3 would favor the LRC time more.
    LRC_START_TIME_BIAS = 0.5
    WHISPER_START_TIME_BIAS = 0.5
    # --- END OF CONFIGURATION ---

    if verbose_logging: print("\n--- Correlating Cues with Aligned Timestamps (Detailed Log) ---")
    timed_lyrics = []

    for cue in display_cues:
        lrc_start_time, lrc_end_time, cue_text = cue['start'], cue['end'], cue['text']
        main_text = clean_text_for_matching(cue_text, keep_parentheses=False)
        ad_libs = re.findall(r'\(([^)]+)\)', cue_text)
        if not main_text.split():
            timed_lyrics.append({'text': cue_text, 'start': lrc_start_time, 'end': lrc_end_time, 'status': 'fallback', 'lines': cue['lines']}); continue
        
        clean_cue_words = main_text.split()
        
        best_match = {'score': 0, 'start_time': -1, 'end_time': -1, 'end_index': -1, 'words': []}
        
        search_start_time, search_end_time = lrc_start_time - 1.0, lrc_start_time + 1.0
        window_indices = [i for i, word in enumerate(aligned_words) if search_start_time <= word['start'] < search_end_time]
        
        if not window_indices:
            if verbose_logging: print(f"  [FALLBACK] No words in window for: '{cue_text.replace(chr(10), ' ')}'")
            timed_lyrics.append({'text': cue_text, 'start': lrc_start_time, 'end': lrc_end_time, 'status': 'fallback', 'lines': cue['lines']}); continue
            
        for i in window_indices:
            phrase_len = len(clean_cue_words)
            if i + phrase_len > len(aligned_words): continue
                
            candidate_words = aligned_words[i : i + phrase_len]
            candidate_phrase = " ".join(clean_text_for_matching(w['word']) for w in candidate_words)
            score = fuzz.ratio(main_text, candidate_phrase)
            
            if score > best_match['score']:
                best_match.update({
                    'score': score, 
                    'start_time': candidate_words[0]['start'], 
                    'end_time': candidate_words[-1]['end'], 
                    'end_index': i + phrase_len - 1,
                    'words': candidate_words
                })

        if best_match['score'] < FUZZY_MATCH_THRESHOLD:
            if verbose_logging: print(f"  [FALLBACK] Score ({best_match['score']}) too low for: '{cue_text.replace(chr(10), ' ')}'")
            timed_lyrics.append({'text': cue_text, 'start': lrc_start_time, 'end': lrc_end_time, 'status': 'fallback', 'lines': cue['lines']}); continue
        
        whisper_start_time = best_match['start_time']
        
        # Calculate the new start time using a weighted average based on the configured biases.
        final_start_time = (lrc_start_time * LRC_START_TIME_BIAS) + (whisper_start_time * WHISPER_START_TIME_BIAS)

        last_word = best_match['words'][-1]
        last_word_duration = last_word['end'] - last_word['start']
        
        final_end_time = best_match['end_time']

        if last_word_duration > MAX_LAST_WORD_DUR:
            final_end_time = last_word['start'] + MAX_LAST_WORD_DUR
        
        # --- FIX: Check if ad_libs actually contain text before processing ---
        # This prevents an IndexError if a cue contains empty parentheses like "()".
        ad_lib_text = " ".join(ad_libs).strip()
        if ad_lib_text:
            # --- THIS IS THE FIX ---
            # Ensure that after cleaning, there are still words to process.
            clean_ad_lib_words = clean_text_for_matching(ad_lib_text).split()
            if not clean_ad_lib_words: continue # Skip to the next cue if ad-lib is empty
            ad_lib_search_start = best_match['end_index'] + 1
            if ad_lib_search_start < len(aligned_words):
                best_ad_lib_match = {'score': 0, 'end_time': -1}; ad_lib_window = aligned_words[ad_lib_search_start : ad_lib_search_start + 10]
                for i in range(len(ad_lib_window)):
                    phrase_len = len(clean_ad_lib_words)
                    if i + phrase_len > len(ad_lib_window): continue
                    candidate_words = ad_lib_window[i : i + phrase_len]
                    candidate_phrase = " ".join(clean_text_for_matching(w['word']) for w in candidate_words)
                    score = fuzz.ratio(clean_text_for_matching(ad_lib_text), candidate_phrase)
                    if score > best_ad_lib_match['score']: best_ad_lib_match.update({'score': score, 'end_time': candidate_words[-1]['end']})
                if best_ad_lib_match['score'] >= FUZZY_MATCH_THRESHOLD:
                    if best_ad_lib_match['end_time'] > final_end_time:
                         final_end_time = best_ad_lib_match['end_time']

        # --- THIS IS THE MODIFIED LOGGING LINE ---
        if verbose_logging:
            timing_info = f"(LRC:{lrc_start_time:.2f}s, W:{whisper_start_time:.2f}s -> {final_start_time:.2f}s)"
            print(f"  [SUCCESS] Score {best_match['score']}: Timed cue: '{cue_text.replace(chr(10), ' ')}' {timing_info}")
        
        timed_lyrics.append({'text': cue_text, 'start': final_start_time, 'end': final_end_time, 'status': 'success', 'lines': cue['lines']})
        
    return timed_lyrics

def refine_word_timings(aligned_words, low_prob_threshold=0.4, long_duration_threshold=1.2, shift_amount_s=0.5, verbose=False):
    """
    Analyzes Whisper's output to correct words stretched over silent gaps.
    Now with a 'verbose' flag to control console output.
    """
    if verbose: print("\n--- Refining Word Timings to Correct for Gaps (Detailed Log) ---")
    refined_count = 0
    for i, word in enumerate(aligned_words):
        duration = word['end'] - word['start']
        
        if word['probability'] < low_prob_threshold and duration > long_duration_threshold:
            new_start_time = word['end'] - shift_amount_s
            
            if i > 0 and new_start_time < aligned_words[i-1]['end']:
                new_start_time = aligned_words[i-1]['end'] + 0.01

            if verbose:
                print(f"  -> [REFINING] Suspect word '{word['word'].strip()}' (Prob: {word['probability']:.2f}, Dur: {duration:.2f}s).")
                print(f"     Original time: {word['start']:.2f}s -> {word['end']:.2f}s. New start: {new_start_time:.2f}s.")
            
            word['start'] = new_start_time
            refined_count += 1
            
    if verbose:
        if refined_count == 0:
            print("  -> No suspect timings found. No changes made.")
        else:
            print(f"  -> Refined {refined_count} word timestamp(s).")
        
    return aligned_words

def preprocess_lyrics_for_display(lrc_path):
    """
    Reads an LRC file, sanitizes the text to prevent "tofu" characters,
    and groups lines for better display timing.
    """
    try:
        with open(lrc_path, 'r', encoding='utf-8-sig') as f:
            # This main regex just separates the timestamp from everything else.
            regex = r"\[(\d{2}):(\d{2}\.\d{2,3})\]\s*(.*)"
            raw_lines = []
            
            SECTION_KEYWORDS = ['intro', 'verse', 'pre-chorus', 'chorus', 'post-chorus', 
                                'bridge', 'hook', 'refrain', 'outro', 'solo', 'instrumental']

            for line in f:
                match = re.match(regex, line.strip())

                if match:
                    # We have a timestamped line. Now we clean the content.
                    content = match.group(3).strip()
                    
                    # --- NEW HEADER STRIPPING LOGIC ---
                    # This regex looks for a bracketed header ONLY at the start of the content.
                    header_match = re.match(r'^\s*\[(.*?)\]\s*', content)
                    
                    if header_match:
                        # We found a bracketed prefix. Check if it's a header.
                        text_inside_brackets = header_match.group(1).lower()
                        if any(keyword in text_inside_brackets for keyword in SECTION_KEYWORDS):
                            # It IS a header. Remove it from the start of the content string.
                            # header_match.group(0) is the full matched string (e.g., "[Chorus: BIA] ")
                            content = content[len(header_match.group(0)):]
                    # --- END OF HEADER STRIPPING LOGIC ---
                    
                    # After stripping, if any real text remains, process it.
                    if content:
                        original_text = content
                        HOMOGLYPH_MAP = {
                            'е': 'e', 'Е': 'E', 'о': 'o', 'О': 'O', 'а': 'a', 'А': 'A',
                            'р': 'p', 'Р': 'P', 'с': 'c', 'С': 'C', 'і': 'i', 'І': 'I'
                        }
                        sanitized_chars = [HOMOGLYPH_MAP.get(char, char) for char in original_text]
                        sanitized_text = "".join(sanitized_chars)

                        start_time = int(match.group(1)) * 60 + float(match.group(2))
                        raw_lines.append({'text': sanitized_text, 'start': start_time})

    except FileNotFoundError: 
        print(f"Error: LRC file not found at {lrc_path}")
        return []
        
    # The rest of the function remains unchanged
    for i, line in enumerate(raw_lines):
        next_start = raw_lines[i+1]['start'] if i + 1 < len(raw_lines) else line['start'] + 5
        line['end'] = next_start
        
    grouped_cues, i = [], 0
    while i < len(raw_lines):
        line1 = raw_lines[i]
        if i + 1 < len(raw_lines):
            line2 = raw_lines[i+1]
            if len(line1['text']) < 35 and len(line2['text']) < 35 and (line2['start'] - line1['start']) < GROUPING_TIME_THRESHOLD:
                grouped_cues.append({'text': f"{line1['text']}\n{line2['text']}", 'start': line1['start'], 'end': line2['end'], 'lines': [line1, line2]})
                i += 2
                continue
        grouped_cues.append({'text': line1['text'], 'start': line1['start'], 'end': line1['end'], 'lines': [line1]})
        i += 1
        
    return grouped_cues

def group_fallbacks(timed_cues):
    print("\n--- Grouping Fallbacks with Successes (True Sandwich) ---")
    final_cues, i = [], 0
    while i < len(timed_cues):
        if timed_cues[i]['status'] == 'fallback':
            if len(timed_cues[i]['text'].split('\n')) > MAX_GROUP_LINES:
                print(f"  [WARNING] Individual fallback cue exceeds MAX_GROUP_LINES. Adding as is.")
            final_cues.append(timed_cues[i]); i += 1; continue

        start_cue, fallbacks_in_middle, end_anchor_index = timed_cues[i], [], -1
        for j in range(i + 1, len(timed_cues)):
            if timed_cues[j]['status'] == 'success':
                end_anchor_index = j
                break
            else:
                fallbacks_in_middle.append(timed_cues[j])

        if end_anchor_index != -1 and fallbacks_in_middle:
            end_anchor_cue = timed_cues[end_anchor_index]
            
            time_between_anchors = end_anchor_cue['start'] - start_cue['end']
            if time_between_anchors > MAX_FALLBACK_GROUP_SECONDS:
                print(f"  [INFO] Time gap of {time_between_anchors:.2f}s is too long to group. Handling individually.")
                final_cues.append(start_cue)
                i += 1
                continue

            potential_full_group_cues = [start_cue] + fallbacks_in_middle + [end_anchor_cue]
            total_lines_in_potential_full_group = sum(len(c['text'].split('\n')) for c in potential_full_group_cues)
            
            if total_lines_in_potential_full_group <= MAX_GROUP_LINES:
                combined_text = "\n".join([c['text'] for c in potential_full_group_cues])
                # --- FIX: Combine the 'lines' from all the original cues. ---
                combined_lines = [line for cue in potential_full_group_cues for line in cue['lines']]
                final_cues.append({
                    'text': combined_text, 
                    'start': potential_full_group_cues[0]['start'], 
                    'end': potential_full_group_cues[-1]['end'], 
                    'status': 'grouped',
                    'lines': combined_lines
                })
                print(f"  [GROUPED] Created a {total_lines_in_potential_full_group}-line sandwich group.")
                i = end_anchor_index + 1
                continue
            else:
                print(f"  [INFO] Full sandwich too large ({total_lines_in_potential_full_group} lines). Splitting.")
                cues_to_distribute = potential_full_group_cues
                while cues_to_distribute:
                    current_sub_group_cues, current_sub_group_line_count = [], 0
                    for k, cue_candidate in enumerate(cues_to_distribute):
                        lines_from_this_cue = len(cue_candidate['text'].split('\n'))
                        if current_sub_group_line_count + lines_from_this_cue <= MAX_GROUP_LINES:
                            current_sub_group_cues.append(cue_candidate)
                            current_sub_group_line_count += lines_from_this_cue
                        else:
                            break
                    if not current_sub_group_cues:
                        print(f"  [CRITICAL WARNING] Could not form sub-group. Adding remaining cues individually.")
                        final_cues.extend(cues_to_distribute)
                        break
                    
                    combined_text = "\n".join([c['text'] for c in current_sub_group_cues])
                    # --- FIX: Also combine the 'lines' for the sub-groups. ---
                    combined_lines = [line for cue in current_sub_group_cues for line in cue['lines']]
                    final_cues.append({
                        'text': combined_text, 
                        'start': current_sub_group_cues[0]['start'], 
                        'end': current_sub_group_cues[-1]['end'], 
                        'status': 'grouped',
                        'lines': combined_lines
                    })
                    print(f"  [GROUPED] Created split {current_sub_group_line_count}-line sub-group.")
                    cues_to_distribute = cues_to_distribute[len(current_sub_group_cues):]
                i = end_anchor_index + 1
                continue
        
        final_cues.append(start_cue)
        i += 1
        
    return final_cues

def finalize_and_enforce_timing(cues):
    """
    Finalizes cue timings, applying a dynamic cushion for readability and ensuring
    that cues do not visually overlap.
    """
    # --- CONFIGURATION FOR NEW DYNAMIC "CUSHION" TIMING ---
    READING_WPM = 220
    REACTION_CUSHION_S = 0.60

    final_timed_lyrics = []
    for cue in cues:
        start_time, end_time, text, status = cue['start'], cue['end'], cue['text'], cue['status']

        # --- THIS IS THE UPDATED LOGIC ---
        # 1. Calculate the actual duration the line was sung.
        sung_duration = end_time - start_time
        if sung_duration <= 0: sung_duration = 0.1 # Avoid zero/negative durations

        # 2. Calculate a dynamic cushion. It's the GREATER of our configured minimum,
        #    or a percentage of the sung duration. This gives held notes a longer tail.
        dynamic_cushion = max(MIN_READABILITY_CUSHION_S, sung_duration * READABILITY_CUSHION_FACTOR)
        extended_end_time = end_time + dynamic_cushion
        
        # 3. Calculate a minimum required time based on a comfortable reading speed (WPM).
        word_count = len(text.split())
        min_reading_time = ((word_count / READING_WPM) * 60) if word_count > 0 else 0
        minimum_duration = min_reading_time + REACTION_CUSHION_S

        # 4. The final duration is the LONGER of the (dynamically extended) performance
        #    or the minimum reading time.
        final_duration = max(extended_end_time - start_time, minimum_duration)
        # --- END OF UPDATE ---

        # Pass the status through in the final output tuple
        final_timed_lyrics.append(((start_time, start_time + final_duration), text, status))

    # --- Overlap Prevention (No changes needed here) ---
    for i in range(len(final_timed_lyrics) - 1):
        if final_timed_lyrics[i][0][1] > final_timed_lyrics[i+1][0][0]:
            new_end_time = final_timed_lyrics[i+1][0][0] - 0.05
            final_timed_lyrics[i] = ((final_timed_lyrics[i][0][0], new_end_time), final_timed_lyrics[i][1], final_timed_lyrics[i][2])

    return final_timed_lyrics

# --- NEW HELPER FUNCTION FOR BLURRED SHADOW TEXT ---
def _get_text_width(text, font):
    """A small helper to get the pixel width of a string."""
    bbox = ImageDraw.Draw(Image.new("RGBA", (1,1))).textbbox((0,0), text, font=font)
    return bbox[2] - bbox[0]

def _wrap_text_advanced(text, font, max_width):
    """
    The definitive text wrapper. Wraps only when necessary. It evaluates all
    possible break points (punctuation and spaces) and chooses the one that
    produces the most balanced lines, with a strong preference for punctuation.
    """
    final_lines = []
    
    for paragraph in text.split('\n'):
        if _get_text_width(paragraph, font) <= max_width:
            final_lines.append(paragraph)
            continue
            
        words = paragraph.split(' ')
        possible_breaks = []

        # --- Step 1: Identify all valid break points and score them ---
        for i in range(1, len(words)):
            line1 = " ".join(words[:i])
            line2 = " ".join(words[i:])
            
            # A break point is only valid if both resulting lines fit on the screen
            if _get_text_width(line1, font) <= max_width and _get_text_width(line2, font) <= max_width:
                # Calculate the "balance score" - lower is better (more balanced)
                balance_score = abs(len(line1) - len(line2))
                
                # Check if this break is after a punctuation mark
                is_punctuation_break = words[i-1].endswith((',', '!', '?'))
                
                # Heavily prioritize punctuation by giving it a massive score reduction
                if is_punctuation_break:
                    balance_score -= 1000 # Make this break extremely attractive
                
                possible_breaks.append({'index': i, 'score': balance_score})
        
        # --- Step 2: Choose the best break point ---
        if not possible_breaks:
            # Fallback for very long words or other rare cases
            current_line = ""
            for word in words:
                if _get_text_width(f"{current_line} {word}".strip(), font) > max_width:
                    final_lines.append(current_line)
                    current_line = word
                else:
                    current_line = f"{current_line} {word}".strip()
            final_lines.append(current_line)
            continue

        # Find the break with the best (lowest) score
        best_break = min(possible_breaks, key=lambda x: x['score'])
        
        # --- Step 3: Apply the best break ---
        break_index = best_break['index']
        final_lines.append(" ".join(words[:break_index]))
        final_lines.append(" ".join(words[break_index:]))
            
    return "\n".join(final_lines)


def create_blurred_text_image(text, font_file, font_size, text_color, shadow_color, shadow_offset, blur_radius, max_width=None):
    """
    Creates a PIL Image of text with a blurred drop shadow, with optional text wrapping.
    """
    try:
        font = ImageFont.truetype(font_file, font_size)
    except IOError:
        print(f"  [WARNING] Font file not found at '{font_file}'. Using default font.")
        font = ImageFont.load_default()
    # Wrap text if a max_width is provided using the new advanced function
    if max_width:
        text = _wrap_text_advanced(text, font, max_width)
    
    # Use your finalized interline spacing for better readability
    interline_spacing = 12 
    bbox = ImageDraw.Draw(Image.new("RGBA", (1,1))).multiline_textbbox((0,0), text, font=font, align='center', spacing=interline_spacing)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Create canvas with generous padding for the blur effect
    padding = int(blur_radius * 4)
    canvas_w = int(text_w + abs(shadow_offset[0]) + padding)
    canvas_h = int(text_h + abs(shadow_offset[1]) + padding)

    # Calculate drawing position to center the text block within the padded canvas
    draw_pos = (padding // 2 - bbox[0], padding // 2 - bbox[1])

    # Draw shadow on a separate layer
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_pos = (draw_pos[0] + shadow_offset[0], draw_pos[1] + shadow_offset[1])
    shadow_draw.multiline_text(shadow_pos, text, font=font, fill=shadow_color, align='center', spacing=interline_spacing)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Draw main text on its own layer
    text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.multiline_text(draw_pos, text, font=font, fill=text_color, align='center', spacing=interline_spacing)

    # Composite the text over the shadow and return the final image
    return Image.alpha_composite(shadow_layer, text_layer)

def enforce_visual_line_limit(grouped_cues, font_file, font_size, max_width, max_lines=6):
    """
    Recursively processes cues to ensure none will render with more than the max_lines limit.
    If a cue exceeds the limit, it's split, and its parts are re-checked until all are valid.
    """
    print("\n--- Enforcing 6-Visual-Line Limit ---")
    
    # Load the font once for efficiency
    try:
        font = ImageFont.truetype(font_file, font_size)
    except IOError:
        font = ImageFont.load_default()
        
    final_cues = []
    # Create a "to-do" list that we can add to while we're processing it
    cues_to_process = list(grouped_cues)
    
    # This loop will continue until we have checked and approved every single cue
    while cues_to_process:
        # Get the next cue from the front of the list
        cue = cues_to_process.pop(0)
        
        # Get the wrapped text to determine the final visual line count
        wrapped_text = _wrap_text_advanced(cue['text'], font, max_width)
        num_visual_lines = wrapped_text.count('\n') + 1
        
        # If the cue is within the limit, it's approved. Add it to our final list.
        if num_visual_lines <= max_lines:
            final_cues.append(cue)
            continue

        # --- RECURSIVE SPLITTING LOGIC ---
        print(f"  [SPLITTING] A cue wrapped to {num_visual_lines} lines (limit is {max_lines}). It will be split.")
        
        # The 'lines' key holds the original, smaller LRC lines that made up this group
        original_lines = cue['lines']
        
        # Edge Case: If a group is made of only ONE original line but still wraps
        # to more than max_lines, it cannot be split further. We must accept it.
        if len(original_lines) < 2:
            print(f"  [WARNING] A single lyric line is too long and wraps to {num_visual_lines} lines. Cannot split further.")
            final_cues.append(cue)
            continue
            
        # Simple and robust split: divide the original lines into two halves
        midpoint = math.ceil(len(original_lines) / 2)
        
        # Create the first new cue from the first half of the lines
        first_half_lines = original_lines[:midpoint]
        cue1 = {
            'text': "\n".join([line['text'] for line in first_half_lines]),
            'start': first_half_lines[0]['start'],
            'end': first_half_lines[-1]['end'],
            'status': 'split',
            'lines': first_half_lines
        }
        
        # Create the second new cue from the second half
        second_half_lines = original_lines[midpoint:]
        cue2 = None
        if second_half_lines:
            cue2 = {
                'text': "\n".join([line['text'] for line in second_half_lines]),
                'start': second_half_lines[0]['start'],
                'end': second_half_lines[-1]['end'],
                'status': 'split',
                'lines': second_half_lines
            }
        
        # *** THE CRITICAL STEP ***
        # Instead of approving these new cues, add them back to the front of the 
        # "to-do" list. They will be re-evaluated in the next loop iterations.
        if cue2:
            cues_to_process.insert(0, cue2)
        cues_to_process.insert(0, cue1)
            
    print("  -> Visual line limit enforcement complete.")
    return final_cues

def cleanup_bakeoff_assets(all_audio_paths, winning_audio_path, asset_folder):
    """
    Cleans up all temporary files created during the bake-off,
    sparing only the final winning audio file.
    """
    print("\n--- Cleaning up all temporary bake-off assets ---")
    
    # Clean up all demucs folders within the correct asset folder
    demucs_folders = glob.glob(os.path.join(asset_folder, "demucs_separated_*"))
    for folder in demucs_folders:
        try:
            shutil.rmtree(folder)
            print(f"  -> Deleted demucs folder: {folder}")
        except OSError as e:
            print(f"  -> [WARNING] Could not delete folder {folder}: {e}")
            
    # Clean up all audio files except the winner
    for audio_path in all_audio_paths:
        base, _ = os.path.splitext(audio_path)
        trimmed_path = f"{base}_trimmed.wav" # Match the .wav extension used in trim_silence
        
        # Delete the trimmed version if it exists
        if os.path.exists(trimmed_path):
            try:
                os.remove(trimmed_path)
                print(f"  -> Deleted trimmed audio: {os.path.basename(trimmed_path)}")
            except OSError as e:
                print(f"  -> [WARNING] Could not delete file {trimmed_path}: {e}")
        
        # Delete the original downloaded audio IF it's not the final winner
        if audio_path != winning_audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"  -> Deleted unused source audio: {os.path.basename(audio_path)}")
            except OSError as e:
                print(f"  -> [WARNING] Could not delete file {audio_path}: {e}")

def generate_video(audio_path, vocal_track_path, background_path, timed_cues_with_status, output_filename, lower_third_video_path=None):
    # Group fallbacks first, applying the time-based constraints
    grouped_lyrics_with_status = group_fallbacks(timed_cues_with_status)

    # Enforce the visual line limit before finalizing timing
    w, h = 1920, 1080
    max_text_width = w * 0.9
    line_limit_enforced_cues = enforce_visual_line_limit(
        grouped_lyrics_with_status, FONT_FILE, FONT_SIZE, max_text_width, max_lines=6
    )
    
    # Finalize the timing and pass the cue status through
    final_timed_lyrics_with_status = finalize_and_enforce_timing(line_limit_enforced_cues)

    # Detect long vocal silences
    forced_end_times = detect_long_silences(vocal_track_path)
    
    # Apply silence cuts selectively based on cue status
    final_timed_lyrics = apply_silence_timing(final_timed_lyrics_with_status, forced_end_times)

    print("\n--- Assembling Video ---")
    audio_clip = AudioFileClip(audio_path)
    
    try:
        bg_pil_img = Image.open(background_path)
        original_width, original_height = bg_pil_img.size
        aspect_ratio = original_height / original_width
        new_height = int(w * aspect_ratio)
        resized_bg_pil_img = bg_pil_img.resize((w, new_height), Image.Resampling.LANCZOS)
        background_clip = ImageClip(np.array(resized_bg_pil_img), duration=audio_clip.duration)
    except Exception as e:
        print(f"  [FATAL ERROR] Could not load or resize background image: {e}")
        return

    snow_overlay_clip = None
    if USE_SNOW_OVERLAY:
        # --- START OF NEW, OPTIMIZED LOGIC ---
        if os.path.exists(SNOW_OVERLAY_PATH):
            print(f"\n--- Loading and preparing pre-rendered overlay: {os.path.basename(SNOW_OVERLAY_PATH)} ---")
            try:
                # Load the pre-rendered clip. NO .set_opacity() is needed as it's baked into the file.
                long_snow_clip = (VideoFileClip(SNOW_OVERLAY_PATH, has_mask=True)
                                  .without_audio()
                                  .resize(height=h))

                # Check if the song is shorter than the overlay (this will be the normal case)
                if audio_clip.duration <= long_snow_clip.duration:
                    print("  -> Trimming overlay to match audio duration.")
                    snow_overlay_clip = long_snow_clip.set_duration(audio_clip.duration)
                else:
                    # Handle the rare edge case where the song is longer than the pre-rendered overlay
                    print("  -> Audio is longer than overlay. Looping overlay.")
                    num_loops = math.ceil(audio_clip.duration / long_snow_clip.duration)
                    looped_clips = [long_snow_clip] * num_loops
                    snow_overlay_clip = concatenate_videoclips(looped_clips)
                    snow_overlay_clip = snow_overlay_clip.set_duration(audio_clip.duration)
                
                print("  -> Snow overlay prepared successfully.")

            except Exception as e:
                print(f"  [WARNING] Could not load or process pre-rendered overlay. Error: {e}")
                snow_overlay_clip = None
        else:
            print(f"\n[INFO] Snow overlay enabled, but file not found at '{SNOW_OVERLAY_PATH}'. Skipping.")
        # --- END OF NEW LOGIC ---

    all_lyric_clips = []
    
    for ((start, end), text) in final_timed_lyrics:
        duration = end - start
        if duration <= 0: continue

        sanitized_text = sanitize_for_rendering(text)
        if not sanitized_text.strip():
            print(f"  [SKIPPING] Line was fully unrenderable: '{text.replace(chr(10), ' ')}'")
            continue
        
        try:
            lyric_pil_img = create_blurred_text_image(text=sanitized_text, font_file=FONT_FILE, font_size=FONT_SIZE, text_color=TEXT_COLOR, shadow_color=SHADOW_COLOR, shadow_offset=SHADOW_OFFSET, blur_radius=SHADOW_BLUR_RADIUS, max_width=max_text_width)
            final_lyric_clip = ImageClip(np.array(lyric_pil_img))
        except Exception as e:
            print(f"  [WARNING] Could not create text clip for '{sanitized_text.replace(chr(10), ' ')}...'. Error: {e}")
            continue
        
        final_lyric_clip = final_lyric_clip.set_position(('center', 'center')).set_start(start).set_duration(duration).crossfadein(FADE_DURATION).crossfadeout(FADE_DURATION)
        all_lyric_clips.append(final_lyric_clip)
        
    print(f"Created {len(all_lyric_clips)} final TextClips.")
    
    final_layers = [background_clip]
    if snow_overlay_clip:
        final_layers.append(snow_overlay_clip)
    final_layers.extend(all_lyric_clips)
    
    overlay_clip = None
    if lower_third_video_path and os.path.exists(lower_third_video_path):
        print(f"\n--- Overlaying video file: {os.path.basename(lower_third_video_path)} ---")
        try:
            overlay_clip = VideoFileClip(lower_third_video_path, has_mask=True).set_start(15)
            final_layers.append(overlay_clip)
            print("  -> Overlay will start at 0:15.")
        except Exception as e:
            print(f"  [WARNING] Could not load or process overlay file. Error: {e}")
            overlay_clip = None
    else:
        print(f"\n[INFO] Lower third video not found at '{lower_third_video_path}'. Skipping overlay.")
    
    final_clip = CompositeVideoClip(final_layers, size=(w,h))
    final_clip.audio = audio_clip
    final_clip = final_clip.set_duration(audio_clip.duration)
    
    try:
        codec, preset, ffmpeg_params = ('libx264', CPU_PRESET, ['-preset', CPU_PRESET, '-b:v', '3500k'])
        if USE_GPU and check_nvenc_support():
            codec, preset, ffmpeg_params = ('h264_nvenc', NVENC_PRESET, ['-preset', NVENC_PRESET, '-pix_fmt', 'yuv420p', '-b:v', '3500k'])
        
        print(f"\n--- Using Encoder: {codec.upper()} ({'GPU' if 'nvenc' in codec else 'CPU'}) with a target bitrate of 3500kbps ---")
        final_clip.write_videofile(output_filename, codec=codec, audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, fps=30, logger='bar', ffmpeg_params=ffmpeg_params)
        print()
        print(f"\nSuccessfully created video: {output_filename}")

    finally:
        print("\n--- Cleaning up resources ---")
        if overlay_clip:
            overlay_clip.close()
            print("  -> Overlay clip resources released.")
        if snow_overlay_clip:
            snow_overlay_clip.close()
            print("  -> Snow overlay clip resources released.")


def find_best_alignment(song_info, lrc_candidate_paths, audio_path, asset_folder, model):
    """
    Takes a SINGLE audio file and a list of LRC files, runs a full bake-off,
    and returns the best combination without generating a video.
    DOES NOT CLEAN UP ITS OWN ASSETS.
    """
    print(f"\n--- Running Bake-off for Audio Source: {os.path.basename(audio_path)} ---")
    
    # --- 1. PREPARE AUDIO VARIANTS ---
    audio_variants = {'untrimmed': {'path': audio_path}}
    trimmed_path = trim_silence_from_audio(audio_path)
    if trimmed_path:
        audio_variants['trimmed'] = {'path': trimmed_path}

    for name, data in audio_variants.items():
        audio_basename = os.path.splitext(os.path.basename(data['path']))[0]
        demucs_dir_name = f"{audio_basename}"
        vocal_path, demucs_dir = separate_vocals(data['path'], demucs_dir_name, asset_folder)
        if not vocal_path:
            return None
        data['vocal_path'] = vocal_path
        data['demucs_dir'] = demucs_dir
        data['variant_name'] = name 

    # --- 3. EVALUATE ALL COMBINATIONS ---
    all_results = []
    for lrc_path in lrc_candidate_paths:
        display_cues = preprocess_lyrics_for_display(lrc_path)
        if not display_cues: continue
        
        candidate_is_explicit = is_lrc_explicit(lrc_path)

        for variant_name, variant_data in audio_variants.items():
            for source_name, audio_path_to_align in {"vocals_only": variant_data['vocal_path'], "full_mix": variant_data['path']}.items():
                
                # Announce the current test combination
                print(f"\n--- Aligning [{os.path.basename(lrc_path)}] against [{variant_name.upper()}/{source_name.upper()}] ---")
                
                aligned_words = align_lyrics_to_vocals(audio_path_to_align, lrc_path, model)
                if not aligned_words: continue
                
                timed_cues_with_status = get_precise_timings_for_cues(display_cues, aligned_words, verbose_logging=False)
                
                if not timed_cues_with_status: continue
                
                fallback_count = sum(1 for cue in timed_cues_with_status if cue['status'] == 'fallback')
                total_cues = len(timed_cues_with_status)
                fallback_percentage = (fallback_count / total_cues) * 100

                # --- NEW: IMMEDIATE LOGGING ---
                print(f"  -> Fallback: {fallback_percentage:.2f}% ({fallback_count}/{total_cues})")

                all_results.append({
                    'fallback_percentage': fallback_percentage, 'lrc_path': lrc_path,
                    'lrc_is_explicit': candidate_is_explicit, 'timed_cues': timed_cues_with_status,
                    'original_audio_path': audio_path,
                    'winning_audio_path': variant_data['path'],
                    'winning_vocal_path': variant_data['vocal_path'],
                    'winning_trim_status': variant_data['variant_name'],
                    'winning_align_source': source_name
                })

    if not all_results:
        return None

    # --- 4. SELECT WINNER ---
    best_overall_result = min(all_results, key=lambda x: x['fallback_percentage'])
    winner = best_overall_result
    
    if song_info.get('is_explicit', False):
        EXPLICIT_SCORE_TOLERANCE = 8.0 
        explicit_candidates = [res for res in all_results if res['lrc_is_explicit']]
        if explicit_candidates:
            best_explicit_result = min(explicit_candidates, key=lambda x: x['fallback_percentage'])
            if best_explicit_result['fallback_percentage'] - best_overall_result['fallback_percentage'] <= EXPLICIT_SCORE_TOLERANCE:
                winner = best_explicit_result

    print(f"\n  -> Best fallback for this audio source: {winner['fallback_percentage']:.2f}% with '{os.path.basename(winner['lrc_path'])}'")
    
    return winner


def generate_video_from_winner(winner_data, background_path, output_filename, model, lower_third_video_path=None):
    """
    Takes a winning result object from the bake-off and generates the final video.
    """
    # --- NEW: CALCULATE COUNTS FOR THE SUMMARY ---
    final_timed_cues = winner_data['timed_cues']
    final_fallback_count = sum(1 for cue in final_timed_cues if cue['status'] == 'fallback')
    final_total_cues = len(final_timed_cues)

    # --- NEW: ENHANCED WINNER SUMMARY ---
    winning_combo_summary = (
        f"  - Audio File: '{os.path.basename(winner_data['original_audio_path'])}'\n"
        f"  - Trim Status: '{winner_data['winning_trim_status']}'\n"
        f"  - Align Source: '{winner_data['winning_align_source']}'\n"
        f"  - Lyric File: '{os.path.basename(winner_data['lrc_path'])}'\n"
        f"  - Final Score: {winner_data['fallback_percentage']:.2f}% ({final_fallback_count}/{final_total_cues})"
    )
    print("\n--- Generating Final Video From Best Overall Result ---")
    print(winning_combo_summary)

    # --- Re-run timing with verbose logging for the winner ---
    print("\n--- Re-running final alignment to generate detailed log ---")
    
    align_path = winner_data['winning_audio_path']
    if winner_data['winning_align_source'] == 'vocals_only':
        align_path = winner_data['winning_vocal_path']
        
    final_aligned_words = align_lyrics_to_vocals(align_path, winner_data['lrc_path'], model)
    final_display_cues = preprocess_lyrics_for_display(winner_data['lrc_path'])
    
    get_precise_timings_for_cues(final_display_cues, final_aligned_words, verbose_logging=True)

    # --- Generate the video ---
    generate_video(
        audio_path=winner_data["winning_audio_path"], 
        vocal_track_path=winner_data["winning_vocal_path"],
        background_path=background_path, 
        timed_cues_with_status=winner_data['timed_cues'], 
        output_filename=output_filename,
        lower_third_video_path=lower_third_video_path
    )

    return winner_data['fallback_percentage']

if __name__ == '__main__':
    if FFMPEG_BINARY_PATH and os.path.exists(FFMPEG_BINARY_PATH):
        print(f"--- Setting custom FFmpeg path: {FFMPEG_BINARY_PATH} ---")
        os.environ['FFMPEG_BINARY'] = FFMPEG_BINARY_PATH
        
    print("--- Running video_generator.py directly in Test Mode ---")
    
    # --- Define test assets ---
    test_asset_folder = "test_assets"
    test_song_info = {
        'name': 'Test Song',
        'artist': 'Test Artist',
        'is_explicit': True
    }
    test_lrc_paths = [os.path.join(test_asset_folder, f) for f in os.listdir(test_asset_folder) if f.startswith('lyrics_candidate_') and f.endswith('.lrc')]
    test_audio_files = glob.glob(os.path.join(test_asset_folder, "test_song*.*"))
    test_audio_path = test_audio_files[0] if test_audio_files else None
    test_background_path = os.path.join(test_asset_folder, "test_background.jpg")

    if not test_audio_path or not os.path.exists(test_audio_path):
        print(f"\n[ERROR] Test audio file not found at '{test_audio_path}'. Cannot proceed.")
    elif not test_lrc_paths:
        print("\n[ERROR] No 'lyrics_candidate_*.lrc' files found in 'test_assets' for direct testing.")
    else:
        # --- NEW LOGIC: Mimic the new two-step process ---
        print("\n--- Step 1: Finding best alignment for the test audio ---")
        
        # We use a placeholder audio path here. For a real test, you could use one of the
        # downloaded files or a dedicated test mp3.
        winner_data = find_best_alignment(test_song_info, test_lrc_paths, test_audio_path, asset_folder=test_asset_folder, model=None) # Model not needed for this part in test
        
        if winner_data:
            print("\n--- Step 2: Found a winning alignment. Generating test video. ---")
            generate_video_from_winner(
                winner_data,
                test_background_path,
                "lyric_video_FINAL.mp4",
                lower_third_video_path="assets/sub_lower_third.mov" # Default for testing
            )
            print("\n--- Test video generation complete. ---")
        else:
            print("\n--- Test failed: Could not find a valid alignment. ---")