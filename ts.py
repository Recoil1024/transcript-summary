import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
import requests
import json
from urllib.parse import parse_qs, urlparse
import pandas as pd
import os
import tempfile
from pydub import AudioSegment
import io
import base64
from groq import Groq
import google.generativeai as genai
import ollama

# Model configurations
GROQ_MODELS = {
    "Gemma-7B": "gemma-7b-it",
    "Gemma2-9B": "gemma2-9b-it",
    "LLaMA-3.1-70B": "llama-3.1-70b-versatile",
    "LLaMA-3.1-8B": "llama-3.1-8b-instant",
    "LLaMA-3.2-11B": "llama-3.2-11b-text-preview",
    "LLaMA-3.2-11B-Vision": "llama-3.2-11b-vision-preview",
    "LLaMA-3.2-90B": "llama-3.2-90b-text-preview",
    "LLaMA-3.2-90B-Vision": "llama-3.2-90b-vision-preview",
    "LLaMA-3.3-70B": "llama-3.3-70b-versatile",
    "LLaMA-Guard-3-8B": "llama-guard-3-8b",
    "LLaMA3-70B": "llama3-70b-8192",
    "LLaMA3-8B": "llama3-8b-8192",
    "LLaMA3-70B-Tool": "llama3-groq-70b-8192-tool-use-preview",
    "LLaMA3-8B-Tool": "llama3-groq-8b-8192-tool-use-preview",
    "LLaVA-1.5-7B": "llava-v1.5-7b-4096-preview",
    "Mixtral-8x7B": "mixtral-8x7b-32768"
}

GEMINI_MODELS = {
    "Gemini-Pro": "gemini-pro"
}

OLLAMA_MODELS = {
    "Llama2": "llama2",
    "Mistral": "mistral",
    "Gemma": "gemma",
    "Neural-Chat": "neural-chat"
}

WHISPER_MODEL = "whisper-large-v3"

def setup_ai_providers(api_provider, api_key=None, ollama_host=None):
    if api_provider == "Groq":
        return Groq(api_key=api_key)
    elif api_provider == "Google Gemini":
        genai.configure(api_key=api_key)
        return genai
    elif api_provider == "Ollama":
        return ollama.Client(host=ollama_host or "http://localhost:11434")
    return None

def extract_audio_from_video(video_file):
    try:
        video_bytes = io.BytesIO(video_file.read())
        audio = AudioSegment.from_file(video_bytes)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
        audio.export(temp_audio_path, format="wav")
        return temp_audio_path, temp_dir
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None, None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
        return transcript, 'Hindi'
    except NoTranscriptFound:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                return transcript.fetch(), transcript.language_code
        except (TranscriptsDisabled, NoTranscriptFound):
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                return transcript, 'English (auto-generated)'
            except:
                st.warning(f"No transcript available for video: {video_id}")
                return None, None

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def generate_smart_summary(transcript_chunk, chunk_number, total_chunks, api_provider, api_key, model_name, ollama_host=None):
    try:
        client = setup_ai_providers(api_provider, api_key, ollama_host)
        
        prompt = f"""As an expert analyzer, please review this transcript chunk ({chunk_number}/{total_chunks}) and create a detailed analysis that focus on point 1, 2, 3 if related to learning or study but itf it contains recipes or gym or calisthenic related stuff focus on point 4 or generate accordingly:

        1. Key Concepts & Ideas:
           - Main points discussed
           - Important concepts explained
           - Any sequential steps or processes mentioned

        2. Special Elements (if any):
           - Formulas or equations (explain with examples)
           - Methods or techniques (provide practical applications)
           - Tips, tricks, or shortcuts
           - Important definitions or terminology

        3. Timestamps:
           - Mark important moments with their timestamps
           - Highlight crucial explanations or demonstrations
           - Note any practical examples or case studies
        
        4. Recipes:
           - Mark all recipes
           - all of the ingredients
           - micro and macronutrients
           - latest cost online 
           - detailed process to make each recipe

        Transcript chunk:
        {transcript_chunk}"""

        if api_provider == "Groq":
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert educational content analyzer."},
                    {"role": "user", "content": prompt}
                ],
                model=GROQ_MODELS[model_name],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        elif api_provider == "Google Gemini":
            model = client.GenerativeModel(GEMINI_MODELS[model_name])
            response = model.generate_content(prompt)
            return response.text
        elif api_provider == "Ollama":
            response = client.generate(
                model=OLLAMA_MODELS[model_name],
                prompt=prompt,
                stream=False
            )
            return response['response']
    except Exception as e:
        st.error(f"Error generating summary for chunk {chunk_number}: {str(e)}")
        return None

def merge_summaries(summaries, api_provider, api_key, model_name, ollama_host=None):
    try:
        client = setup_ai_providers(api_provider, api_key, ollama_host)
        merge_prompt = f"""Create a comprehensive final summary from these section analyses. The final summary should:
        1. Maintain chronological flow of concepts
        2. Highlight key learning points with their timestamps
        3. Group related concepts together
        4. Explain complex ideas with examples
        5. Preserve important formulas, methods, and tricks
        6. Include practical applications where relevant

        Section summaries:
        {' '.join(summaries)}"""

        if api_provider == "Groq":
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at creating comprehensive educational summaries."},
                    {"role": "user", "content": merge_prompt}
                ],
                model=GROQ_MODELS[model_name],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        elif api_provider == "Google Gemini":
            model = client.GenerativeModel(GEMINI_MODELS[model_name])
            response = model.generate_content(merge_prompt)
            return response.text
        elif api_provider == "Ollama":
            response = client.generate(
                model=OLLAMA_MODELS[model_name],
                prompt=merge_prompt,
                stream=False
            )
            return response['response']
    except Exception as e:
        st.error(f"Error generating final summary: {str(e)}")
        return None

def generate_summary(transcript_text, api_provider, api_key, model_name, ollama_host=None):
    try:
        words = transcript_text.split()
        chunk_size = 1500
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            status_text.text(f"Analyzing part {i+1} of {len(chunks)}...")
            chunk_summary = generate_smart_summary(chunk, i+1, len(chunks), api_provider, api_key, model_name, ollama_host)
            if chunk_summary:
                summaries.append(chunk_summary)
            progress_bar.progress((i + 1) / len(chunks))
        
        status_text.text("Creating final summary...")
        
        if len(summaries) > 1:
            final_summary = merge_summaries(summaries, api_provider, api_key, model_name, ollama_host)
            status_text.text("Summary completed!")
            return final_summary
        else:
            status_text.text("Summary completed!")
            return summaries[0] if summaries else None
    except Exception as e:
        st.error(f"Error in summary generation: {str(e)}")
        return None

def process_video(video_id, api_provider, api_key, model_name, ollama_host=None):
    transcript, language = get_transcript(video_id)
    
    if transcript:
        st.success(f"Transcript found in {language}")
        
        df = pd.DataFrame(transcript)
        df['timestamp'] = df['start'].apply(format_timestamp)
        df = df[['timestamp', 'text']]
        
        transcript_tab, summary_tab = st.tabs(["Transcript", "AI Summary"])
        
        with transcript_tab:
            st.dataframe(df, hide_index=True)
            transcript_text = "\n".join([f"[{format_timestamp(line['start'])}] {line['text']}" 
                                       for line in transcript])
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name=f"{video_id}_{language}_transcript.txt",
                mime="text/plain",
                key=f"download_transcript_{video_id}"
            )
        
        with summary_tab:
            if api_key or api_provider == "Ollama":
                if st.button("Generate Smart Summary", key=f"generate_summary_{video_id}"):
                    with st.spinner(f"Analyzing content using {model_name}..."):
                        full_text = " ".join([f"[{format_timestamp(line['start'])}] {line['text']}" 
                                            for line in transcript])
                        summary = generate_summary(full_text, api_provider, api_key, model_name, ollama_host)
                        if summary:
                            st.markdown(summary)
                            st.download_button(
                                label="Download Summary",
                                data=summary,
                                file_name=f"{video_id}_smart_summary.txt",
                                mime="text/plain",
                                key=f"download_summary_{video_id}"
                            )
            else:
                st.warning("Please configure API settings in the sidebar.")
        return True
    else:
        st.error(f"No transcript available for video ID: {video_id}")
        return False

def get_playlist_videos(playlist_url):
    try:
        query = parse_qs(urlparse(playlist_url).query)
        playlist_id = query["list"][0]
        response = requests.get(f"https://www.youtube.com/playlist?list={playlist_id}")
        if response.status_code != 200:
            st.error("Failed to fetch playlist")
            return []
        
        page_content = response.text
        start_marker = 'var ytInitialData = '
        end_marker = '};'
        start_index = page_content.find(start_marker) + len(start_marker)
        end_index = page_content.find(end_marker, start_index) + 1
        
        if start_index == -1 or end_index == -1:
            return []
            
        json_data = json.loads(page_content[start_index:end_index])
        video_ids = []
        
        try:
            items = json_data['contents']['twoColumnBrowseResultsRenderer']['tabs'][0]['tabRenderer']['content']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['playlistVideoListRenderer']['contents']
            for item in items:
                if 'playlistVideoRenderer' in item:
                    video_ids.append(item['playlistVideoRenderer']['videoId'])
        except:
            st.error("Error parsing playlist data")
            return []
            
        return video_ids
    except Exception as e:
        st.error(f"Error fetching playlist: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="Smart Video Analyzer",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Smart Video Analyzer")
    st.markdown("Extract transcripts and generate intelligent summaries from videos!")
    
    with st.sidebar:
        st.header("AI Provider Configuration")
        
        if 'api_provider' not in st.session_state:
            st.session_state.api_provider = "Groq"
            st.session_state.model_name = list(GROQ_MODELS.keys())[0]
        
        api_provider = st.selectbox(
            "Select AI Provider:",
            ["Groq", "Google Gemini", "Ollama"],
            key="api_provider"
        )
        
        # Reset model name when provider changes
        if 'last_provider' not in st.session_state or st.session_state.last_provider != api_provider:
            if api_provider == "Groq":
                st.session_state.model_name = list(GROQ_MODELS.keys())[0]
            elif api_provider == "Google Gemini":
                st.session_state.model_name = list(GEMINI_MODELS.keys())[0]
            else:
                st.session_state.model_name = list(OLLAMA_MODELS.keys())[0]
            st.session_state.last_provider = api_provider
        
        # Set up provider-specific options
        if api_provider == "Groq":
            model_options = GROQ_MODELS
            if 'groq_api_key' not in st.session_state:
                st.session_state.groq_api_key = ''
            api_key = st.text_input(
                "Enter Groq API Key:",
                type="password",
                value=st.session_state.groq_api_key,
                key="groq_api_key_input"
            )
            st.session_state.groq_api_key = api_key
            ollama_host = None
            
        elif api_provider == "Google Gemini":
            model_options = GEMINI_MODELS
            if 'gemini_api_key' not in st.session_state:
                st.session_state.gemini_api_key = ''
            api_key = st.text_input(
                "Enter Google API Key:",
                type="password",
                value=st.session_state.gemini_api_key,
                key="gemini_api_key_input"
            )
            st.session_state.gemini_api_key = api_key
            ollama_host = None
            
        else:  # Ollama
            model_options = OLLAMA_MODELS
            api_key = None
            ollama_host = st.text_input(
                "Ollama Host (default: http://localhost:11434):",
                value="http://localhost:11434",
                key="ollama_host"
            )
        
        model_name = st.selectbox(
            "Select Model:",
            options=list(model_options.keys()),
            key="model_select"
        )
        st.session_state.model_name = model_name

    input_type = st.radio(
        "Choose input type:",
        ["YouTube Video/Playlist", "Upload Media"],
        key="input_type"
    )
    
    if input_type == "YouTube Video/Playlist":
        source_type = st.radio(
            "Select source:", 
            ["Single Video", "Playlist"],
            key="source_type"
        )
        
        if source_type == "Single Video":
            video_url = st.text_input(
                "Enter YouTube video URL or ID:",
                key="video_url"
            )
            if video_url:
                try:
                    if "youtube.com" in video_url or "youtu.be" in video_url:
                        if "youtube.com/watch?v=" in video_url:
                            video_id = video_url.split("watch?v=")[1].split("&")[0]
                        else:
                            video_id = video_url.split("/")[-1].split("?")[0]
                    else:
                        video_id = video_url
                    
                    st.video(f"https://www.youtube.com/watch?v={video_id}")
                    process_video(video_id, api_provider, api_key, model_name, ollama_host)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
        
        else:  # Playlist
            playlist_url = st.text_input(
                "Enter YouTube playlist URL:",
                key="playlist_url"
            )
            if playlist_url:
                with st.spinner("Fetching playlist videos..."):
                    video_ids = get_playlist_videos(playlist_url)
                
                if video_ids:
                    st.success(f"Found {len(video_ids)} videos in playlist")
                    for i, video_id in enumerate(video_ids, 1):
                        st.subheader(f"Video {i}")
                        st.video(f"https://www.youtube.com/watch?v={video_id}")
                        process_video(video_id, api_provider, api_key, model_name, ollama_host)
                else:
                    st.error("No videos found in playlist or error occurred.")
    
    else:  # Upload Media
        st.info("Upload video or audio file for transcription and analysis")
        st.warning("Supported formats: MP4, MP3, WAV, AVI, MOV, M4A")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['mp4', 'mp3', 'wav', 'avi', 'mov', 'm4a'],
            key="media_uploader"
        )
        
        if uploaded_file:
            if uploaded_file.type.startswith('video'):
                st.video(uploaded_file)
            elif uploaded_file.type.startswith('audio'):
                st.audio(uploaded_file)
            
            if api_key or api_provider == "Ollama":
                audio_path, temp_dir = extract_audio_from_video(uploaded_file)
                if audio_path:
                    try:
                        with st.spinner("Processing audio..."):
                            # Process audio file
                            pass  # Add audio processing logic here
                    finally:
                        os.remove(audio_path)
                        os.rmdir(temp_dir)
            else:
                st.warning("Please configure API settings in the sidebar.")

if __name__ == "__main__":
    main()