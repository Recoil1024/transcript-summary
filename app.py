# Chunk 1: Imports and Configurations

import streamlit as st
from pydub import AudioSegment
import io
import base64
from groq import Groq
import tempfile
import os
import pandas as pd

# Model configurations
WHISPER_MODEL = "whisper-large-v3"
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Japanese": "ja",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}

def setup_groq_client(api_key):
    """Initialize Groq client with API key"""
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None

def extract_audio_from_video(video_file):
    """Extract and format audio for transcription"""
    try:
        # Read file into memory
        video_bytes = io.BytesIO(video_file.read())
        
        # Load audio and set required parameters
        audio = AudioSegment.from_file(video_bytes)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Convert to MP3 format (supported by Groq)
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3", parameters=["-ac", "1"])
        buffer.seek(0)
        
        return buffer.read()
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None
# Chunk 2: Audio Processing and Transcription Functions

def transcribe_audio_groq(audio_data, api_key, language='en'):
    """Transcribe audio using Groq's speech-to-text API with language support"""
    try:
        client = Groq(api_key=api_key)
        
        # Convert audio data to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Call Groq's speech-to-text API
        response = client.audio.transcriptions.create(
            file=("audio.mp3", audio_base64),
            model=WHISPER_MODEL,
            language=language
        )
        
        return response.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def process_audio_file(file_data, api_key, language):
    """Process uploaded audio file and generate transcript"""
    try:
        # Extract audio if it's a video file, or use directly if it's audio
        audio_data = extract_audio_from_video(file_data)
        
        if audio_data:
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio_groq(audio_data, api_key, language)
                
                if transcript:
                    return {
                        'status': 'success',
                        'transcript': transcript,
                        'error': None
                    }
                else:
                    return {
                        'status': 'error',
                        'transcript': None,
                        'error': 'Transcription failed'
                    }
        return {
            'status': 'error',
            'transcript': None,
            'error': 'Audio extraction failed'
        }
    except Exception as e:
        return {
            'status': 'error',
            'transcript': None,
            'error': str(e)
        }

def format_transcript(transcript):
    """Format the transcript for display and download"""
    try:
        # Split transcript into sentences for better readability
        sentences = transcript.split('. ')
        formatted_text = '\n'.join([f"{i+1}. {sentence.strip()}." 
                                  for i, sentence in enumerate(sentences)])
        return formatted_text
    except Exception as e:
        st.error(f"Error formatting transcript: {str(e)}")
        return transcript
# Chunk 3: Transcription and Processing Functions

def transcribe_audio_groq(audio_data, api_key, language='en'):
    """Transcribe audio using Groq's speech-to-text API with language support"""
    try:
        client = Groq(api_key=api_key)
        
        # Convert audio data to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Call Groq's speech-to-text API
        response = client.audio.transcriptions.create(
            file=("audio.mp3", audio_base64),
            model=WHISPER_MODEL,
            language=language
        )
        
        return response.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def process_audio_file(file_data, api_key, language):
    """Process uploaded audio file and generate transcript"""
    try:
        # Extract audio if it's a video file, or use directly if it's audio
        audio_data = extract_audio_from_video(file_data)
        
        if audio_data:
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio_groq(audio_data, api_key, language)
                
                if transcript:
                    return {
                        'status': 'success',
                        'transcript': transcript,
                        'error': None
                    }
                else:
                    return {
                        'status': 'error',
                        'transcript': None,
                        'error': 'Transcription failed'
                    }
        return {
            'status': 'error',
            'transcript': None,
            'error': 'Audio extraction failed'
        }
    except Exception as e:
        return {
            'status': 'error',
            'transcript': None,
            'error': str(e)
        }

def format_transcript(transcript):
    """Format the transcript for display and download"""
    try:
        # Split transcript into sentences for better readability
        sentences = transcript.split('. ')
        formatted_text = '\n'.join([f"{i+1}. {sentence.strip()}." 
                                  for i, sentence in enumerate(sentences)])
        return formatted_text
    except Exception as e:
        st.error(f"Error formatting transcript: {str(e)}")
        return transcript
# Chunk 4: User Interface Components and Main Application Logic

def create_sidebar():
    """Create and configure the sidebar UI elements"""
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter Groq API Key:",
            type="password",
            help="Enter your Groq API key for transcription"
        )
        
        # Language selector
        language = st.selectbox(
            "Select transcription language:",
            list(SUPPORTED_LANGUAGES.keys()),
            help="Choose the language of the audio content"
        )
        
        return {
            'api_key': api_key,
            'language': SUPPORTED_LANGUAGES[language]  # Return language code
        }

def display_file_uploader():
    """Display and handle file upload interface"""
    st.header("Upload Media File")
    
    uploaded_file = st.file_uploader(
        "Choose a video or audio file",
        type=['mp4', 'mp3', 'wav', 'avi', 'mov', 'm4a'],
        help="Upload your media file for transcription"
    )
    
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
            
        return uploaded_file
    return None

def display_results(transcript, file_name):
    """Display transcription results and download options"""
    if transcript:
        st.success("Transcription completed successfully!")
        
        # Display tabs for different views
        tab1, tab2 = st.tabs(["Transcript", "Download"])
        
        with tab1:
            st.markdown("### Transcript")
            st.text_area(
                "Full Transcript",
                transcript,
                height=300,
                disabled=True
            )
        
        with tab2:
            st.markdown("### Download Options")
            # Text file download
            st.download_button(
                label="Download as TXT",
                data=transcript,
                file_name=f"{file_name}_transcript.txt",
                mime="text/plain"
            )
            
            # CSV format with timestamps (if applicable)
            # Assuming transcript is a simple string for now
            csv_data = format_csv_transcript(transcript)
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f"{file_name}_transcript.csv",
                mime="text/csv"
            )

def main():
    st.set_page_config(
        page_title="Groq Speech-to-Text",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ Groq Speech-to-Text Converter")
    st.markdown("Convert your audio/video files to text using Groq's advanced speech recognition")
    
    # Get configuration from sidebar
    config = create_sidebar()
    
    # Main content area
    uploaded_file = display_file_uploader()
    
    if uploaded_file and config['api_key']:
        file_name = os.path.splitext(uploaded_file.name)[0]
        
        with st.spinner("Processing your file..."):
            # Process the uploaded file
            result = process_audio_file(uploaded_file, config['api_key'], config['language'])
            
            if result['status'] == 'success':
                formatted_transcript = format_transcript(result['transcript'])
                display_results(formatted_transcript, file_name)
            else:
                st.error(result['error'])
    
    elif uploaded_file and not config['api_key']:
        st.warning("Please enter your Groq API key in the sidebar to proceed.")
    
    # Display usage instructions
    with st.expander("Usage Instructions"):
        st.markdown("""
        ### How to use:
        1. Enter your Groq API key in the sidebar
        2. Upload an audio or video file
        3. Wait for processing
        4. View and download your transcript
        
        ### Supported file formats:
        - Audio: MP3, WAV, M4A
        - Video: MP4, AVI, MOV
        
        ### Tips:
        - For best results, ensure clear audio quality
        - Larger files may take longer to process
        - Keep your API key secure
        """)      
# Chunk 5: Utility Functions and Final Application Setup

def format_csv_transcript(transcript):
    """Format transcript data for CSV export"""
    try:
        data = []
        lines = transcript.split('\n')
        for i, line in enumerate(lines, 1):
            data.append([i, line])
        
        df = pd.DataFrame(data, columns=['Line', 'Text'])
        return df.to_csv(index=False)
    except Exception as e:
        st.error(f"Error formatting CSV: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Groq Speech-to-Text",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ Groq Speech-to-Text Converter")
    st.markdown("Convert your audio/video files to text using Groq's advanced speech recognition")
    
    # Get configuration from sidebar
    config = create_sidebar()
    
    # Main content area
    uploaded_file = display_file_uploader()
    
    if uploaded_file and config['api_key']:
        file_name = os.path.splitext(uploaded_file.name)[0]
        
        with st.spinner("Processing your file..."):
            # Process the uploaded file
            result = process_audio_file(uploaded_file, config['api_key'], config['language'])
            
            if result['status'] == 'success':
                formatted_transcript = format_transcript(result['transcript'])
                display_results(formatted_transcript, file_name)
            else:
                st.error(result['error'])
    
    elif uploaded_file and not config['api_key']:
        st.warning("Please enter your Groq API key in the sidebar to proceed.")
    
    # Display usage instructions
    with st.expander("Usage Instructions"):
        st.markdown("""
        ### How to use:
        1. Enter your Groq API key in the sidebar
        2. Upload an audio or video file
        3. Wait for processing
        4. View and download your transcript
        
        ### Supported file formats:
        - Audio: MP3, WAV, M4A
        - Video: MP4, AVI, MOV
        
        ### Tips:
        - For best results, ensure clear audio quality
        - Larger files may take longer to process
        - Keep your API key secure
        """)

if __name__ == "__main__":
    main()              