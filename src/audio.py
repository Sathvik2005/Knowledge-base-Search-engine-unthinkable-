"""
Audio processing module for Answer IQ.
Handles interactive voice recording and speech-to-text using Whisper via HuggingFace.
"""

import tempfile
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch

from .config import DEVICE, SUPPORTED_AUDIO_FORMATS


@dataclass
class TranscriptionResult:
    """Represents the result of audio transcription."""
    text: str
    language: str
    confidence: float
    duration: float
    success: bool
    error_message: Optional[str] = None


class WhisperTranscriber:
    """
    Speech-to-text transcription using OpenAI Whisper via HuggingFace Transformers.
    Supports multilingual transcription and various audio formats.
    Uses the whisper-large-v3 model for high accuracy.
    """
    
    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_name: HuggingFace model identifier for Whisper
        """
        self.model_name = model_name
        self.pipe = None
        self.processor = None
        self.model = None
        self._loaded = False
    
    @staticmethod
    def is_available() -> bool:
        """Check if Whisper is available."""
        try:
            from transformers import pipeline
            return True
        except ImportError:
            return False
    
    def load(self) -> bool:
        """
        Load the Whisper model using HuggingFace Transformers.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True
        
        try:
            from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
            
            # Determine device and dtype
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Try loading the specified model
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                self.model.to(device)
                
                # Create pipeline
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=device,
                )
            except Exception as e:
                print(f"Error loading {self.model_name}: {e}")
                print("Falling back to whisper-base...")
                # Fallback to smaller model
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            self._loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return False
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code (e.g., 'en', 'es', 'fr')
                     If None, language is auto-detected
            
        Returns:
            TranscriptionResult object
        """
        if not self._loaded:
            if not self.load():
                return TranscriptionResult(
                    text="",
                    language="",
                    confidence=0.0,
                    duration=0.0,
                    success=False,
                    error_message="Failed to load Whisper model."
                )
        
        try:
            # Prepare generation kwargs
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
            
            # Perform transcription
            result = self.pipe(
                audio_path,
                generate_kwargs=generate_kwargs if generate_kwargs else None,
                return_timestamps=True
            )
            
            # Extract text
            text = result.get("text", "").strip()
            
            # Get chunks for duration calculation
            chunks = result.get("chunks", [])
            if chunks and chunks[-1].get("timestamp"):
                timestamp = chunks[-1]["timestamp"]
                duration = timestamp[1] if timestamp[1] else 0.0
            else:
                duration = 0.0
            
            return TranscriptionResult(
                text=text,
                language=language or "auto",
                confidence=0.9,
                duration=duration,
                success=True
            )
            
        except Exception as e:
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                duration=0.0,
                success=False,
                error_message=f"Transcription error: {str(e)}"
            )
    
    def transcribe_from_upload(self, uploaded_file) -> TranscriptionResult:
        """
        Transcribe from a Streamlit uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            TranscriptionResult object
        """
        suffix = Path(uploaded_file.name).suffix.lower()
        
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                duration=0.0,
                success=False,
                error_message=f"Unsupported audio format: {suffix}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            result = self.transcribe(tmp_path)
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def transcribe_from_bytes(self, audio_bytes: bytes, 
                               format_hint: str = ".wav") -> TranscriptionResult:
        """
        Transcribe from audio bytes (for microphone recording).
        
        Args:
            audio_bytes: Raw audio bytes
            format_hint: File extension hint for the audio format
            
        Returns:
            TranscriptionResult object
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=format_hint) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            result = self.transcribe(tmp_path)
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def transcribe_from_array(self, audio_array: np.ndarray, 
                               sampling_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe from numpy audio array (for real-time recording).
        
        Args:
            audio_array: Numpy array of audio samples
            sampling_rate: Sample rate of the audio
            
        Returns:
            TranscriptionResult object
        """
        if not self._loaded:
            if not self.load():
                return TranscriptionResult(
                    text="",
                    language="",
                    confidence=0.0,
                    duration=0.0,
                    success=False,
                    error_message="Failed to load Whisper model."
                )
        
        try:
            # Ensure audio is float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize if needed
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / 32768.0
            
            # Create input dict for pipeline
            audio_input = {
                "array": audio_array,
                "sampling_rate": sampling_rate
            }
            
            result = self.pipe(audio_input, return_timestamps=True)
            text = result.get("text", "").strip()
            
            chunks = result.get("chunks", [])
            if chunks and chunks[-1].get("timestamp"):
                timestamp = chunks[-1]["timestamp"]
                duration = timestamp[1] if timestamp[1] else len(audio_array) / sampling_rate
            else:
                duration = len(audio_array) / sampling_rate
            
            return TranscriptionResult(
                text=text,
                language="auto",
                confidence=0.9,
                duration=duration,
                success=True
            )
            
        except Exception as e:
            return TranscriptionResult(
                text="",
                language="",
                confidence=0.0,
                duration=0.0,
                success=False,
                error_message=f"Transcription error: {str(e)}"
            )


class AudioProcessor:
    """
    High-level audio processing interface.
    Coordinates transcription and query preparation.
    """
    
    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        self.transcriber = WhisperTranscriber(model_name=model_name)
        self._model_loaded = False
    
    def ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded before processing."""
        if not self._model_loaded:
            self._model_loaded = self.transcriber.load()
        return self._model_loaded
    
    def process_audio_query(self, audio_source, 
                           source_type: str = "upload") -> Tuple[str, Optional[str]]:
        """
        Process audio input and extract text query.
        
        Args:
            audio_source: Audio file (upload, bytes, or numpy array)
            source_type: Type of source ("upload", "bytes", or "array")
            
        Returns:
            Tuple of (transcribed_text, error_message)
        """
        if source_type == "upload":
            result = self.transcriber.transcribe_from_upload(audio_source)
        elif source_type == "bytes":
            result = self.transcriber.transcribe_from_bytes(audio_source)
        elif source_type == "array":
            audio_array, sample_rate = audio_source
            result = self.transcriber.transcribe_from_array(audio_array, sample_rate)
        else:
            return "", f"Unknown source type: {source_type}"
        
        if result.success:
            return result.text, None
        else:
            return "", result.error_message
    
    def process_microphone_input(self, audio_data) -> Tuple[str, Optional[str]]:
        """
        Process microphone input from Streamlit audio_input.
        
        Args:
            audio_data: Audio data from st.audio_input
            
        Returns:
            Tuple of (transcribed_text, error_message)
        """
        if audio_data is None:
            return "", "No audio recorded"
        
        # Streamlit's audio_input returns bytes
        if isinstance(audio_data, bytes):
            return self.process_audio_query(audio_data, "bytes")
        elif hasattr(audio_data, 'getvalue'):
            return self.process_audio_query(audio_data.getvalue(), "bytes")
        elif hasattr(audio_data, 'read'):
            return self.process_audio_query(audio_data.read(), "bytes")
        
        return "", "Unsupported audio format from microphone"
    
    def get_supported_formats(self) -> list:
        """Return list of supported audio formats."""
        return SUPPORTED_AUDIO_FORMATS
    
    def is_available(self) -> bool:
        """Check if audio processing is available."""
        return WhisperTranscriber.is_available()
