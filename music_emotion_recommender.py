import os
import torch
import torch.nn as nn
import tempfile
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try importing VAD_Models.vad with error handling
vad_import_error = None
try:
    import VAD_Models.vad as vad
except ImportError as e:
    vad_import_error = str(e)
    print(f"Warning: Could not import VAD_Models.vad: {e}")
    print("VAD score prediction will not be available. Only using emotion and recommendation functionality.")

from emotion_predictor_from_vad import predict_emotions_from_vad

class MusicEmotionRecommender:
    def __init__(self):
        # Set up embedding model for vector DB queries
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Paths to vector databases
        self.persist_base = os.getenv("VECTOR_DB_BASE_PATH")
        
        # Load vector stores
        self.vector_stores = {}
        for partition in ["arxiv", "pubmed", "blogs"]:
            partition_path = os.path.join(self.persist_base, partition)
            if os.path.exists(partition_path):
                self.vector_stores[partition] = Chroma(
                    persist_directory=partition_path,
                    embedding_function=self.embedding_model
                )
        
        # Check if we have any loaded vector stores
        if not self.vector_stores:
            print("Warning: No vector stores could be loaded. Using fallback recommendation system.")
            
        # Load emotion context from JSON file
        self.emotion_context = self._load_emotion_context()
        
        # Load composition strategies from JSON file
        self.composition_strategies = self._load_composition_strategies()
    
    def _load_emotion_context(self):
        """Load emotion context from JSON file"""
        try:
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.getenv("EMOTION_CONTEXT_PATH", "emotion_context.json"))
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load emotion context from JSON file: {e}")
            # Return a minimal default emotion context
            return {
                "Happy": ["upbeat", "cheerful", "positive"],
                "Sad": ["melancholic", "somber", "reflective"],
                "Neutral": ["balanced", "moderate", "calm"]
            }
    
    def _load_composition_strategies(self):
        """Load composition strategies from JSON file"""
        try:
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.getenv("COMPOSITION_STRATEGIES_PATH", "composition_strategies.json"))
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load composition strategies from JSON file: {e}")
            # Return a minimal default composition strategies
            return {
                "Happy": "Create uplifting music with bright major keys, upbeat tempos, and melodic progressions.",
                "Sad": "Compose melancholic pieces using minor keys, slow tempos, and emotionally resonant chord progressions.",
                "Neutral": "Compose ambient music with balanced elements, moderate tempo, and subtle emotional cues."
            }
    
    def predict_vad_from_audio(self, audio_path):
        """
        Get VAD (Valence, Arousal, Dominance) scores from audio
        
        Parameters:
        - audio_path: Path to the audio file
        
        Returns:
        - Dictionary containing VAD scores
        """
        if vad_import_error is not None:
            print("VAD module could not be imported. Using default VAD scores.")
            return {'valence': 3.0, 'arousal': 3.0, 'dominance': 3.0}
            
        valence_checkpoint_path = os.getenv("VALENCE_CHECKPOINT_PATH")
        ad_checkpoint_path = os.getenv("AD_CHECKPOINT_PATH")
        
        try:
            # Call the predict_emotions function from vad.py
            vad_scores = vad.predict_emotions(audio_path, valence_checkpoint_path, ad_checkpoint_path)
            return vad_scores
        except Exception as e:
            print(f"Error predicting VAD scores: {e}")
            return {'valence': 3.0, 'arousal': 3.0, 'dominance': 3.0}
    
    def get_emotions_from_vad(self, vad_scores):
        """
        Get predicted emotions from VAD scores
        
        Parameters:
        - vad_scores: Dictionary containing 'valence', 'arousal', and 'dominance' values
        
        Returns:
        - List of predicted emotions
        """
        # Call the function from emotion_predictor_from_vad.py
        result = predict_emotions_from_vad(vad_scores['valence'], vad_scores['arousal'], vad_scores['dominance'])
        
        # Extract emotions from the result
        if "top emotions are:" in result:
            emotions_text = result.split("top emotions are:")[-1].strip()
            emotions = [e.strip() for e in emotions_text.split(',')]
            return emotions
        else:
            # Default emotions if prediction fails
            return ["Neutral"]
    
    def get_music_recommendations(self, emotions, top_k=3):
        """
        Generate specific music composition suggestions based on detected emotions
        
        Parameters:
        - emotions: List of emotions
        - top_k: Number of recommendations to return (not used in new implementation)
        
        Returns:
        - String with detailed music composition suggestions
        """
        # Extract relevant composition strategies for the detected emotions
        strategies = []
        for emotion in emotions:
            if emotion in self.composition_strategies:
                strategies.append(self.composition_strategies[emotion])
        
        # If no strategies found, use a default neutral strategy
        if not strategies:
            strategies.append(self.composition_strategies.get("Neutral", 
                             "Compose balanced, ambient music with moderate tempo and neutral emotional characteristics."))
        
        # Get musical characteristics for context and detail
        contexts = []
        for emotion in emotions:
            if emotion in self.emotion_context:
                contexts.extend(self.emotion_context[emotion])
        
        # Remove duplicates and get top characteristics
        contexts = list(set(contexts))[:10]  
        
        # Create a comprehensive composition recommendation
        if len(emotions) > 1:
            # For multiple emotions (complex emotional state)
            composition_suggestion = f"""\nCompose music that reflects the complex emotional blend of {', '.join(emotions)} through these specific approaches:

1. Structure & Harmony: 
   {strategies[0] if strategies else "Use a balanced structure with moderate harmonic complexity."}
   
2. Rhythm & Tempo: 
   {strategies[1] if len(strategies) > 1 else "Incorporate rhythmic elements that transition between emotions."}
   
3. Timbre & Instrumentation:
   Include {', '.join(contexts[:5])} as key sonic elements.
   {strategies[2] if len(strategies) > 2 else "Select instrumentation that can express emotional complexity."}

4. Dynamic Progression:
   Begin with {emotions[0]}'s characteristics, then gradually introduce elements of {', '.join(emotions[1:])} to create emotional depth.
   Allow for natural transitions between emotional states rather than abrupt changes.

5. Production Techniques:
   Apply appropriate effects and processing to enhance the emotional qualities: {', '.join(contexts[5:10] if len(contexts) > 5 else contexts)}.
   Create a cohesive sound that maintains emotional authenticity throughout the piece."""
        else:
            # For single emotion (focused emotional state)
            composition_suggestion = f"""\nCompose music that authentically captures {emotions[0]} through these specific approaches:

1. Structure & Harmony: 
   {strategies[0] if strategies else "Use a structure and harmonic approach appropriate for the emotion."}
   
2. Rhythm & Tempo: 
   Develop rhythmic patterns that reinforce the emotional quality of {emotions[0]}.
   
3. Timbre & Instrumentation:
   Feature sounds that embody {', '.join(contexts[:5])}.
   Select instruments and timbres that naturally express this emotional state.

4. Dynamic Progression:
   Create a coherent emotional journey that explores different intensities of {emotions[0]}.
   Allow for natural development while maintaining the core emotional essence.

5. Production Techniques:
   Apply production elements that enhance the {emotions[0]} quality: {', '.join(contexts[5:10] if len(contexts) > 5 else contexts)}.
   Ensure that every sonic element contributes to the intended emotional impact."""
            
        return composition_suggestion
    
    def format_recommendations(self, composition_suggestion):
        """
        Format the composition suggestion for display
        
        Parameters:
        - composition_suggestion: String with composition suggestions
        
        Returns:
        - Formatted string with the composition suggestion
        """
        return composition_suggestion
    
    def save_prompt_to_file(self, emotions, composition_suggestion, vad_scores):
        """Save only the music composition suggestions to a prompt.txt file"""
        try:
            prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
            with open(prompt_file, 'w') as f:
                # Write only the music composition suggestions without VAD scores or predicted emotions
                f.write(composition_suggestion)
                
            print(f"\nPrompt saved to {prompt_file}")
            return True
        except Exception as e:
            print(f"Error saving prompt to file: {e}")
            return False
    
    def process_audio(self, audio_path):
        """
        Process an audio file and return music recommendations
        
        Parameters:
        - audio_path: Path to the audio file
        
        Returns:
        - Dictionary with VAD scores, emotions, and music composition suggestions
        """
        # Get VAD scores
        vad_scores = self.predict_vad_from_audio(audio_path)
        
        # Get emotions from VAD scores
        emotions = self.get_emotions_from_vad(vad_scores)
        
        # Store the emotions for use in recommendations
        self.last_emotions = emotions
        
        # Get music composition suggestions
        composition_suggestion = self.get_music_recommendations(emotions)
        
        # Format composition suggestions
        formatted_suggestion = self.format_recommendations(composition_suggestion)
        
        # Save the prompt to a file
        self.save_prompt_to_file(emotions, formatted_suggestion, vad_scores)
        
        return {
            "vad_scores": vad_scores,
            "emotions": emotions,
            "recommendations": formatted_suggestion
        }


if __name__ == "__main__":
    # Initialize the recommender
    recommender = MusicEmotionRecommender()
    
    # Get audio file path from user
    audio_path = input("Enter the path to your audio file: ")
    
    # Process the audio and get recommendations
    results = recommender.process_audio(audio_path)
    
    # Display only the music composition suggestions to match the prompt.txt output
    print(results['recommendations'])