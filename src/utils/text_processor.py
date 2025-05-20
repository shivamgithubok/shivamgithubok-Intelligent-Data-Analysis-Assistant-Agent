from typing import List, Dict, Any
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

class TextProcessor:
    """Utility class for processing and analyzing text data."""
    
    def __init__(self):
        """Initialize the text processor with required NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return tokens
    
    def get_word_frequencies(self, text: str) -> Dict[str, int]:
        """
        Get word frequencies from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, int]: Dictionary of word frequencies
        """
        tokens = self.tokenize_text(text)
        return dict(Counter(tokens))
    
    def analyze_text_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analyze a text column in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Name of the text column
            
        Returns:
            Dict[str, Any]: Dictionary containing text analysis results
        """
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        
        # Combine all text
        all_text = ' '.join(df[column].astype(str))
        
        # Get word frequencies
        word_freq = self.get_word_frequencies(all_text)
        
        # Calculate basic statistics
        total_words = sum(word_freq.values())
        unique_words = len(word_freq)
        
        # Get most common words
        most_common = dict(sorted(word_freq.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:10])
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'most_common_words': most_common,
            'average_word_length': sum(len(word) * freq for word, freq in word_freq.items()) / total_words
        }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text (str): Input text
            top_n (int): Number of keywords to extract
            
        Returns:
            List[str]: List of keywords
        """
        word_freq = self.get_word_frequencies(text)
        return [word for word, _ in sorted(word_freq.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:top_n]]
    
    def get_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Jaccard similarity.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Tokenize texts
        tokens1 = set(self.tokenize_text(text1))
        tokens2 = set(self.tokenize_text(text2))
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def generate_text_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the text.
        
        Args:
            text (str): Input text
            max_length (int): Maximum length of the summary
            
        Returns:
            str: Text summary
        """
        # Tokenize and get word frequencies
        tokens = self.tokenize_text(text)
        word_freq = Counter(tokens)
        
        # Score sentences based on word frequencies
        sentences = nltk.sent_tokenize(text)
        sentence_scores = {}
        
        for sentence in sentences:
            sentence_tokens = self.tokenize_text(sentence)
            score = sum(word_freq[token] for token in sentence_tokens)
            sentence_scores[sentence] = score
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        
        # Generate summary
        summary = []
        current_length = 0
        
        for sentence, _ in top_sentences:
            if current_length + len(sentence) <= max_length:
                summary.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        return ' '.join(summary) 