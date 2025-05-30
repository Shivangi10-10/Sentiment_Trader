import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, List, Union

class SentimentEngine:
    """
    Sentiment analysis engine using HuggingFace transformers.
    Analyzes crypto-related text for sentiment scoring.
    """
    
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.logger = self._setup_logging()
        
        # Initialize the model
        self._load_model()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the sentiment engine."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """Load the sentiment analysis model and tokenizer."""
        try:
            self.logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            
            self.logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading sentiment model: {str(e)}")
            # Fallback to a simpler model
            try:
                self.logger.info("Falling back to distilbert model")
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                self.logger.info("Fallback model loaded successfully")
            except Exception as fallback_error:
                self.logger.error(f"Error loading fallback model: {str(fallback_error)}")
                raise
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of given text and return a score between 0 and 1.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score (0 = very negative, 1 = very positive)
        """
        try:
            if not text or len(text.strip()) == 0:
                return 0.5  # Neutral for empty text
            
            # Clean and truncate text if too long
            cleaned_text = self._preprocess_text(text)
            
            # Get sentiment scores
            results = self.pipeline(cleaned_text)
            
            # Extract sentiment score
            sentiment_score = self._extract_sentiment_score(results[0])
            
            self.logger.debug(f"Text: '{cleaned_text[:50]}...' -> Sentiment: {sentiment_score:.3f}")
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.5  # Return neutral sentiment on error
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace and newlines
        cleaned = ' '.join(text.split())
        
        # Truncate if too long (BERT models have token limits)
        max_length = 500
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
    def _extract_sentiment_score(self, results: List[Dict]) -> float:
        """
        Extract a normalized sentiment score from pipeline results.
        
        Args:
            results (List[Dict]): Raw results from sentiment pipeline
            
        Returns:
            float: Normalized sentiment score (0-1)
        """
        try:
            # Handle different model output formats
            if isinstance(results, list) and len(results) > 0:
                # For models that return multiple scores
                positive_score = 0.0
                negative_score = 0.0
                
                for result in results:
                    label = result['label'].upper()
                    score = result['score']
                    
                    if 'POSITIVE' in label or label == 'LABEL_1' or '5' in label or '4' in label:
                        positive_score += score
                    elif 'NEGATIVE' in label or label == 'LABEL_0' or '1' in label or '2' in label:
                        negative_score += score
                    # Neutral scores (3 stars) contribute to middle ground
                
                # Normalize to 0-1 scale
                if positive_score + negative_score > 0:
                    sentiment_score = positive_score / (positive_score + negative_score)
                else:
                    sentiment_score = 0.5
                    
            else:
                # Fallback to neutral
                sentiment_score = 0.5
            
            # Ensure score is within bounds
            return max(0.0, min(1.0, sentiment_score))
            
        except Exception as e:
            self.logger.error(f"Error extracting sentiment score: {str(e)}")
            return 0.5
    
    def analyze_batch(self, texts: List[str]) -> List[float]:
        """
        Analyze sentiment for multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[float]: List of sentiment scores
        """
        try:
            if not texts:
                return []
            
            # Preprocess all texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Analyze in batch
            results = self.pipeline(cleaned_texts)
            
            # Extract scores
            scores = []
            for result in results:
                score = self._extract_sentiment_score(result)
                scores.append(score)
            
            self.logger.info(f"Analyzed {len(texts)} texts in batch")
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error in batch sentiment analysis: {str(e)}")
            # Return neutral scores for all texts on error
            return [0.5] * len(texts)
    
    def get_crypto_sentiment_keywords(self) -> Dict[str, List[str]]:
        """
        Get crypto-specific keywords that might influence sentiment.
        
        Returns:
            Dict[str, List[str]]: Dictionary of positive and negative keywords
        """
        return {
            "positive": [
                "bullish", "moon", "pump", "surge", "breakout", "rally", "gains",
                "adoption", "partnership", "upgrade", "integration", "launch",
                "hodl", "diamond hands", "to the moon", "green", "profit",
                "growth", "innovation", "revolutionary", "game changer"
            ],
            "negative": [
                "bearish", "dump", "crash", "drop", "fall", "correction", "dip",
                "sell-off", "panic", "fud", "fear", "uncertainty", "doubt",
                "red", "loss", "scam", "hack", "vulnerability", "regulation",
                "ban", "crackdown", "bubble", "overvalued", "risky"
            ]
        }
    
    def enhance_crypto_sentiment(self, text: str, base_score: float) -> float:
        """
        Enhance sentiment score based on crypto-specific keywords.
        
        Args:
            text (str): Original text
            base_score (float): Base sentiment score
            
        Returns:
            float: Enhanced sentiment score
        """
        try:
            keywords = self.get_crypto_sentiment_keywords()
            text_lower = text.lower()
            
            positive_count = sum(1 for word in keywords["positive"] if word in text_lower)
            negative_count = sum(1 for word in keywords["negative"] if word in text_lower)
            
            # Apply keyword-based adjustment
            keyword_adjustment = (positive_count - negative_count) * 0.05
            enhanced_score = base_score + keyword_adjustment
            
            # Ensure bounds
            enhanced_score = max(0.0, min(1.0, enhanced_score))
            
            return enhanced_score
            
        except Exception as e:
            self.logger.error(f"Error enhancing crypto sentiment: {str(e)}")
            return base_score
    
    def analyze_crypto_sentiment(self, text: str) -> float:
        """
        Analyze sentiment with crypto-specific enhancements.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Enhanced sentiment score
        """
        # Get base sentiment
        base_score = self.analyze_sentiment(text)
        
        # Enhance with crypto keywords
        enhanced_score = self.enhance_crypto_sentiment(text, base_score)
        
        return enhanced_score
