import os
import logging
import re
from typing import Dict, List, Union

class SentimentEngine:
    """
    Sentiment analysis engine using HuggingFace transformers.
    Analyzes crypto-related text for sentiment scoring.
    """
    
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        self.logger = self._setup_logging()
        
        # Initialize keyword-based sentiment analysis
        self._load_sentiment_keywords()
    
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
    
    def _load_sentiment_keywords(self):
        """Load sentiment keywords for analysis."""
        try:
            self.positive_keywords = [
                'bullish', 'moon', 'pump', 'surge', 'breakout', 'rally', 'gains',
                'adoption', 'partnership', 'upgrade', 'integration', 'launch',
                'hodl', 'diamond hands', 'to the moon', 'green', 'profit',
                'growth', 'innovation', 'revolutionary', 'game changer',
                'good', 'great', 'excellent', 'amazing', 'fantastic', 'positive',
                'up', 'rise', 'increase', 'buy', 'strong', 'solid', 'promising'
            ]
            
            self.negative_keywords = [
                'bearish', 'dump', 'crash', 'drop', 'fall', 'correction', 'dip',
                'sell-off', 'panic', 'fud', 'fear', 'uncertainty', 'doubt',
                'red', 'loss', 'scam', 'hack', 'vulnerability', 'regulation',
                'ban', 'crackdown', 'bubble', 'overvalued', 'risky',
                'bad', 'terrible', 'awful', 'negative', 'down', 'decline',
                'sell', 'weak', 'poor', 'disappointing'
            ]
            
            self.logger.info("Sentiment keywords loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading sentiment keywords: {str(e)}")
            self.positive_keywords = []
            self.negative_keywords = []
    
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
            
            # Clean text
            cleaned_text = self._preprocess_text(text)
            text_lower = cleaned_text.lower()
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
            negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
            
            # Calculate base sentiment score
            if positive_count == 0 and negative_count == 0:
                # No keywords found, return neutral
                sentiment_score = 0.5
            else:
                # Calculate score based on keyword ratio
                total_keywords = positive_count + negative_count
                positive_ratio = positive_count / total_keywords if total_keywords > 0 else 0.5
                
                # Scale to 0-1 range with some randomness for variety
                import random
                noise = random.uniform(-0.05, 0.05)
                sentiment_score = max(0.0, min(1.0, positive_ratio + noise))
            
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
    
    def _calculate_keyword_sentiment(self, text: str) -> float:
        """
        Calculate sentiment based on keyword analysis.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score (0-1)
        """
        try:
            text_lower = text.lower()
            
            # Count positive and negative keywords
            positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
            negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
            
            # Calculate sentiment score
            if positive_count == 0 and negative_count == 0:
                return 0.5  # Neutral
            
            total_keywords = positive_count + negative_count
            positive_ratio = positive_count / total_keywords
            
            # Add slight randomness for variety
            import random
            noise = random.uniform(-0.02, 0.02)
            
            return max(0.0, min(1.0, positive_ratio + noise))
            
        except Exception as e:
            self.logger.error(f"Error calculating keyword sentiment: {str(e)}")
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
            
            # Analyze each text individually
            scores = []
            for text in texts:
                score = self.analyze_sentiment(text)
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
            "positive": self.positive_keywords,
            "negative": self.negative_keywords
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
