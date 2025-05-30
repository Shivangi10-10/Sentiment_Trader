import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Union

class Utils:
    """
    Utility functions for the Sentiment Sage trading agent.
    Provides common functionality used across multiple modules.
    """
    
    def __init__(self):
        """Initialize utilities with logging."""
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for utilities."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def format_timestamp(self, timestamp: Union[datetime, str], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format timestamp to string.
        
        Args:
            timestamp: Datetime object or string
            format_str: Format string for output
            
        Returns:
            str: Formatted timestamp string
        """
        try:
            if isinstance(timestamp, str):
                # Try to parse string timestamp
                timestamp = pd.to_datetime(timestamp)
            
            return timestamp.strftime(format_str)
            
        except Exception as e:
            self.logger.error(f"Error formatting timestamp: {str(e)}")
            return "Invalid timestamp"
    
    def calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """
        Calculate percentage change between two values.
        
        Args:
            old_value: Original value
            new_value: New value
            
        Returns:
            float: Percentage change
        """
        try:
            if old_value == 0:
                return 0.0 if new_value == 0 else float('inf')
            
            return ((new_value - old_value) / abs(old_value)) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating percentage change: {str(e)}")
            return 0.0
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize a score to be within specified bounds.
        
        Args:
            score: Score to normalize
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            float: Normalized score
        """
        try:
            return max(min_val, min(max_val, score))
        except:
            return min_val
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        try:
            if not text:
                return ""
            
            # Remove extra whitespace
            cleaned = ' '.join(text.split())
            
            # Remove or replace special characters that might cause issues
            cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            
            # Limit length
            max_length = 1000
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length] + "..."
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            return ""
    
    def generate_hash(self, data: str) -> str:
        """
        Generate SHA-256 hash of string data.
        
        Args:
            data: String to hash
            
        Returns:
            str: Hex hash string
        """
        try:
            return hashlib.sha256(data.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating hash: {str(e)}")
            return ""
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers with fallback.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division fails
            
        Returns:
            float: Division result or default
        """
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except:
            return default
    
    def format_currency(self, amount: float, currency: str = "USDT", decimals: int = 2) -> str:
        """
        Format currency amount for display.
        
        Args:
            amount: Amount to format
            currency: Currency symbol
            decimals: Number of decimal places
            
        Returns:
            str: Formatted currency string
        """
        try:
            if currency == "APT":
                decimals = 4  # More precision for APT
            
            return f"{amount:,.{decimals}f} {currency}"
            
        except Exception as e:
            self.logger.error(f"Error formatting currency: {str(e)}")
            return f"0.00 {currency}"
    
    def calculate_moving_average(self, data: List[float], window: int = 5) -> List[float]:
        """
        Calculate moving average of a data series.
        
        Args:
            data: List of numerical values
            window: Moving average window size
            
        Returns:
            List[float]: Moving averages
        """
        try:
            if len(data) < window:
                return data
            
            moving_averages = []
            for i in range(len(data)):
                if i < window - 1:
                    # For initial values, use available data
                    avg = sum(data[:i+1]) / (i+1)
                else:
                    # Calculate moving average
                    avg = sum(data[i-window+1:i+1]) / window
                moving_averages.append(avg)
            
            return moving_averages
            
        except Exception as e:
            self.logger.error(f"Error calculating moving average: {str(e)}")
            return data
    
    def get_time_bucket(self, timestamp: datetime, bucket_size: str = "1H") -> datetime:
        """
        Round timestamp to time bucket for aggregation.
        
        Args:
            timestamp: Timestamp to bucket
            bucket_size: Bucket size (1H, 30T, 15T, etc.)
            
        Returns:
            datetime: Bucketed timestamp
        """
        try:
            if bucket_size == "1H":
                return timestamp.replace(minute=0, second=0, microsecond=0)
            elif bucket_size == "30T":
                minute = (timestamp.minute // 30) * 30
                return timestamp.replace(minute=minute, second=0, microsecond=0)
            elif bucket_size == "15T":
                minute = (timestamp.minute // 15) * 15
                return timestamp.replace(minute=minute, second=0, microsecond=0)
            elif bucket_size == "5T":
                minute = (timestamp.minute // 5) * 5
                return timestamp.replace(minute=minute, second=0, microsecond=0)
            else:
                return timestamp
                
        except Exception as e:
            self.logger.error(f"Error bucketing timestamp: {str(e)}")
            return timestamp
    
    def validate_sentiment_score(self, score: float) -> bool:
        """
        Validate that sentiment score is within expected range.
        
        Args:
            score: Sentiment score to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
        except:
            return False
    
    def aggregate_sentiment_by_time(self, sentiment_data: pd.DataFrame, 
                                   bucket_size: str = "1H") -> pd.DataFrame:
        """
        Aggregate sentiment data by time buckets.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            bucket_size: Time bucket size
            
        Returns:
            pd.DataFrame: Aggregated sentiment data
        """
        try:
            if sentiment_data.empty:
                return sentiment_data
            
            # Create time buckets
            sentiment_data = sentiment_data.copy()
            sentiment_data['time_bucket'] = sentiment_data['timestamp'].apply(
                lambda x: self.get_time_bucket(x, bucket_size)
            )
            
            # Aggregate by time bucket and token
            aggregated = sentiment_data.groupby(['time_bucket', 'token']).agg({
                'sentiment_score': ['mean', 'count', 'std'],
                'source': lambda x: list(x.unique())
            }).reset_index()
            
            # Flatten column names
            aggregated.columns = [
                'timestamp', 'token', 'avg_sentiment', 'data_points', 
                'sentiment_std', 'sources'
            ]
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating sentiment data: {str(e)}")
            return sentiment_data
    
    def calculate_sentiment_trend(self, sentiment_data: pd.DataFrame, 
                                 window: int = 5) -> Dict[str, float]:
        """
        Calculate sentiment trend (slope) for each token.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            window: Number of data points for trend calculation
            
        Returns:
            Dict[str, float]: Trend slopes by token
        """
        try:
            trends = {}
            
            for token in sentiment_data['token'].unique():
                token_data = sentiment_data[sentiment_data['token'] == token].copy()
                token_data = token_data.sort_values('timestamp')
                
                if len(token_data) >= window:
                    recent_scores = token_data['sentiment_score'].tail(window).values
                    x = np.arange(len(recent_scores))
                    
                    # Calculate linear regression slope
                    slope = np.polyfit(x, recent_scores, 1)[0]
                    trends[token] = slope
                else:
                    trends[token] = 0.0
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment trends: {str(e)}")
            return {}
    
    def format_large_number(self, number: float) -> str:
        """
        Format large numbers with appropriate suffixes (K, M, B).
        
        Args:
            number: Number to format
            
        Returns:
            str: Formatted number string
        """
        try:
            if abs(number) >= 1e9:
                return f"{number/1e9:.2f}B"
            elif abs(number) >= 1e6:
                return f"{number/1e6:.2f}M"
            elif abs(number) >= 1e3:
                return f"{number/1e3:.2f}K"
            else:
                return f"{number:.2f}"
                
        except Exception as e:
            self.logger.error(f"Error formatting large number: {str(e)}")
            return "0"
    
    def get_sentiment_label(self, score: float) -> str:
        """
        Convert sentiment score to human-readable label.
        
        Args:
            score: Sentiment score (0-1)
            
        Returns:
            str: Sentiment label
        """
        try:
            if score >= 0.7:
                return "Very Positive"
            elif score >= 0.6:
                return "Positive"
            elif score >= 0.4:
                return "Neutral"
            elif score >= 0.3:
                return "Negative"
            else:
                return "Very Negative"
                
        except:
            return "Unknown"
    
    def save_data_to_json(self, data: Any, filename: str) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to JSON: {str(e)}")
            return False
    
    def load_data_from_json(self, filename: str) -> Optional[Any]:
        """
        Load data from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Any: Loaded data or None if failed
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading data from JSON: {str(e)}")
            return None
