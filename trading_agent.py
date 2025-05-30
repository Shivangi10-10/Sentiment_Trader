import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import random
from utils import Utils

class TradingAgent:
    """
    AI-powered trading agent that makes decisions based on sentiment analysis.
    Implements automated trading strategies for Aptos ecosystem tokens.
    """
    
    def __init__(self):
        """Initialize the trading agent with default parameters."""
        self.logger = self._setup_logging()
        self.utils = Utils()
        
        # Trading parameters
        self.min_trade_amount = 10.0  # Minimum trade amount in USDT
        self.max_trade_percentage = 0.3  # Maximum 30% of balance per trade
        self.sentiment_window = 20  # Number of data points for rolling average
        self.confidence_threshold = 0.1  # Minimum confidence for trading
        
        # Risk management
        self.max_daily_trades = 10
        self.stop_loss_percentage = 0.05  # 5% stop loss
        self.take_profit_percentage = 0.10  # 10% take profit
        
        # Track daily trades
        self.daily_trades = {}
        
        # Mock price data for simulation
        self.current_prices = {"APT": 8.50, "USDT": 1.00}
        self.price_history = []
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the trading agent."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_rolling_sentiment(self, sentiment_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate rolling average sentiment for each token.
        
        Args:
            sentiment_data (pd.DataFrame): Historical sentiment data
            
        Returns:
            Dict[str, float]: Rolling sentiment averages by token
        """
        try:
            if sentiment_data.empty:
                return {"APT": 0.5, "USDT": 0.5}
            
            rolling_sentiments = {}
            
            # Calculate for each token
            for token in sentiment_data['token'].unique():
                token_data = sentiment_data[sentiment_data['token'] == token].copy()
                
                if len(token_data) > 0:
                    # Sort by timestamp
                    token_data = token_data.sort_values('timestamp')
                    
                    # Calculate rolling average
                    window_size = min(self.sentiment_window, len(token_data))
                    recent_scores = token_data['sentiment_score'].tail(window_size)
                    rolling_avg = recent_scores.mean()
                    
                    rolling_sentiments[token] = rolling_avg
                else:
                    rolling_sentiments[token] = 0.5
            
            # Ensure we have values for main tokens
            if 'APT' not in rolling_sentiments:
                rolling_sentiments['APT'] = 0.5
            if 'USDT' not in rolling_sentiments:
                rolling_sentiments['USDT'] = 0.5
            
            self.logger.debug(f"Rolling sentiments: {rolling_sentiments}")
            return rolling_sentiments
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling sentiment: {str(e)}")
            return {"APT": 0.5, "USDT": 0.5}
    
    def make_trading_decision(self, sentiment_scores: Dict[str, float], 
                            buy_threshold: float, sell_threshold: float) -> Dict[str, Any]:
        """
        Make trading decision based on sentiment scores and thresholds.
        
        Args:
            sentiment_scores (Dict[str, float]): Current sentiment scores
            buy_threshold (float): Threshold for buy signals
            sell_threshold (float): Threshold for sell signals
            
        Returns:
            Dict[str, Any]: Trading decision with action, token, and reasoning
        """
        try:
            apt_sentiment = sentiment_scores.get('APT', 0.5)
            
            # Check if we've exceeded daily trade limit
            today = datetime.now().date()
            if today not in self.daily_trades:
                self.daily_trades[today] = 0
            
            if self.daily_trades[today] >= self.max_daily_trades:
                return {
                    'action': 'HOLD',
                    'token': 'APT',
                    'confidence': 0.0,
                    'reason': 'Daily trade limit reached'
                }
            
            # Determine action based on sentiment
            if apt_sentiment >= buy_threshold:
                # Strong positive sentiment - BUY signal
                confidence = min((apt_sentiment - buy_threshold) / (1.0 - buy_threshold), 1.0)
                
                if confidence >= self.confidence_threshold:
                    return {
                        'action': 'BUY',
                        'token': 'APT',
                        'confidence': confidence,
                        'reason': f'Strong positive sentiment: {apt_sentiment:.3f} >= {buy_threshold}'
                    }
            
            elif apt_sentiment <= sell_threshold:
                # Strong negative sentiment - SELL signal
                confidence = min((sell_threshold - apt_sentiment) / sell_threshold, 1.0)
                
                if confidence >= self.confidence_threshold:
                    return {
                        'action': 'SELL',
                        'token': 'APT',
                        'confidence': confidence,
                        'reason': f'Strong negative sentiment: {apt_sentiment:.3f} <= {sell_threshold}'
                    }
            
            # No strong signal - HOLD
            return {
                'action': 'HOLD',
                'token': 'APT',
                'confidence': 0.0,
                'reason': f'Neutral sentiment: {apt_sentiment:.3f} between thresholds'
            }
            
        except Exception as e:
            self.logger.error(f"Error making trading decision: {str(e)}")
            return {
                'action': 'HOLD',
                'token': 'APT',
                'confidence': 0.0,
                'reason': f'Error in decision making: {str(e)}'
            }
    
    def calculate_trade_amount(self, action: str, confidence: float, 
                             balance: Dict[str, float]) -> float:
        """
        Calculate the appropriate trade amount based on confidence and balance.
        
        Args:
            action (str): Trading action (BUY/SELL)
            confidence (float): Confidence level (0-1)
            balance (Dict[str, float]): Current portfolio balance
            
        Returns:
            float: Trade amount
        """
        try:
            if action == 'BUY':
                # Calculate based on USDT balance
                available_usdt = balance.get('USDT', 0)
                max_trade_usdt = available_usdt * self.max_trade_percentage
                
                # Adjust based on confidence
                trade_usdt = max_trade_usdt * confidence
                
                # Ensure minimum trade amount
                if trade_usdt < self.min_trade_amount:
                    if available_usdt >= self.min_trade_amount:
                        trade_usdt = self.min_trade_amount
                    else:
                        return 0.0
                
                # Convert to APT amount
                apt_price = self.current_prices['APT']
                trade_amount = trade_usdt / apt_price
                
            elif action == 'SELL':
                # Calculate based on APT balance
                available_apt = balance.get('APT', 0)
                max_trade_apt = available_apt * self.max_trade_percentage
                
                # Adjust based on confidence
                trade_amount = max_trade_apt * confidence
                
                # Ensure minimum trade value in USDT
                trade_value_usdt = trade_amount * self.current_prices['APT']
                if trade_value_usdt < self.min_trade_amount:
                    min_apt_amount = self.min_trade_amount / self.current_prices['APT']
                    if available_apt >= min_apt_amount:
                        trade_amount = min_apt_amount
                    else:
                        return 0.0
            else:
                return 0.0
            
            return max(0.0, trade_amount)
            
        except Exception as e:
            self.logger.error(f"Error calculating trade amount: {str(e)}")
            return 0.0
    
    def execute_trade(self, trade_decision: Dict[str, Any], 
                     current_balance: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute the trading decision (simulated for demo).
        
        Args:
            trade_decision (Dict[str, Any]): Trading decision from make_trading_decision
            current_balance (Dict[str, float]): Current portfolio balance
            
        Returns:
            Dict[str, Any]: Trade execution result
        """
        try:
            action = trade_decision['action']
            token = trade_decision['token']
            confidence = trade_decision['confidence']
            
            if action == 'HOLD':
                return {
                    'success': False,
                    'reason': 'No trade executed - HOLD decision',
                    'amount': 0.0,
                    'price': self.current_prices[token],
                    'new_balance': current_balance.copy()
                }
            
            # Calculate trade amount
            trade_amount = self.calculate_trade_amount(action, confidence, current_balance)
            
            if trade_amount <= 0:
                return {
                    'success': False,
                    'reason': 'Insufficient balance or amount too small',
                    'amount': 0.0,
                    'price': self.current_prices[token],
                    'new_balance': current_balance.copy()
                }
            
            # Simulate price slippage (small random variation)
            base_price = self.current_prices[token]
            slippage = random.uniform(-0.002, 0.002)  # 0.2% max slippage
            execution_price = base_price * (1 + slippage)
            
            # Update balance
            new_balance = current_balance.copy()
            
            if action == 'BUY':
                # Buy APT with USDT
                required_usdt = trade_amount * execution_price
                
                if new_balance.get('USDT', 0) >= required_usdt:
                    new_balance['USDT'] -= required_usdt
                    new_balance['APT'] = new_balance.get('APT', 0) + trade_amount
                    
                    # Update daily trade count
                    today = datetime.now().date()
                    self.daily_trades[today] = self.daily_trades.get(today, 0) + 1
                    
                    self.logger.info(f"BUY executed: {trade_amount:.4f} APT at {execution_price:.2f}")
                    
                    return {
                        'success': True,
                        'reason': f'BUY order executed successfully',
                        'amount': trade_amount,
                        'price': execution_price,
                        'new_balance': new_balance
                    }
                else:
                    return {
                        'success': False,
                        'reason': 'Insufficient USDT balance',
                        'amount': 0.0,
                        'price': execution_price,
                        'new_balance': current_balance.copy()
                    }
            
            elif action == 'SELL':
                # Sell APT for USDT
                if new_balance.get('APT', 0) >= trade_amount:
                    received_usdt = trade_amount * execution_price
                    new_balance['APT'] -= trade_amount
                    new_balance['USDT'] = new_balance.get('USDT', 0) + received_usdt
                    
                    # Update daily trade count
                    today = datetime.now().date()
                    self.daily_trades[today] = self.daily_trades.get(today, 0) + 1
                    
                    self.logger.info(f"SELL executed: {trade_amount:.4f} APT at {execution_price:.2f}")
                    
                    return {
                        'success': True,
                        'reason': f'SELL order executed successfully',
                        'amount': trade_amount,
                        'price': execution_price,
                        'new_balance': new_balance
                    }
                else:
                    return {
                        'success': False,
                        'reason': 'Insufficient APT balance',
                        'amount': 0.0,
                        'price': execution_price,
                        'new_balance': current_balance.copy()
                    }
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return {
                'success': False,
                'reason': f'Trade execution failed: {str(e)}',
                'amount': 0.0,
                'price': self.current_prices.get(token, 0),
                'new_balance': current_balance.copy()
            }
    
    def update_prices(self, new_prices: Dict[str, float]):
        """
        Update current market prices.
        
        Args:
            new_prices (Dict[str, float]): New price data
        """
        try:
            self.current_prices.update(new_prices)
            
            # Add to price history
            self.price_history.append({
                'timestamp': datetime.now(),
                'prices': self.current_prices.copy()
            })
            
            # Keep only last 1000 price points
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]
            
            self.logger.debug(f"Prices updated: {self.current_prices}")
            
        except Exception as e:
            self.logger.error(f"Error updating prices: {str(e)}")
    
    def get_trading_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the current trading strategy.
        
        Returns:
            Dict[str, Any]: Strategy information
        """
        return {
            'sentiment_window': self.sentiment_window,
            'min_trade_amount': self.min_trade_amount,
            'max_trade_percentage': self.max_trade_percentage,
            'confidence_threshold': self.confidence_threshold,
            'max_daily_trades': self.max_daily_trades,
            'stop_loss_percentage': self.stop_loss_percentage,
            'take_profit_percentage': self.take_profit_percentage,
            'current_prices': self.current_prices
        }
    
    def get_performance_metrics(self, trading_history: pd.DataFrame, 
                               initial_balance: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate performance metrics for the trading strategy.
        
        Args:
            trading_history (pd.DataFrame): Historical trades
            initial_balance (Dict[str, float]): Starting portfolio balance
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            if trading_history.empty:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'total_return': 0.0
                }
            
            total_trades = len(trading_history)
            
            # Calculate P&L for each trade (simplified)
            winning_trades = 0
            losing_trades = 0
            total_pnl = 0.0
            
            # This is a simplified P&L calculation
            # In reality, you'd need more sophisticated accounting
            
            for _, trade in trading_history.iterrows():
                # Mock P&L calculation based on random market movements
                pnl = random.uniform(-50, 100)  # Simplified
                total_pnl += pnl
                
                if pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculate total return (mock)
            initial_value = initial_balance.get('USDT', 1000)
            total_return = (total_pnl / initial_value) * 100 if initial_value > 0 else 0.0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return': total_return
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0
            }
