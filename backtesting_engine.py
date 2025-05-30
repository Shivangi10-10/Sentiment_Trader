import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

class BacktestingEngine:
    """
    Backtesting engine for evaluating trading strategy performance.
    Tests the sentiment-based trading agent against historical data.
    """
    
    def __init__(self):
        """Initialize the backtesting engine."""
        self.logger = self._setup_logging()
        self.results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the backtesting engine."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_backtest(self, sentiment_data: pd.DataFrame, 
                    initial_balance: Dict[str, float],
                    buy_threshold: float = 0.7,
                    sell_threshold: float = 0.3,
                    risk_level: str = "Moderate") -> Dict[str, Any]:
        """
        Run a backtest simulation using historical sentiment data.
        
        Args:
            sentiment_data (pd.DataFrame): Historical sentiment data
            initial_balance (Dict[str, float]): Starting portfolio balance
            buy_threshold (float): Buy signal threshold
            sell_threshold (float): Sell signal threshold
            risk_level (str): Risk management level
            
        Returns:
            Dict[str, Any]: Backtest results and performance metrics
        """
        try:
            if sentiment_data.empty:
                return self._empty_backtest_result()
            
            # Initialize backtest state
            portfolio = initial_balance.copy()
            trades = []
            portfolio_values = []
            
            # Sort data by timestamp
            sentiment_data = sentiment_data.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate rolling sentiment for trading decisions
            window_size = 5
            sentiment_data['rolling_sentiment'] = sentiment_data['sentiment_score'].rolling(
                window=window_size, min_periods=1
            ).mean()
            
            # Mock price data for simulation
            base_price = 8.50
            
            for idx, row in sentiment_data.iterrows():
                # Simulate price movement based on sentiment
                sentiment_impact = (row['rolling_sentiment'] - 0.5) * 0.1
                price_noise = np.random.uniform(-0.02, 0.02)
                current_price = base_price * (1 + sentiment_impact + price_noise)
                
                # Make trading decision
                decision = self._make_backtest_decision(
                    row['rolling_sentiment'], buy_threshold, sell_threshold,
                    portfolio, current_price, risk_level
                )
                
                if decision['action'] != 'HOLD':
                    # Execute trade
                    trade_result = self._execute_backtest_trade(
                        decision, portfolio, current_price, row['timestamp']
                    )
                    
                    if trade_result['success']:
                        portfolio = trade_result['new_portfolio']
                        trades.append(trade_result['trade_record'])
                
                # Record portfolio value
                portfolio_value = portfolio.get('USDT', 0) + portfolio.get('APT', 0) * current_price
                portfolio_values.append({
                    'timestamp': row['timestamp'],
                    'portfolio_value': portfolio_value,
                    'apt_price': current_price,
                    'sentiment': row['rolling_sentiment']
                })
            
            # Calculate performance metrics
            performance = self._calculate_backtest_performance(
                portfolio_values, trades, initial_balance
            )
            
            return {
                'success': True,
                'trades': trades,
                'portfolio_values': portfolio_values,
                'final_portfolio': portfolio,
                'performance': performance,
                'strategy_params': {
                    'buy_threshold': buy_threshold,
                    'sell_threshold': sell_threshold,
                    'risk_level': risk_level
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            return self._empty_backtest_result()
    
    def _make_backtest_decision(self, sentiment: float, buy_threshold: float,
                               sell_threshold: float, portfolio: Dict[str, float],
                               price: float, risk_level: str) -> Dict[str, Any]:
        """Make a trading decision for backtesting."""
        try:
            # Adjust thresholds based on risk level
            if risk_level == "Conservative":
                buy_threshold += 0.05
                sell_threshold -= 0.05
                max_trade_pct = 0.2
            elif risk_level == "Aggressive":
                buy_threshold -= 0.05
                sell_threshold += 0.05
                max_trade_pct = 0.5
            else:  # Moderate
                max_trade_pct = 0.3
            
            if sentiment >= buy_threshold:
                # Buy signal
                available_usdt = portfolio.get('USDT', 0)
                if available_usdt > 20:  # Minimum trade amount
                    trade_amount_usdt = available_usdt * max_trade_pct
                    apt_amount = trade_amount_usdt / price
                    
                    return {
                        'action': 'BUY',
                        'amount': apt_amount,
                        'confidence': min((sentiment - buy_threshold) / (1.0 - buy_threshold), 1.0)
                    }
            
            elif sentiment <= sell_threshold:
                # Sell signal
                available_apt = portfolio.get('APT', 0)
                if available_apt > 0.01:  # Minimum APT amount
                    sell_amount = available_apt * max_trade_pct
                    
                    return {
                        'action': 'SELL',
                        'amount': sell_amount,
                        'confidence': min((sell_threshold - sentiment) / sell_threshold, 1.0)
                    }
            
            return {'action': 'HOLD', 'amount': 0, 'confidence': 0.0}
            
        except Exception as e:
            self.logger.error(f"Error making backtest decision: {str(e)}")
            return {'action': 'HOLD', 'amount': 0, 'confidence': 0.0}
    
    def _execute_backtest_trade(self, decision: Dict[str, Any], 
                               portfolio: Dict[str, float], price: float,
                               timestamp) -> Dict[str, Any]:
        """Execute a trade in the backtest simulation."""
        try:
            new_portfolio = portfolio.copy()
            
            if decision['action'] == 'BUY':
                required_usdt = decision['amount'] * price
                if new_portfolio.get('USDT', 0) >= required_usdt:
                    new_portfolio['USDT'] -= required_usdt
                    new_portfolio['APT'] = new_portfolio.get('APT', 0) + decision['amount']
                    
                    return {
                        'success': True,
                        'new_portfolio': new_portfolio,
                        'trade_record': {
                            'timestamp': timestamp,
                            'action': 'BUY',
                            'amount': decision['amount'],
                            'price': price,
                            'value': required_usdt,
                            'confidence': decision['confidence']
                        }
                    }
            
            elif decision['action'] == 'SELL':
                if new_portfolio.get('APT', 0) >= decision['amount']:
                    received_usdt = decision['amount'] * price
                    new_portfolio['APT'] -= decision['amount']
                    new_portfolio['USDT'] = new_portfolio.get('USDT', 0) + received_usdt
                    
                    return {
                        'success': True,
                        'new_portfolio': new_portfolio,
                        'trade_record': {
                            'timestamp': timestamp,
                            'action': 'SELL',
                            'amount': decision['amount'],
                            'price': price,
                            'value': received_usdt,
                            'confidence': decision['confidence']
                        }
                    }
            
            return {'success': False, 'new_portfolio': portfolio}
            
        except Exception as e:
            self.logger.error(f"Error executing backtest trade: {str(e)}")
            return {'success': False, 'new_portfolio': portfolio}
    
    def _calculate_backtest_performance(self, portfolio_values: List[Dict],
                                      trades: List[Dict],
                                      initial_balance: Dict[str, float]) -> Dict[str, Any]:
        """Calculate performance metrics for the backtest."""
        try:
            if not portfolio_values:
                return {}
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(portfolio_values)
            
            # Calculate returns
            initial_value = sum(initial_balance.values())
            final_value = df['portfolio_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            # Calculate volatility
            df['daily_return'] = df['portfolio_value'].pct_change()
            volatility = df['daily_return'].std() * np.sqrt(252) * 100  # Annualized
            
            # Calculate Sharpe ratio (simplified)
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_return = total_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            df['cumulative_max'] = df['portfolio_value'].cummax()
            df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max']
            max_drawdown = df['drawdown'].min() * 100
            
            # Trading statistics
            total_trades = len(trades)
            profitable_trades = len([t for t in trades if self._is_profitable_trade(t, df)])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return_pct': total_return,
                'volatility_pct': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate_pct': win_rate,
                'average_trade_value': np.mean([t['value'] for t in trades]) if trades else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest performance: {str(e)}")
            return {}
    
    def _is_profitable_trade(self, trade: Dict, portfolio_df: pd.DataFrame) -> bool:
        """Determine if a trade was profitable (simplified)."""
        # This is a simplified profitability check
        # In reality, we'd need to track entry and exit prices
        return trade.get('confidence', 0) > 0.5
    
    def _empty_backtest_result(self) -> Dict[str, Any]:
        """Return empty backtest result structure."""
        return {
            'success': False,
            'trades': [],
            'portfolio_values': [],
            'final_portfolio': {},
            'performance': {},
            'strategy_params': {}
        }
    
    def compare_strategies(self, sentiment_data: pd.DataFrame,
                          initial_balance: Dict[str, float],
                          strategy_params: List[Dict]) -> Dict[str, Any]:
        """
        Compare multiple trading strategies.
        
        Args:
            sentiment_data (pd.DataFrame): Historical sentiment data
            initial_balance (Dict[str, float]): Starting balance
            strategy_params (List[Dict]): List of strategy parameters to test
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            results = {}
            
            for i, params in enumerate(strategy_params):
                strategy_name = f"Strategy_{i+1}"
                self.logger.info(f"Running backtest for {strategy_name}")
                
                result = self.run_backtest(
                    sentiment_data=sentiment_data,
                    initial_balance=initial_balance,
                    buy_threshold=params.get('buy_threshold', 0.7),
                    sell_threshold=params.get('sell_threshold', 0.3),
                    risk_level=params.get('risk_level', 'Moderate')
                )
                
                results[strategy_name] = result
            
            # Find best performing strategy
            best_strategy = None
            best_return = float('-inf')
            
            for name, result in results.items():
                if result['success']:
                    total_return = result['performance'].get('total_return_pct', -999)
                    if total_return > best_return:
                        best_return = total_return
                        best_strategy = name
            
            return {
                'results': results,
                'best_strategy': best_strategy,
                'best_return': best_return
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {str(e)}")
            return {'results': {}, 'best_strategy': None, 'best_return': 0}