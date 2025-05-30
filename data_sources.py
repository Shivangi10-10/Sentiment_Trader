import requests
import feedparser
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import os
from utils import Utils

class DataSources:
    """
    Data sources manager for fetching crypto news and social media data.
    Handles multiple data sources and formats them consistently.
    """
    
    def __init__(self):
        """Initialize data sources with API configurations."""
        self.logger = self._setup_logging()
        self.utils = Utils()
        
        # API configurations
        self.news_apis = {
            'cryptopanic': {
                'url': 'https://cryptopanic.com/api/v1/posts/',
                'params': {
                    'auth_token': os.getenv('CRYPTOPANIC_API_KEY', ''),
                    'public': 'true',
                    'currencies': 'APT',
                    'filter': 'news'
                }
            },
            'rss_feeds': [
                'https://cointelegraph.com/rss',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://decrypt.co/feed',
                'https://cryptoslate.com/feed/'
            ]
        }
        
        # Mock data for demonstration when APIs are not available
        self.mock_news_enabled = True
        self.mock_social_enabled = True
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for data sources."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fetch_crypto_news(self) -> List[Dict[str, Any]]:
        """
        Fetch crypto news from multiple sources.
        
        Returns:
            List[Dict]: List of news items with standardized format
        """
        all_news = []
        
        # Try to fetch from real APIs first
        try:
            # Fetch from CryptoPanic API
            cryptopanic_news = self._fetch_cryptopanic_news()
            all_news.extend(cryptopanic_news)
            
            # Fetch from RSS feeds
            rss_news = self._fetch_rss_news()
            all_news.extend(rss_news)
            
        except Exception as e:
            self.logger.error(f"Error fetching real news data: {str(e)}")
        
        # If no real data or mock is enabled, add realistic demo data
        if len(all_news) == 0 or self.mock_news_enabled:
            mock_news = self._generate_realistic_news()
            all_news.extend(mock_news)
        
        # Sort by timestamp (newest first)
        all_news.sort(key=lambda x: x['timestamp'], reverse=True)
        
        self.logger.info(f"Fetched {len(all_news)} news items")
        return all_news[:50]  # Return top 50 items
    
    def _fetch_cryptopanic_news(self) -> List[Dict[str, Any]]:
        """Fetch news from CryptoPanic API."""
        try:
            api_key = os.getenv('CRYPTOPANIC_API_KEY', '')
            if not api_key:
                self.logger.warning("CryptoPanic API key not found")
                return []
            
            params = self.news_apis['cryptopanic']['params'].copy()
            params['auth_token'] = api_key
            
            response = requests.get(
                self.news_apis['cryptopanic']['url'],
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                news_items = []
                
                for item in data.get('results', []):
                    news_items.append({
                        'timestamp': datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')),
                        'text': item['title'],
                        'source': 'CryptoPanic',
                        'url': item.get('url', ''),
                        'token': 'APT'
                    })
                
                return news_items
            else:
                self.logger.warning(f"CryptoPanic API returned status {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching CryptoPanic news: {str(e)}")
            return []
    
    def _fetch_rss_news(self) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds."""
        all_rss_news = []
        
        for feed_url in self.news_apis['rss_feeds']:
            try:
                self.logger.info(f"Fetching RSS from {feed_url}")
                
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Limit to 5 per feed
                    # Check if entry is crypto-related
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    text = f"{title} {summary}"
                    
                    if self._is_crypto_related(text):
                        # Parse publication date
                        pub_date = entry.get('published_parsed')
                        if pub_date:
                            timestamp = datetime(*pub_date[:6])
                        else:
                            timestamp = datetime.now()
                        
                        all_rss_news.append({
                            'timestamp': timestamp,
                            'text': title,
                            'source': feed.feed.get('title', 'RSS Feed'),
                            'url': entry.get('link', ''),
                            'token': self._extract_token_mention(text)
                        })
                
            except Exception as e:
                self.logger.error(f"Error fetching RSS from {feed_url}: {str(e)}")
                continue
        
        return all_rss_news
    
    def _is_crypto_related(self, text: str) -> bool:
        """Check if text is crypto-related."""
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'aptos', 'apt', 'token', 'coin',
            'trading', 'price', 'market', 'bullish', 'bearish'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crypto_keywords)
    
    def _extract_token_mention(self, text: str) -> str:
        """Extract token mention from text."""
        text_lower = text.lower()
        
        if 'aptos' in text_lower or 'apt' in text_lower:
            return 'APT'
        elif 'usdt' in text_lower or 'tether' in text_lower:
            return 'USDT'
        else:
            return 'APT'  # Default to APT
    
    def fetch_social_media_data(self) -> List[Dict[str, Any]]:
        """
        Fetch social media data (simulated for demo).
        In production, this would connect to Twitter API, Reddit API, etc.
        
        Returns:
            List[Dict]: List of social media posts with standardized format
        """
        social_data = []
        
        # Generate realistic social media posts
        if self.mock_social_enabled:
            mock_social = self._generate_realistic_social_posts()
            social_data.extend(mock_social)
        
        self.logger.info(f"Fetched {len(social_data)} social media posts")
        return social_data
    
    def _generate_realistic_news(self) -> List[Dict[str, Any]]:
        """Generate realistic news data for demonstration."""
        realistic_headlines = [
            "Aptos Network Sees 300% Increase in Daily Active Users",
            "Major DeFi Protocol Launches on Aptos Blockchain",
            "Aptos Foundation Announces $100M Developer Fund",
            "Institutional Investors Show Growing Interest in APT Token",
            "Aptos Mainnet Processes Record 32,000 TPS Without Downtime",
            "New Partnership Between Aptos and Leading Crypto Exchange",
            "Aptos Labs Releases Major SDK Update for Developers",
            "Cross-Chain Bridge Connects Aptos to Ethereum Ecosystem",
            "Aptos-Based GameFi Project Raises $50M in Series A",
            "Move Programming Language Gains Adoption Beyond Aptos",
            "Regulatory Clarity Boosts Confidence in Aptos Ecosystem",
            "Aptos Network Upgrade Introduces Enhanced Security Features",
            "Major Payment Processor Integrates Aptos for Settlements",
            "Decentralized Exchange on Aptos Reaches $1B in TVL",
            "Aptos Foundation Partners with University for Blockchain Research"
        ]
        
        news_items = []
        now = datetime.now()
        
        # Generate 10-15 realistic news items
        num_items = random.randint(10, 15)
        selected_headlines = random.sample(realistic_headlines, min(num_items, len(realistic_headlines)))
        
        for i, headline in enumerate(selected_headlines):
            # Create timestamps spread over the last 24 hours
            hours_ago = random.randint(0, 24)
            minutes_ago = random.randint(0, 59)
            timestamp = now - timedelta(hours=hours_ago, minutes=minutes_ago)
            
            news_items.append({
                'timestamp': timestamp,
                'text': headline,
                'source': random.choice(['CoinDesk', 'Cointelegraph', 'The Block', 'Decrypt', 'CryptoSlate']),
                'url': f'https://example.com/news/{i}',
                'token': 'APT'
            })
        
        return news_items
    
    def _generate_realistic_social_posts(self) -> List[Dict[str, Any]]:
        """Generate realistic social media posts for demonstration."""
        realistic_posts = [
            "Just bridged my assets to @Aptos_Network - the speed is incredible! 🚀 #AptosEcosystem",
            "Building on Aptos has been a game-changer for our DeFi protocol. Move language is 🔥",
            "APT showing strong fundamentals despite market conditions. Long-term outlook remains bullish 📈",
            "The Aptos developer experience is top-notch. Coming from Solidity, Move feels refreshing",
            "Excited to see the growing ecosystem on Aptos. So many innovative projects launching!",
            "Gas fees on Aptos are practically non-existent. This is how crypto should work ⚡",
            "Just used an Aptos DEX and the transaction was instant. Web2 level user experience!",
            "Aptos Foundation's commitment to education and developer support is impressive 👏",
            "The parallel execution model on Aptos is revolutionary. True scalability at last!",
            "Started learning Move programming. The safety features are exactly what DeFi needs",
            "Aptos NFT marketplace just launched and it's smooth as butter 🎨",
            "Comparing Aptos to other L1s, the technical advantages are clear. Undervalued imo",
            "Just staked my APT tokens. The staking rewards are competitive 💰",
            "Aptos governance proposal voting is live. Love seeing community participation!",
            "The institutional adoption of Aptos is accelerating. Big things coming 🏛️"
        ]
        
        negative_posts = [
            "Still waiting for more major dApps on Aptos. Ecosystem needs more development",
            "APT price action has been disappointing lately. When will we see a breakout? 📉",
            "Aptos marketing could be more aggressive. Other chains are getting more attention",
            "Had a transaction fail on Aptos today. Even the best chains have issues sometimes",
            "The learning curve for Move programming is steep. Need better documentation",
            "Concerned about the centralization of validators on Aptos network",
            "Aptos needs more fiat on-ramps to improve accessibility for new users",
            "The tokenomics of APT could use some adjustments based on community feedback"
        ]
        
        all_posts = realistic_posts + negative_posts
        social_items = []
        now = datetime.now()
        
        # Generate 15-20 social posts
        num_posts = random.randint(15, 20)
        selected_posts = random.sample(all_posts, min(num_posts, len(all_posts)))
        
        for i, post in enumerate(selected_posts):
            # Create timestamps spread over the last 6 hours (more recent than news)
            hours_ago = random.randint(0, 6)
            minutes_ago = random.randint(0, 59)
            timestamp = now - timedelta(hours=hours_ago, minutes=minutes_ago)
            
            social_items.append({
                'timestamp': timestamp,
                'text': post,
                'source': random.choice(['Twitter', 'Reddit', 'Discord', 'Telegram']),
                'url': f'https://twitter.com/user/status/{random.randint(1000000000000000000, 9999999999999999999)}',
                'token': 'APT'
            })
        
        return social_items
    
    def fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch current market data for Aptos tokens.
        
        Returns:
            Dict: Market data including prices and volumes
        """
        try:
            # Try to fetch real market data from CoinGecko
            response = requests.get(
                'https://api.coingecko.com/api/v3/simple/price',
                params={
                    'ids': 'aptos',
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_24hr_vol': 'true'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                apt_data = data.get('aptos', {})
                
                return {
                    'APT': {
                        'price': apt_data.get('usd', 8.50),
                        'change_24h': apt_data.get('usd_24h_change', 0),
                        'volume_24h': apt_data.get('usd_24h_vol', 0)
                    },
                    'USDT': {
                        'price': 1.00,
                        'change_24h': 0,
                        'volume_24h': 0
                    }
                }
            else:
                raise Exception(f"API returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            # Return mock market data
            return {
                'APT': {
                    'price': 8.50 + random.uniform(-0.5, 0.5),
                    'change_24h': random.uniform(-5, 5),
                    'volume_24h': random.randint(50000000, 200000000)
                },
                'USDT': {
                    'price': 1.00,
                    'change_24h': random.uniform(-0.1, 0.1),
                    'volume_24h': random.randint(1000000000, 5000000000)
                }
            }
    
    def get_data_sources_status(self) -> Dict[str, bool]:
        """
        Check the status of all data sources.
        
        Returns:
            Dict: Status of each data source
        """
        status = {
            'cryptopanic_api': False,
            'rss_feeds': False,
            'social_media': False,
            'market_data': False
        }
        
        # Check CryptoPanic API
        try:
            api_key = os.getenv('CRYPTOPANIC_API_KEY', '')
            if api_key:
                response = requests.get(
                    self.news_apis['cryptopanic']['url'],
                    params={'auth_token': api_key, 'public': 'true'},
                    timeout=5
                )
                status['cryptopanic_api'] = response.status_code == 200
        except:
            pass
        
        # Check RSS feeds
        try:
            feed = feedparser.parse(self.news_apis['rss_feeds'][0])
            status['rss_feeds'] = len(feed.entries) > 0
        except:
            pass
        
        # Social media is currently mock data
        status['social_media'] = True
        
        # Check market data
        try:
            response = requests.get('https://api.coingecko.com/api/v3/ping', timeout=5)
            status['market_data'] = response.status_code == 200
        except:
            pass
        
        return status
