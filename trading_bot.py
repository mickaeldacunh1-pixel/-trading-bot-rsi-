Home Tab
Home

crypto-portfolio-57

cryptowise-1


Balance Coin
45,23
right arrow
Buy Credits


M

info

code
Code

preview
Preview

Deploy
Redeploy
Robot
Hey Mike, Quick input needed :
Ask Question
Agent will continue working after your reply
📄 Voici le code complet à copier
Sélectionnez TOUT le texte ci-dessous (de la première ligne à la dernière) et copiez-le :

"""
Binance RSI Trading Bot - Automated Trading System
"""
import os
import logging
import time
from datetime import datetime, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class RSITradingBot:
    """RSI-based trading bot for Binance"""
    
    def __init__(self):
        """Initialize the trading bot with Binance credentials"""
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        self.testnet = os.environ.get('BINANCE_TESTNET', 'false').lower() == 'true'
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials not found in environment variables")
        
        # Initialize Binance client
        self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
        
        # Trading parameters
        self.symbol = os.environ.get('TRADING_SYMBOL', 'BTCUSDT')
        self.rsi_period = int(os.environ.get('RSI_PERIOD', '14'))
        self.rsi_oversold = float(os.environ.get('RSI_OVERSOLD', '30'))
        self.rsi_overbought = float(os.environ.get('RSI_OVERBOUGHT', '70'))
        self.stop_loss_pct = float(os.environ.get('STOP_LOSS_PERCENTAGE', '5'))
        self.take_profit_pct = float(os.environ.get('TAKE_PROFIT_PERCENTAGE', '8'))
        self.max_trade_amount = float(os.environ.get('MAX_TRADE_AMOUNT', '50'))
        
        # Trading state
        self.position = None  # None, 'LONG'
        self.entry_price = None
        self.position_amount = 0
        self.last_rsi = None
        self.prev_rsi = None
        
        logger.info(f"RSI Trading Bot initialized for {self.symbol}")
        logger.info(f"Parameters: RSI={self.rsi_period}, Oversold={self.rsi_oversold}, Overbought={self.rsi_overbought}")
    
    def get_account_balance(self, asset: str = 'USDT') -> float:
        """Get account balance for specific asset"""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free']) if balance else 0.0
        except BinanceAPIException as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_historical_klines(self, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get historical kline/candlestick data"""
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=interval, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['open'] = pd.to_numeric(df['open'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            return df
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI indicator (Python implementation)"""
        try:
            if len(prices) < self.rsi_period + 1:
                return None
            
            # Calculate price changes
            deltas = prices.diff()
            
            # Separate gains and losses
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gains.rolling(window=self.rsi_period).mean()
            avg_loss = losses.rolling(window=self.rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Return last RSI value
            last_rsi = rsi.iloc[-1]
            return last_rsi if not pd.isna(last_rsi) else None
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None
    
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def place_market_buy(self, usdt_amount: float) -> Dict:
        """Place market buy order"""
        try:
            # Get current price to calculate quantity
            current_price = self.get_current_price()
            if not current_price:
                raise Exception("Could not get current price")
            
            # Calculate quantity (BTC amount)
            quantity = usdt_amount / current_price
            
            # Get symbol info for LOT_SIZE filter
            info = self.client.get_symbol_info(self.symbol)
            lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                # Round down to step size
                quantity = np.floor(quantity / step_size) * step_size
                # Format to remove trailing zeros
                precision = len(str(step_size).rstrip('0').split('.')[-1])
                quantity = float(f"{quantity:.{precision}f}")
            
            logger.info(f"Placing BUY order: {quantity} {self.symbol} at ~${current_price:.2f}")
            
            order = self.client.order_market_buy(
                symbol=self.symbol,
                quantity=quantity
            )
            
            logger.info(f"BUY order executed: Order ID {order['orderId']}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Error placing buy order: {e}")
            raise
    
    def place_market_sell(self, quantity: float) -> Dict:
        """Place market sell order"""
        try:
            # Get symbol info for LOT_SIZE filter
            info = self.client.get_symbol_info(self.symbol)
            lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                # Round down to step size
                quantity = np.floor(quantity / step_size) * step_size
                # Format to remove trailing zeros
                precision = len(str(step_size).rstrip('0').split('.')[-1])
                quantity = float(f"{quantity:.{precision}f}")
            
            logger.info(f"Placing SELL order: {quantity} {self.symbol}")
            
            order = self.client.order_market_sell(
                symbol=self.symbol,
                quantity=quantity
            )
            
            logger.info(f"SELL order executed: Order ID {order['orderId']}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Error placing sell order: {e}")
            raise
    
    def check_stop_loss_take_profit(self, current_price: float) -> bool:
        """Check if stop-loss or take-profit should trigger"""
        if not self.position or not self.entry_price:
            return False
        
        price_change_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Check stop-loss
        if price_change_pct <= -self.stop_loss_pct:
            logger.warning(f"🛑 Stop-loss triggered! Price change: {price_change_pct:.2f}%")
            return True
        
        # Check take-profit
        if price_change_pct >= self.take_profit_pct:
            logger.info(f"🎯 Take-profit triggered! Price change: {price_change_pct:.2f}%")
            return True
        
        return False
    
    def execute_trade_logic(self) -> Dict:
        """Main trading logic execution"""
        try:
            # Get historical data
            df = self.get_historical_klines(interval='1h', limit=100)
            if df.empty:
                return {"status": "error", "message": "No historical data"}
            
            # Calculate RSI
            self.prev_rsi = self.last_rsi
            self.last_rsi = self.calculate_rsi(df['close'])
            
            if self.last_rsi is None:
                return {"status": "error", "message": "Could not calculate RSI"}
            
            # Get current price
            current_price = self.get_current_price()
            if not current_price:
                return {"status": "error", "message": "Could not get current price"}
            
            logger.info(f"📊 Current Price: ${current_price:.2f} | RSI: {self.last_rsi:.2f}")
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": current_price,
                "rsi": self.last_rsi,
                "position": self.position,
                "action": "HOLD"
            }
            
            # Check stop-loss / take-profit if in position
            if self.position == 'LONG':
                if self.check_stop_loss_take_profit(current_price):
                    # Close position
                    order = self.place_market_sell(self.position_amount)
                    profit_loss = (current_price - self.entry_price) / self.entry_price * 100
                    
                    result.update({
                        "action": "SELL",
                        "reason": "Stop-loss/Take-profit triggered",
                        "entry_price": self.entry_price,
                        "exit_price": current_price,
                        "profit_loss_pct": profit_loss,
                        "order": order
                    })
                    
                    self.position = None
                    self.entry_price = None
                    self.position_amount = 0
                    
                    logger.info(f"✅ Position closed. P/L: {profit_loss:.2f}%")
                    return result
            
            # Trading signals
            if self.prev_rsi is not None:
                # BUY signal: RSI crosses below oversold threshold
                if self.position is None and self.prev_rsi > self.rsi_oversold and self.last_rsi <= self.rsi_oversold:
                    logger.info(f"🚀 BUY SIGNAL: RSI crossed below {self.rsi_oversold}")
                    
                    # Check USDT balance
                    usdt_balance = self.get_account_balance('USDT')
                    trade_amount = min(self.max_trade_amount, usdt_balance * 0.9)  # Use 90% max
                    
                    if trade_amount >= 10:  # Minimum $10
                        order = self.place_market_buy(trade_amount)
                        self.position = 'LONG'
                        self.entry_price = current_price
                        self.position_amount = float(order['executedQty'])
                        
                        result.update({
                            "action": "BUY",
                            "reason": "RSI oversold signal",
                            "entry_price": current_price,
                            "amount_usdt": trade_amount,
                            "amount_btc": self.position_amount,
                            "order": order
                        })
                    else:
                        result["action"] = "BUY_SIGNAL_IGNORED"
                        result["reason"] = f"Insufficient balance: ${usdt_balance:.2f}"
                
                # SELL signal: RSI crosses above overbought threshold
                elif self.position == 'LONG' and self.prev_rsi < self.rsi_overbought and self.last_rsi >= self.rsi_overbought:
                    logger.info(f"📉 SELL SIGNAL: RSI crossed above {self.rsi_overbought}")
                    
                    order = self.place_market_sell(self.position_amount)
                    profit_loss = (current_price - self.entry_price) / self.entry_price * 100
                    
                    result.update({
                        "action": "SELL",
                        "reason": "RSI overbought signal",
                        "entry_price": self.entry_price,
                        "exit_price": current_price,
                        "profit_loss_pct": profit_loss,
                        "order": order
                    })
                    
                    self.position = None
                    self.entry_price = None
                    self.position_amount = 0
                    
                    logger.info(f"✅ Position closed. P/L: {profit_loss:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in trading logic: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        try:
            current_price = self.get_current_price()
            usdt_balance = self.get_account_balance('USDT')
            btc_balance = self.get_account_balance('BTC')
            
            status = {
                "symbol": self.symbol,
                "current_price": current_price,
                "rsi": self.last_rsi,
                "position": self.position,
                "entry_price": self.entry_price,
                "position_amount": self.position_amount,
                "usdt_balance": usdt_balance,
                "btc_balance": btc_balance,
                "parameters": {
                    "rsi_period": self.rsi_period,
                    "rsi_oversold": self.rsi_oversold,
                    "rsi_overbought": self.rsi_overbought,
                    "stop_loss_pct": self.stop_loss_pct,
                    "take_profit_pct": self.take_profit_pct,
                    "max_trade_amount": self.max_trade_amount
                }
            }
            
            if self.position and self.entry_price and current_price:
                status["unrealized_pnl_pct"] = ((current_price - self.entry_price) / self.entry_price) * 100
                status["unrealized_pnl_usdt"] = (current_price - self.entry_price) * self.position_amount
            
            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}


# Global bot instance
trading_bot = None

def get_trading_bot() -> RSITradingBot:
    """Get or create the global trading bot instance"""
    global trading_bot
    if trading_bot is None:
        trading_bot = RSITradingBot()
    return trading_bot
