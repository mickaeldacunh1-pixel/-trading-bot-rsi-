#!/usr/bin/env python3
import logging
import os
from apscheduler.schedulers.blocking import BlockingScheduler

os.makedirs('logs', exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from binance.client import Client
import pandas as pd

class SimpleBot:
    def __init__(self):
        self.client = Client(os.environ['BINANCE_API_KEY'], os.environ['BINANCE_API_SECRET'])
        self.symbol = os.environ.get('TRADING_SYMBOL', 'BTCUSDT')
        self.max_amount = float(os.environ.get('MAX_TRADE_AMOUNT', '50'))
        
    def calculate_rsi(self, prices):
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0).rolling(window=14).mean()
        losses = -deltas.where(deltas < 0, 0).rolling(window=14).mean()
        rs = gains / losses
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    def run(self):
        try:
            logger.info("🤖 Bot Trading RSI - Exécution")
            
            klines = self.client.get_klines(symbol=self.symbol, interval='1h', limit=100)
            prices = pd.DataFrame(klines)[4].astype(float)
            
            rsi = self.calculate_rsi(prices)
            current_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])
            
            logger.info(f"💰 Prix: ${current_price:.2f} | RSI: {rsi:.2f}")
            
            if rsi < 30:
                logger.info(f"🟢 SIGNAL ACHAT - RSI en survente ({rsi:.2f})")
            elif rsi > 70:
                logger.info(f"🔴 SIGNAL VENTE - RSI en surachat ({rsi:.2f})")
            else:
                logger.info(f"⏸️ ATTENTE - RSI neutre ({rsi:.2f})")
                
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")

bot = SimpleBot()

def execute():
    bot.run()

def main():
    logger.info("=" * 60)
    logger.info("🚀 Démarrage Bot Trading RSI")
    logger.info("=" * 60)
    
    scheduler = BlockingScheduler()
    scheduler.add_job(execute, 'interval', hours=1)
    
    logger.info("⏰ Exécution toutes les heures")
    execute()
    scheduler.start()

if __name__ == "__main__":
    main()
