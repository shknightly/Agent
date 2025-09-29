import asyncio
import os
import logging
import sys
from dotenv import load_dotenv
from aiogram import Bot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

async def setup_webhook():
    """
    A one-time script to set the Telegram webhook for the Vercel deployment.
    This should be run locally after deploying to Vercel or if the Vercel URL changes.
    """
    load_dotenv()

    BOT_TOKEN = os.environ.get("BOT_TOKEN")
    VERCEL_URL = os.environ.get("VERCEL_URL")

    if not BOT_TOKEN or not VERCEL_URL:
        logger.critical(
            "FATAL: Missing required environment variables: BOT_TOKEN, VERCEL_URL. "
            "Please create a .env file with these values or set them in your environment."
        )
        sys.exit(1)

    # This must match the entry point file in the /api directory
    WEBHOOK_PATH = "/api/index.py"
    WEBHOOK_URL = f"https://{VERCEL_URL}{WEBHOOK_PATH}"

    bot = Bot(token=BOT_TOKEN)

    try:
        logger.info(f"Setting webhook to: {WEBHOOK_URL}")
        await bot.set_webhook(url=WEBHOOK_URL)
        webhook_info = await bot.get_webhook_info()

        logger.info("--- Webhook Information ---")
        logger.info(f"URL: {webhook_info.url}")
        logger.info(f"Has custom certificate: {webhook_info.has_custom_certificate}")
        logger.info(f"Pending update count: {webhook_info.pending_update_count}")
        if webhook_info.last_error_date:
            logger.error(f"Last error date: {webhook_info.last_error_date}")
            logger.error(f"Last error message: {webhook_info.last_error_message}")
        logger.info("--------------------------")
        logger.info("✅ Webhook set successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to set webhook: {e}")
    finally:
        if bot.session and not bot.session.closed:
            await bot.session.close()

if __name__ == "__main__":
    try:
        import aiogram
        import python_dotenv
    except ImportError:
        print("Missing dependencies. Please run 'pip install -r requirements.txt' first.")
        sys.exit(1)

    asyncio.run(setup_webhook())