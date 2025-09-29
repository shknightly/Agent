#!/usr/bin/env python3
"""
Vercel ASGI Entry Point for the Telegram Bot

This script creates an aiohttp web application that serves as the webhook
endpoint for the aiogram bot. It imports the core bot logic from `bot_logic.py`,
initializes all components, and handles the application lifecycle.
"""

import os
import logging
import sys
from dotenv import load_dotenv
from aiohttp import web

from aiogram import types
from bot_logic import TelegramBot, llm_manager, GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL, WORKSPACE_PATH

# --- Load .env for local development ---
load_dotenv()

# --- Configuration ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "your-super-secret")
WEBHOOK_PATH = "/api" # Vercel routes requests to /api/index.py here

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s', stream=sys.stdout)
logger = logging.getLogger("vercel-entrypoint")

# --- Bot Initialization ---
# Initialize the main bot class from our logic file
bot_instance = TelegramBot(bot_token=BOT_TOKEN, database_url=DATABASE_URL)

async def on_startup(app: web.Application):
    """
    Initialize resources when the application starts.
    This is where we set up the bot's handlers, LLM, and database pool.
    """
    logger.info("Initializing application resources...")

    # Create the workspace directory if it doesn't exist
    WORKSPACE_PATH.mkdir(exist_ok=True)

    # Initialize the LLM Manager
    bot_instance.llm = llm_manager
    if not GEMINI_API_KEY or not GROQ_API_KEY:
        logger.warning("One or more AI API keys are missing.")

    # Register all command and message handlers
    bot_instance.register_handlers()

    # Initialize the database connection pool
    await bot_instance.db.init()

    logger.info("Application resources initialized successfully.")

async def on_shutdown(app: web.Application):
    """
    Clean up resources when the application shuts down.
    """
    logger.info("Closing application resources...")
    await bot_instance.db.close()
    if bot_instance.bot.session and not bot_instance.bot.session.closed:
        await bot_instance.bot.session.close()
    logger.info("Application resources closed.")

async def webhook_handler(request: web.Request) -> web.Response:
    """
    Handles incoming webhook requests from Telegram.
    """
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        logger.warning("Invalid secret token received.")
        return web.Response(status=403, text="Forbidden")

    try:
        # Get the JSON update from the request
        update_data = await request.json()

        # Create an aiogram Update object and feed it to the dispatcher
        update = types.Update.model_validate(update_data, context={"bot": bot_instance.bot})
        await bot_instance.dp.feed_update(update)

        return web.Response(status=200)
    except Exception as e:
        logger.error(f"Error processing update: {e}", exc_info=True)
        return web.Response(status=500, text="Internal Server Error")

# --- Vercel ASGI Application ---
# The 'app' variable is what Vercel's Python runtime looks for.
app = web.Application()

# Register the webhook handler route
app.router.add_post(WEBHOOK_PATH, webhook_handler)

# Register the startup and shutdown event handlers
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)