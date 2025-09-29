#!/usr/bin/env python3
"""
Telegram Generative AI Coding Agent (Single-File Vercel Edition)

This bot is a resilient, generative AI assistant built on the modern aiogram
framework, architected into a single file for simple, robust deployment on Vercel.
"""

import asyncio
import json
import os
import re
import logging
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

from aiohttp import web
import asyncpg
from groq import AsyncGroq
from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BotCommand, CallbackQuery, Update
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode

# --- Load .env for local development ---
load_dotenv()

# --- Configuration ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "your-super-secret")
WEBHOOK_PATH = "/api/index.py"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
WORKSPACE_PATH = Path("/tmp/workspace")
MAX_CODE_SIZE = 15000

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s', stream=sys.stdout)
logger = logging.getLogger("vercel-bot")

def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', str(text))

# --- Core Components ---

class RateLimiter:
    def __init__(self, max_calls: int = 25, period: float = 1.0):
        self.max_calls, self.period = max_calls, period
        self.user_calls: Dict[int, deque] = {}
        self.lock = asyncio.Lock()
    async def acquire(self, user_id: int):
        async with self.lock:
            now = datetime.now()
            calls = self.user_calls.setdefault(user_id, deque())
            while calls and calls[0] < now - timedelta(seconds=self.period):
                calls.popleft()
            if len(calls) >= self.max_calls:
                if (s := self.period - (now - calls[0]).total_seconds()) > 0:
                    await asyncio.sleep(s)
                if calls:
                    calls.popleft()
            calls.append(now)

class KnowledgeGraphManager:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None
    async def init(self):
        if not self.dsn:
            logger.error("DATABASE_URL is not set. KnowledgeGraphManager cannot be initialized.")
            return
        if not self.pool or self.pool.is_closing():
            try:
                self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
                logger.info("Database connection pool created.")
                await self._create_tables()
            except Exception as e:
                logger.critical(f"Failed to connect to database: {e}")
                self.pool = None
    async def _create_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS entities (id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, type TEXT);
                CREATE TABLE IF NOT EXISTS executions (id SERIAL PRIMARY KEY, user_id BIGINT NOT NULL, code TEXT, result JSONB, success BOOLEAN);
                CREATE TABLE IF NOT EXISTS observations (id SERIAL PRIMARY KEY, entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE, content TEXT, timestamp TIMESTAMPTZ DEFAULT NOW());
            ''')
        logger.info("Database tables verified/created.")
    async def log_execution(self, user_id: int, code: str, res: dict, suc: bool) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval('INSERT INTO executions (user_id, code, result, success) VALUES ($1, $2, $3, $4) RETURNING id', user_id, code, json.dumps(res), suc)
    async def get_execution_details(self, exec_id: int) -> Optional[Tuple[str, str]]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT code, result FROM executions WHERE id = $1', exec_id)
            return (row['code'], json.dumps(row['result'])) if row else None
    async def remember(self, user_id: int, text: str):
        async with self.pool.acquire() as conn:
            user_entity = f"user_{user_id}"
            await conn.execute('INSERT INTO entities (name, type) VALUES ($1, $2) ON CONFLICT (name) DO NOTHING', user_entity, "user")
            entity_id = await conn.fetchval('SELECT id FROM entities WHERE name = $1', user_entity)
            await conn.execute('INSERT INTO observations (entity_id, content) VALUES ($1, $2)', entity_id, text)
    async def recall(self, user_id: int) -> str:
        async with self.pool.acquire() as conn:
            user_entity = f"user_{user_id}"
            rows = await conn.fetch("SELECT o.content FROM observations o JOIN entities e ON e.id = o.entity_id WHERE e.name = $1 ORDER BY o.timestamp DESC LIMIT 10", user_entity)
            if not rows: return "No recent memories found\\."
            obs = [row['content'] for row in rows]
            return f"Recent memories for *{escape_markdown_v2(user_entity)}*:\n\\- " + "\n\\- ".join(escape_markdown_v2(o) for o in obs)
    async def close(self):
        if self.pool: await self.pool.close(); logger.info("Database connection pool closed.")

class SecurityValidator:
    @staticmethod
    def validate_code(code: str) -> Tuple[bool, str]:
        if not code.strip(): return False, "Empty code"
        if len(code) > MAX_CODE_SIZE: return False, f"Code too long (max {MAX_CODE_SIZE} chars)"
        if re.search(r'rm\s+-rf\s+/', code): return False, "Dangerous pattern"
        return True, ""

class CodeExecutor:
    async def execute(self, code: str, lang: str) -> Dict[str, Any]:
        ext, cmd = {"python":(".py",[sys.executable]),"javascript":(".js",["node"]),"bash":(".sh",["bash"])}.get(lang,(None,None))
        if not ext: return {"error": f"Unsupported language: {lang}"}
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, dir=WORKSPACE_PATH) as f:
                f.write(code); temp_file = f.name
            p = await asyncio.create_subprocess_exec(*cmd, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(p.communicate(), timeout=60)
            os.unlink(temp_file)
            return {"stdout":out.decode(),"stderr":err.decode(),"returncode":p.returncode}
        except asyncio.TimeoutError: return {"error": "Execution timed out."}
        except Exception as e: return {"error": str(e)}

class GeminiLLM:
    def __init__(self, api_key: str, model_name: str):
        import google.generativeai as genai
        self.model = genai.GenerativeModel(model_name=model_name, api_key=api_key)
    async def generate(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt); return response.text

class GroqLLM:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncGroq(api_key=api_key); self.model = model
    async def generate(self, prompt: str) -> str:
        completion = await self.client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=self.model); return completion.choices[0].message.content

class LLMManager:
    def __init__(self, gemini_key: str, groq_key: str, gemini_model: str, groq_model: str):
        self.primary_llm = GeminiLLM(gemini_key, gemini_model) if gemini_key else None
        self.fallback_llm = GroqLLM(groq_key, groq_model) if groq_key else None
    async def generate(self, prompt: str) -> str:
        if self.primary_llm:
            try:
                return await self.primary_llm.generate(prompt)
            except Exception as e: logger.warning(f"Primary LLM failed: {e}. Failing over.")
        if self.fallback_llm:
            try:
                return await self.fallback_llm.generate(prompt)
            except Exception as e2: logger.error(f"Fallback LLM failed: {e2}")
        return "Both AI models are unavailable. Please try again later."

# --- Module-level Singleton Initialization ---
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2))
dp = Dispatcher()
router = Router()
dp.include_router(router)
db_manager = KnowledgeGraphManager(dsn=DATABASE_URL)
llm_manager = LLMManager(GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL)
rate_limiter = RateLimiter()
executor = CodeExecutor()
security_validator = SecurityValidator()

# --- Command Handlers ---
@router.message(CommandStart())
async def cmd_start(message: Message):
    text = "👋 *Welcome to your AI Coding Agent*\\!\n\nI can build things, execute code, and remember information for you."
    kb = InlineKeyboardBuilder().button(text="🚀 Try a Build Demo", callback_data="demo:build")
    await message.answer(text, reply_markup=kb.as_markup())

@router.message(Command("build"))
async def cmd_build(message: Message):
    prompt_text = message.text.replace("/build", "", 1).strip()
    if not prompt_text: await message.answer("Please provide a prompt after `/build`\\."); return
    thinking_msg = await message.answer("🧠 Thinking and building\\.\\.\\.")
    prompt = f"You are an expert AI developer. Generate a response in MDX-like format for Telegram's MarkdownV2.\n\n**User Request:** \"{prompt_text}\""
    response = await llm_manager.generate(prompt); await thinking_msg.edit_text(response)

@router.message(Command("python", "p", "node", "n", "bash", "b"))
async def cmd_execute(message: Message):
    cmd_map = {"p": "python", "n": "node", "b": "bash"}
    command, *code_parts = message.text.split(maxsplit=1)
    lang_key = command[1:]; lang = cmd_map.get(lang_key, lang_key)
    code = code_parts[0] if code_parts else ""
    if not code: await message.answer(f"Usage: `/{lang_key} <code>`"); return
    valid, err = security_validator.validate_code(code)
    if not valid: await message.answer(f"❌ *Invalid Code*: {escape_markdown_v2(err)}"); return
    status_msg = await message.answer(f"🚀 Executing {lang}\\.\\.\\.")
    result = await executor.execute(code, lang)
    exec_id = await db_manager.log_execution(message.from_user.id, code, result, result.get("returncode") == 0)
    rc = result.get("returncode")
    if "error" in result: response = f"❌ *Execution Error*:\n```\n{escape_markdown_v2(result['error'])}\n```"
    elif rc != 0: response = f"❌ *Runtime Error* (Exit: {rc}):\n```\n{escape_markdown_v2(result.get('stderr', ''))}\n```"
    else: response = f"✅ *Output*:\n```\n{escape_markdown_v2(result.get('stdout', 'No output'))}\n```"
    kb = InlineKeyboardBuilder().button(text="🔍 Debug with AI", callback_data=f"analyze:{exec_id}")
    await status_msg.edit_text(response, reply_markup=kb.as_markup())

@router.message(Command("remember", "recall"))
async def cmd_memory(message: Message):
    command, *args = message.text.split(maxsplit=1)
    action = command[1:]
    if action == "remember":
        text_to_remember = args[0] if args else ""
        if not text_to_remember: await message.answer("Usage: `/remember <text to save>`"); return
        await db_manager.remember(message.from_user.id, text_to_remember); await message.answer("✅ Memory updated.")
    elif action == "recall":
        await message.answer(await db_manager.recall(message.from_user.id))

@router.message(F.text)
async def handle_text_conversation(message: Message):
    thinking_msg = await message.answer("🤖 Thinking\\.\\.\\.")
    user_memory = await db_manager.recall(message.from_user.id)
    prompt = f"You are a helpful AI assistant. Use the user's memory for a personalized response.\n**Memory Context**:\n{user_memory}\n\n**User Message**:\n{message.text}"
    response = await llm_manager.generate(prompt); await thinking_msg.edit_text(response)

@router.callback_query()
async def handle_callback(query: CallbackQuery):
    await query.answer()
    action, data = query.data.split(":", 1)
    if action == "analyze":
        details = await db_manager.get_execution_details(int(data))
        if details:
            code, result_json = details; result = json.loads(result_json)
            error_context = result.get("stderr") or result.get("error") if result.get("returncode") != 0 else "No error output."
            await query.message.edit_reply_markup(reply_markup=None)
            thinking_msg = await query.message.answer("🔍 Debugging with AI\\.\\.\\.")
            prompt = f"Act as an expert debugger for the failed code.\n\n**Code:**\n```\n{code}\n```\n\n**Error:**\n```\n{error_context}\n```\n\nProvide a root cause analysis and a suggested fix."
            analysis = await llm_manager.generate(prompt); await thinking_msg.edit_text(f"*AI Debugger Analysis:*\n\n{analysis}")
    elif action == "demo":
        await query.message.answer("Try this: `/build a python script that prints the current time`")

# --- Vercel ASGI Application ---
app = web.Application()

async def webhook_handler(request: web.Request) -> web.Response:
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        logger.warning("Invalid secret token."); return web.Response(status=403)
    try:
        update = Update.model_validate(await request.json(), context={"bot": bot})
        await dp.feed_update(update)
        return web.Response(status=200)
    except Exception as e:
        logger.error(f"Error processing update: {e}", exc_info=True)
        return web.Response(status=500)

async def on_startup(app_instance: web.Application):
    logger.info("Initializing application resources..."); WORKSPACE_PATH.mkdir(exist_ok=True)
    await db_manager.init(); logger.info("Application resources initialized.")

async def on_shutdown(app_instance: web.Application):
    logger.info("Closing application resources..."); await db_manager.close()
    if bot.session and not bot.session.closed: await bot.session.close()
    logger.info("Application resources closed.")

app.router.add_post(WEBHOOK_PATH, webhook_handler)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)