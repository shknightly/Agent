#!/usr/bin/env python3
"""
Telegram Generative AI Coding Agent (aiogram Edition)

This bot is a resilient, generative AI assistant built on the modern aiogram
framework. It is fully compliant with the latest Google Gemini API client.

**BEFORE RUNNING:**
You must install the required packages using pip:
pip install -r requirements.txt

Core Features:
- Modern Async Framework: Built on aiogram for high performance.
- Resilient AI: Uses Gemini as the primary model and automatically fails over
  to Groq if Gemini is unavailable.
- Generative Code: Creates code from natural language prompts via the /build command.
- Advanced Tooling: Retains direct code execution, AI-powered debugging,
  and a local Knowledge Graph memory using SQLite.
- Polling-Only: Optimized for simplicity.

Environment Variables (to be placed in a .env file):
    BOT_TOKEN          - Your Telegram Bot API token.
    GEMINI_API_KEY     - Your Google Gemini API key.
    GROQ_API_KEY       - Your Groq API key.
"""

import asyncio
import json
import os
import re
import logging
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

import aiohttp
import asyncpg
from groq import AsyncGroq
from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BotCommand, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode

# --- Load .env and Configure ---
load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
WORKSPACE_PATH = Path("/tmp/workspace")
MAX_CODE_SIZE = 15000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s', stream=sys.stdout)
logger = logging.getLogger("termux-bot")

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
            now = datetime.now(); calls = self.user_calls.setdefault(user_id, deque())
            while calls and calls[0] < now - timedelta(seconds=self.period): calls.popleft()
            if len(calls) >= self.max_calls:
                if (s := self.period - (now - calls[0]).total_seconds()) > 0: await asyncio.sleep(s)
                if calls: calls.popleft()
            calls.append(now)

class KnowledgeGraphManager:
    """Manages the connection and queries to the PostgreSQL database."""
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def init(self):
        if not self.dsn:
            logger.error("DATABASE_URL is not set. KnowledgeGraphManager cannot be initialized.")
            return
        if not self.pool:
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
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT
                );
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS executions (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    code TEXT,
                    result JSONB,
                    success BOOLEAN
                );
            ''')
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS observations (
                    id SERIAL PRIMARY KEY,
                    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    content TEXT,
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
            ''')
        logger.info("Database tables verified/created.")

    async def log_execution(self, user_id: int, code: str, res: dict, suc: bool) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                'INSERT INTO executions (user_id, code, result, success) VALUES ($1, $2, $3, $4) RETURNING id',
                user_id, code, json.dumps(res), suc
            )

    async def get_execution_details(self, exec_id: int) -> Optional[Tuple[str, str]]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT code, result FROM executions WHERE id = $1', exec_id)
            return (row['code'], json.dumps(row['result'])) if row else None

    async def remember(self, user_id: int, text: str):
        async with self.pool.acquire() as conn:
            user_entity = f"user_{user_id}"
            await conn.execute(
                'INSERT INTO entities (name, type) VALUES ($1, $2) ON CONFLICT (name) DO NOTHING',
                user_entity, "user"
            )
            entity_id = await conn.fetchval('SELECT id FROM entities WHERE name = $1', user_entity)
            await conn.execute('INSERT INTO observations (entity_id, content) VALUES ($1, $2)', entity_id, text)

    async def recall(self, user_id: int) -> str:
        async with self.pool.acquire() as conn:
            user_entity = f"user_{user_id}"
            rows = await conn.fetch(
                '''
                SELECT o.content FROM observations o
                JOIN entities e ON e.id = o.entity_id
                WHERE e.name = $1 ORDER BY o.timestamp DESC LIMIT 10
                ''',
                user_entity
            )
            if not rows: return "No recent memories found\\."
            obs = [row['content'] for row in rows]
            return f"Recent memories for *{escape_markdown_v2(user_entity)}*:\n\\- " + "\n\\- ".join(escape_markdown_v2(o) for o in obs)

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed.")

class SecurityValidator:
    @staticmethod
    def validate_code(code: str) -> Tuple[bool, str]:
        if not code.strip(): return False, "Empty code"
        if len(code) > MAX_CODE_SIZE: return False, f"Code too long (max {MAX_CODE_SIZE} chars)"
        if re.search(r'rm\s+-rf\s+/', code): return False, "Dangerous pattern: Recursive root deletion"
        return True, ""

class CodeExecutor:
    async def execute(self, code: str, lang: str) -> Dict[str, Any]:
        ext, cmd = {"python":(".py",[sys.executable]),"javascript":(".js",["node"]),"bash":(".sh",["bash"])}.get(lang,(None,None))
        if not ext: return {"error": f"Unsupported language: {lang}"}
        try:
            with tempfile.NamedTemporaryFile(mode='w',suffix=ext,delete=False) as f: f.write(code); temp_file=f.name
            p = await asyncio.create_subprocess_exec(*cmd, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(p.communicate(), timeout=60)
            os.unlink(temp_file)
            return {"stdout":out.decode(),"stderr":err.decode(),"returncode":p.returncode}
        except asyncio.TimeoutError: return {"error": "Execution timed out."}
        except Exception as e: return {"error": str(e)}

class GeminiLLM:
    """
    Updated GeminiLLM class compliant with the latest google-generativeai API.
    """
    def __init__(self, api_key: str, model_name: str):
        import google.generativeai as genai
        # The new, recommended way to initialize the client
        self.model = genai.GenerativeModel(
            model_name=model_name,
            api_key=api_key
        )
    async def generate(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text

class GroqLLM:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
    async def generate(self, prompt: str) -> str:
        completion = await self.client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=self.model)
        return completion.choices[0].message.content

class LLMManager:
    def __init__(self, gemini_key: str, groq_key: str, gemini_model: str, groq_model: str):
        self.primary_llm = GeminiLLM(gemini_key, gemini_model)
        self.fallback_llm = GroqLLM(groq_key, groq_model)
    async def generate(self, prompt: str) -> str:
        try:
            logger.info("Attempting to generate with primary model (Gemini)...")
            response = await self.primary_llm.generate(prompt)
            if not response: raise ValueError("Empty response from Gemini")
            logger.info("Successfully generated with Gemini.")
            return response
        except Exception as e:
            logger.warning(f"Primary model (Gemini) failed: {e}. Failing over to Groq.")
            try:
                response = await self.fallback_llm.generate(prompt)
                logger.info("Successfully generated with fallback model (Groq).")
                return response
            except Exception as e2:
                logger.error(f"Fallback model (Groq) also failed: {e2}")
                return "Both primary and fallback AI models are currently unavailable. Please try again later."

# --- Main Bot Class ---
class TelegramBot:
    def __init__(self, bot_token: str, database_url: str):
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2))
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)
        self.db = KnowledgeGraphManager(dsn=database_url)
        self.rate_limiter = RateLimiter()
        self.executor = CodeExecutor()
        self.llm: Optional[LLMManager] = None

    def register_handlers(self):
        self.router.message(CommandStart())(self.cmd_start)
        self.router.message(Command("build"))(self.cmd_build)
        self.router.message(Command("python", "p", "node", "n", "bash", "b"))(self.cmd_execute)
        self.router.message(Command("remember", "recall"))(self.cmd_memory)
        self.router.callback_query()(self.handle_callback)
        self.router.message(F.text)(self.handle_text_conversation)

    async def setup_commands(self):
        cmds = [BotCommand(c,d) for c,d in [("start","Restart & see help"),("build","Create code/content with a prompt"),("p","Execute Python"),("n","Execute Node.js"),("b","Execute Bash"),("remember","Save info to memory"),("recall","View your memory")]]
        await self.bot.set_my_commands(cmds, scope=types.BotCommandScopeAllPrivateChats())

    async def send_safe(self, chat_id: int, text: str, **kwargs):
        await self.rate_limiter.acquire(chat_id)
        if len(text) <= 4096:
            try: return await self.bot.send_message(chat_id, text, **kwargs)
            except Exception as e: logger.error(f"Failed to send message: {e}")
        else:
            parts = [text[i:i+4096] for i in range(0, len(text), 4096)]; msg = None
            for p in parts: msg = await self.send_safe(chat_id, p, **kwargs)
            return msg

    async def cmd_start(self, message: Message):
        text = ("👋 *Welcome to your AI Coding Agent*\\!\n\n"
                "I use Gemini and Groq for resilient, high-speed responses\\. The primary command is `/build`\\.\n\n"
                "*Example:*\n`/build a simple React component with a button that shows an alert`\n\n"
                "You can also use advanced tools for direct code execution \\(`/p`, `/n`, `/b`\\) and memory management \\(`/remember`, `/recall`\\)\\.")
        kb = InlineKeyboardBuilder(); kb.button(text="🚀 Try a Build Demo", callback_data="demo:build")
        await message.answer(text, reply_markup=kb.as_markup())

    async def cmd_build(self, message: Message):
        prompt_text = message.text.replace("/build", "", 1).strip()
        if not prompt_text: await message.answer("Please provide a prompt after `/build`\\."); return
        thinking_msg = await message.answer("🧠 Thinking and building\\.\\.\\.")
        prompt = (
            "You are an expert AI developer building things from a single prompt. Your responses must be in MDX-like format within Telegram's MarkdownV2.\n"
            "**Core Instructions:**\n"
            "1.  **Chain of Thought**: First, think step-by-step to create a plan. This is for your internal use and should not be in the final output.\n"
            "2.  **Language Detection**: The user is asking in a specific language. All your explanations and code comments must be in that same language.\n"
            "3.  **Code Generation**: Generate the code or content requested.\n"
            "4.  **Formatted Output**: Wrap all generated code in extended code blocks with metadata (e.g., ```language project=\"Name\" file=\"path/file.ext\" type=\"...\"). For non-code, use structured markdown.\n"
            "5.  **Explanation**: Provide a concise explanation of what you've built.\n\n"
            f"**User Request:** \"{prompt_text}\""
        )
        response = await self.llm.generate(prompt)
        await thinking_msg.edit_text(response)

    async def cmd_execute(self, message: Message):
        cmd_map = {"p": "python", "n": "node", "b": "bash"}
        parts = message.text.split(maxsplit=1)
        lang = cmd_map.get(parts[1:], parts[1:])
        code = parts if len(parts) > 1 else ""
        if not code: await message.answer(f"Usage: `/{lang} <code>`"); return
        valid, err = SecurityValidator.validate_code(code)
        if not valid: await message.answer(f"❌ *Invalid Code*: {escape_markdown_v2(err)}"); return

        status_msg = await message.answer(f"🚀 Executing {lang}\\.\\.\\.")
        result = await self.executor.execute(code, lang)
        exec_id = await self.db.log_execution(message.from_user.id, code, result, result.get("returncode") == 0)

        rc = result.get("returncode")
        if "error" in result: response = f"❌ *Execution Error*:\n```\n{escape_markdown_v2(result['error'])}\n```"
        elif rc != 0: response = f"❌ *Runtime Error* (Exit: {rc}):\n```\n{escape_markdown_v2(result.get('stderr', ''))}\n```"
        else: response = f"✅ *Output*:\n```\n{escape_markdown_v2(result.get('stdout', 'No output'))}\n```"

        kb = InlineKeyboardBuilder(); kb.button(text="🔍 Debug with AI", callback_data=f"analyze:{exec_id}")
        await status_msg.edit_text(response, reply_markup=kb.as_markup())

    async def cmd_memory(self, message: Message):
        command, *args = message.text.split(maxsplit=1)
        action = command[1:]

        if action == "remember":
            text_to_remember = " ".join(args).strip()
            if not text_to_remember: await message.answer("Usage: `/remember <text to save>`"); return
            await self.db.remember(message.from_user.id, text_to_remember)
            await message.answer("✅ Memory updated.")
        elif action == "recall":
            recalled_data = await self.db.recall(message.from_user.id)
            await message.answer(recalled_data)

    async def handle_text_conversation(self, message: Message):
        thinking_msg = await message.answer("🤖 Thinking\\.\\.\\.")
        user_memory = await self.db.recall(message.from_user.id)
        prompt = (f"You are a helpful AI assistant. Use the user's memory to provide a personalized response. Format for MarkdownV2.\n"
                  f"**Memory Context**:\n{user_memory}\n\n**User Message**:\n{message.text}")
        response = await self.llm.generate(prompt)
        await thinking_msg.edit_text(response)

    async def handle_callback(self, query: CallbackQuery):
        await query.answer()
        action, data = query.data.split(":", 1)
        if action == "analyze":
            details = await self.db.get_execution_details(int(data))
            if details:
                code, result_json = details; result = json.loads(result_json)
                error_context = result.get("stderr") or result.get("error") if not result.get("success") else None
                await query.message.edit_reply_markup(reply_markup=None)

                thinking_msg = await query.message.answer("🔍 Debugging with AI\\.\\.\\.")
                prompt = (f"The following code failed. Act as an expert debugger.\n\n**Code:**\n```\n{code}\n```\n\n**Error:**\n```\n{error_context}\n```\n\n"
                          f"Follow this sequence, formatting for Telegram MarkdownV2:\n"
                          f"1\\. **Root Cause**: Pinpoint the exact reason for the failure\\.\n"
                          f"2\\. **Suggested Fix**: Provide the corrected code and explain why it works\\.")
                analysis = await self.llm.generate(prompt)
                await thinking_msg.edit_text(f"*AI Debugger Analysis:*\n\n{analysis}")

        elif action == "demo":
            if data == "build":
                await query.message.answer("Of course\\! Try sending this command:\n\n`/build a python script that prints the current time`")

    # --- Boilerplate for Startup/Shutdown ---
    async def start(self):
        logger.info("🚀 Starting bot...")
        WORKSPACE_PATH.mkdir(exist_ok=True)
        await self.db.init()
        self.llm = LLMManager(GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL)
        self.register_handlers()
        await self.setup_commands()
        await self.bot.delete_webhook(drop_pending_updates=True)
        await self.dp.start_polling(self.bot)

    async def stop(self):
        logger.info("🔌 Shutting down...");
        await self.db.close()
        if hasattr(self.bot, 'session') and self.bot.session and not self.bot.session.closed:
            await self.bot.session.close()
