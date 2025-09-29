#!/usr/bin/env python3
"""
Telegram Generative AI Coding Agent (Single-File Vercel Edition)

This bot is a resilient, generative AI assistant built on the modern aiogram
framework, architected into a single file for simple, robust deployment on Vercel.
"""

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from aiohttp import web
import asyncpg
from groq import AsyncGroq
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    CallbackQuery,
    Document,
    Message,
    Update,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
import secrets

# --- Load .env for local development ---
load_dotenv()

# --- Configuration ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
_raw_webhook_secret = os.environ.get("WEBHOOK_SECRET")
WEBHOOK_SECRET_GENERATED = False
if _raw_webhook_secret:
    WEBHOOK_SECRET = _raw_webhook_secret.strip()
else:
    WEBHOOK_SECRET = secrets.token_hex(32)
    WEBHOOK_SECRET_GENERATED = True
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
WEBHOOK_PATH = "/api/index"

AUTHORIZED_USERS: Tuple[int, ...] = tuple(
    int(user_id.strip())
    for user_id in os.environ.get("AUTHORIZED_USERS", "").split(",")
    if user_id.strip()
)
ALLOW_ALL_USERS = os.environ.get("ALLOW_ALL_USERS", "").lower() == "true"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
WORKSPACE_PATH = Path("/tmp/workspace")
MAX_CODE_SIZE = 15000
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB upload limit
UPLOADS_PATH = WORKSPACE_PATH / "uploads"
START_TIME = datetime.utcnow()

ALLOWED_MIME_TYPES: Tuple[str, ...] = (
    "text/x-python",
    "text/x-script.python",
    "text/javascript",
    "application/javascript",
    "application/x-javascript",
    "application/x-sh",
    "text/x-sh",
)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s', stream=sys.stdout)
logger = logging.getLogger("vercel-bot")
if WEBHOOK_SECRET_GENERATED:
    logger.warning("WEBHOOK_SECRET not set. Generated ephemeral secret for this runtime.")

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
            logger.warning("DATABASE_URL is not set. Knowledge graph features are disabled.")
            return
        if not self.pool or self.pool.is_closing():
            try:
                self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
                logger.info("Database connection pool created.")
                await self._create_tables()
            except Exception as exc:
                logger.critical(f"Failed to connect to database: {exc}")
                self.pool = None

    async def _create_tables(self):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (id SERIAL PRIMARY KEY, name TEXT UNIQUE NOT NULL, type TEXT);
                CREATE TABLE IF NOT EXISTS executions (id SERIAL PRIMARY KEY, user_id BIGINT NOT NULL, code TEXT, result JSONB, success BOOLEAN);
                CREATE TABLE IF NOT EXISTS observations (id SERIAL PRIMARY KEY, entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE, content TEXT, timestamp TIMESTAMPTZ DEFAULT NOW());
                """
            )
        logger.info("Database tables verified/created.")

    async def log_execution(self, user_id: int, code: str, res: dict, suc: bool) -> int:
        if not self.pool:
            return -1
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "INSERT INTO executions (user_id, code, result, success) VALUES ($1, $2, $3, $4) RETURNING id",
                user_id,
                code,
                json.dumps(res),
                suc,
            )

    async def get_execution_details(self, exec_id: int) -> Optional[Tuple[str, str]]:
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT code, result FROM executions WHERE id = $1", exec_id)
            return (row["code"], json.dumps(row["result"])) if row else None

    async def remember(self, user_id: int, text: str):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            user_entity = f"user_{user_id}"
            await conn.execute(
                "INSERT INTO entities (name, type) VALUES ($1, $2) ON CONFLICT (name) DO NOTHING",
                user_entity,
                "user",
            )
            entity_id = await conn.fetchval("SELECT id FROM entities WHERE name = $1", user_entity)
            await conn.execute(
                "INSERT INTO observations (entity_id, content) VALUES ($1, $2)",
                entity_id,
                text,
            )

    async def recall(self, user_id: int) -> str:
        if not self.pool:
            return "No database configured for memories."
        async with self.pool.acquire() as conn:
            user_entity = f"user_{user_id}"
            rows = await conn.fetch(
                "SELECT o.content FROM observations o JOIN entities e ON e.id = o.entity_id WHERE e.name = $1 ORDER BY o.timestamp DESC LIMIT 10",
                user_entity,
            )
            if not rows:
                return "No recent memories found."
            obs = [row["content"] for row in rows]
            lines = "\n".join(f"- {escape_markdown_v2(o)}" for o in obs)
            return (
                f"Recent memories for *{escape_markdown_v2(user_entity)}*:"
                f"\n{lines}"
            )

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed.")

class SecurityValidator:
    @staticmethod
    def validate_code(code: str) -> Tuple[bool, str]:
        if not code.strip():
            return False, "Empty code"
        if len(code) > MAX_CODE_SIZE:
            return False, f"Code too long (max {MAX_CODE_SIZE} chars)"

        dangerous_patterns = (
            (r"rm\s+-rf\s+/", "Recursive deletion of root"),
            (r":\(\)\s*\{\s*:\|:\s*;\s*\}&", "Fork bomb"),
            (r"dd\s+if=", "Direct disk write"),
            (r"mkfs", "Filesystem formatting"),
            (r"\bshutdown\b", "Shutdown command"),
            (r"curl\s+[^\n]+\|\s*(bash|sh)", "Piping remote script to shell"),
            (r"wget\s+[^\n]+\|\s*(bash|sh)", "Piping remote script to shell"),
        )

        for pattern, reason in dangerous_patterns:
            if re.search(pattern, code, flags=re.IGNORECASE):
                return False, reason

        return True, ""

    @staticmethod
    def validate_file(file: Document) -> Tuple[bool, str]:
        if file.file_size and file.file_size > MAX_FILE_SIZE:
            return False, f"File too large (max {MAX_FILE_SIZE // (1024 * 1024)}MB)"
        if file.mime_type and file.mime_type not in ALLOWED_MIME_TYPES:
            return False, "Unsupported MIME type"
        return True, ""

class CodeExecutor:
    async def execute(self, code: str, lang: str) -> Dict[str, Any]:
        ext, cmd = {"python":(".py",[sys.executable]),"javascript":(".js",["node"]),"bash":(".sh",["bash"])}.get(lang,(None,None))
        if not ext:
            return {"error": f"Unsupported language: {lang}"}
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, dir=WORKSPACE_PATH) as temp:
                temp.write(code)
                temp_file = temp.name
            process = await asyncio.create_subprocess_exec(
                *cmd,
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
            os.unlink(temp_file)
            return {"stdout": stdout.decode(), "stderr": stderr.decode(), "returncode": process.returncode}
        except asyncio.TimeoutError:
            return {"error": "Execution timed out."}
        except Exception as exc:
            return {"error": str(exc)}


def build_execution_reply(result: Dict[str, Any]) -> Tuple[str, bool]:
    if "error" in result:
        message = "❌ *Execution Error*:\n```\n{error}\n```".format(
            error=escape_markdown_v2(result["error"]),
        )
        return message, False

    return_code = result.get("returncode")
    if return_code not in (None, 0):
        stderr = escape_markdown_v2(result.get("stderr") or "") or "No diagnostics"
        message = "❌ *Runtime Error* (exit code {code}):\n```\n{stderr}\n```".format(
            code=return_code,
            stderr=stderr,
        )
        return message, False

    stdout = escape_markdown_v2(result.get("stdout", "").strip() or "No output")
    message = "✅ *Output*:\n```\n{stdout}\n```".format(stdout=stdout)
    return message, True


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


def chunk_text(text: str, limit: int = 4096) -> List[str]:
    if not text:
        return [""]
    segments: List[str] = []
    cursor = text
    while cursor:
        if len(cursor) <= limit:
            segments.append(cursor)
            break
        split_at = cursor.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        segments.append(cursor[:split_at])
        cursor = cursor[split_at:].lstrip()
    return segments


async def send_long_message(chat_id: int, text: str, **kwargs) -> Optional[Message]:
    messages: List[Message] = []
    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks):
        payload = dict(kwargs)
        if idx > 0:
            payload.pop("reply_markup", None)
        await rate_limiter.acquire(chat_id)
        try:
            message = await bot.send_message(chat_id, chunk, **payload)
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            message = await bot.send_message(chat_id, chunk, **payload)
        except TelegramBadRequest as exc:
            logger.error("Failed to send message chunk: %s", exc)
            break
        messages.append(message)
    return messages[0] if messages else None


async def answer_long(message: Message, text: str, **kwargs) -> Optional[Message]:
    return await send_long_message(message.chat.id, text, **kwargs)


def is_user_authorized(user_id: Optional[int]) -> bool:
    if ALLOW_ALL_USERS:
        return True
    if not AUTHORIZED_USERS:
        return False
    return user_id in AUTHORIZED_USERS


async def notify_unauthorized(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else "unknown"
    text = (
        "⛔ *Access Denied*\n\n"
        "This bot is secured with a user whitelist.\n"
        "Your user ID: `{user_id}`\n\n"
        "Ask the administrator to add you to `AUTHORIZED_USERS`."
    ).format(user_id=user_id)
    await answer_markdown(message, text)


async def ensure_authorized(message: Message) -> bool:
    user_id = message.from_user.id if message.from_user else None
    if is_user_authorized(user_id):
        return True
    await notify_unauthorized(message)
    return False


async def send_status_message(message: Message, text: str) -> Message:
    await rate_limiter.acquire(message.chat.id)
    return await message.answer(text)


async def answer_markdown(message: Message, text: str, **kwargs) -> Optional[Message]:
    try:
        return await answer_long(message, text, parse_mode=ParseMode.MARKDOWN_V2, **kwargs)
    except TelegramBadRequest:
        logger.debug("Markdown formatting failed; falling back to escaped output")
        escaped = escape_markdown_v2(text)
        return await answer_long(message, escaped, parse_mode=ParseMode.MARKDOWN_V2, **kwargs)


BOT_COMMANDS: List[BotCommand] = [
    BotCommand(command="start", description="Show help"),
    BotCommand(command="build", description="Generate code with AI"),
    BotCommand(command="python", description="Run Python code"),
    BotCommand(command="node", description="Run Node.js code"),
    BotCommand(command="bash", description="Run Bash commands"),
    BotCommand(command="remember", description="Store a short note"),
    BotCommand(command="recall", description="Recall saved notes"),
]

GROUP_COMMANDS: List[BotCommand] = [
    BotCommand(command="start", description="Show help"),
    BotCommand(command="python", description="Run Python code"),
    BotCommand(command="node", description="Run Node.js code"),
    BotCommand(command="bash", description="Run Bash commands"),
]

# --- Module-level Singleton Initialization ---
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is required for the bot to start.")

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
    if not is_user_authorized(message.from_user.id if message.from_user else None):
        await notify_unauthorized(message)
        return

    kb = InlineKeyboardBuilder()
    kb.button(text="🚀 Try a Build Demo", callback_data="demo:build")
    kb.button(text="🧠 Ask AI", callback_data="demo:ai")
    kb.adjust(1)
    text = (
        "👋 *Welcome to your AI Coding Agent*!\n\n"
        "Send /python, /node, or /bash commands to execute code, or /ai to chat with the assistant.\n"
        "Upload .py, .js, or .sh files to run them securely."
    )
    await answer_markdown(message, text, reply_markup=kb.as_markup())

@router.message(Command("build"))
async def cmd_build(message: Message):
    if not await ensure_authorized(message):
        return
    prompt_text = message.text.replace("/build", "", 1).strip()
    if not prompt_text:
        await answer_markdown(
            message,
            "Please provide a prompt after `/build`.",
        )
        return
    thinking_msg = await send_status_message(message, "🧠 Thinking and building...")
    prompt = f"You are an expert AI developer. Generate a response in MDX-like format for Telegram's MarkdownV2.\n\n**User Request:** \"{prompt_text}\""
    response = await llm_manager.generate(prompt)
    await answer_markdown(message, response)
    try:
        await thinking_msg.edit_text("✅ Build ready", parse_mode=ParseMode.MARKDOWN_V2)
    except TelegramBadRequest:
        await thinking_msg.delete()

@router.message(Command("python", "p", "node", "n", "bash", "b"))
async def cmd_execute(message: Message):
    if not await ensure_authorized(message):
        return
    cmd_map = {"p": "python", "n": "javascript", "b": "bash", "node": "javascript"}
    command, *code_parts = message.text.split(maxsplit=1)
    lang_key = command[1:]
    lang = cmd_map.get(lang_key, lang_key)
    code = code_parts[0] if code_parts else ""
    if not code:
        await answer_markdown(
            message,
            f"Usage: `/{lang_key} <code>`",
        )
        return
    valid, err = security_validator.validate_code(code)
    if not valid:
        await answer_markdown(
            message,
            f"❌ *Invalid Code*: {escape_markdown_v2(err)}",
        )
        return
    status_msg = await send_status_message(message, f"🚀 Executing {lang}...")
    result = await executor.execute(code, lang)
    response, success = build_execution_reply(result)
    exec_id = await db_manager.log_execution(message.from_user.id, code, result, success)
    kb = InlineKeyboardBuilder()
    if exec_id != -1:
        kb.button(text="🔍 Debug with AI", callback_data=f"analyze:{exec_id}")
        kb.adjust(1)
        markup = kb.as_markup()
    else:
        markup = None
    await answer_markdown(
        message,
        response,
        reply_markup=markup,
    )
    await status_msg.edit_text(
        "✅ Execution finished" if success else "⚠️ Execution finished with issues",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

@router.message(Command("remember", "recall"))
async def cmd_memory(message: Message):
    if not await ensure_authorized(message):
        return
    command, *args = message.text.split(maxsplit=1)
    action = command[1:]
    if action == "remember":
        text_to_remember = args[0] if args else ""
        if not text_to_remember:
            await answer_markdown(
                message,
                "Usage: `/remember <text to save>`",
            )
            return
        await db_manager.remember(message.from_user.id, text_to_remember)
        await answer_markdown(message, "✅ Memory updated.")
    elif action == "recall":
        recall_text = await db_manager.recall(message.from_user.id)
        await answer_markdown(message, recall_text)

@router.message(F.document)
async def handle_document_upload(message: Message):
    if not await ensure_authorized(message):
        return

    if not message.document:
        return

    document = message.document
    valid_file, reason = security_validator.validate_file(document)
    if not valid_file:
        await answer_markdown(message, f"❌ *Upload rejected*: {escape_markdown_v2(reason)}")
        return

    extension = (Path(document.file_name or "").suffix or "").lower()
    language_map = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".sh": "bash",
        ".bash": "bash",
    }
    language = language_map.get(extension)
    if not language:
        await answer_markdown(
            message,
            "❌ *Unsupported file type*. Please upload .py, .js, or .sh files.",
        )
        return

    status_msg = await send_status_message(message, "📥 Downloading file...")
    UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
    filename = document.file_name or f"uploaded{extension or '.txt'}"
    destination = UPLOADS_PATH / f"{document.file_unique_id}_{filename}"

    try:
        await bot.download(document, destination=destination)
        code = destination.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Failed to download file: %s", exc)
        await status_msg.edit_text("❌ Download failed", parse_mode=ParseMode.MARKDOWN_V2)
        return

    valid_code, reason = security_validator.validate_code(code)
    if not valid_code:
        await status_msg.edit_text("❌ Invalid code", parse_mode=ParseMode.MARKDOWN_V2)
        await answer_markdown(message, f"❌ *Invalid Code*: {escape_markdown_v2(reason)}")
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        return

    await status_msg.edit_text(f"🚀 Executing {language} code...", parse_mode=ParseMode.MARKDOWN_V2)
    result = await executor.execute(code, language)
    response, success = build_execution_reply(result)
    exec_id = await db_manager.log_execution(message.from_user.id, code, result, success)

    kb = InlineKeyboardBuilder()
    if exec_id != -1:
        kb.button(text="🔍 Debug with AI", callback_data=f"analyze:{exec_id}")
        kb.adjust(1)
        markup = kb.as_markup()
    else:
        markup = None

    await answer_markdown(message, response, reply_markup=markup)
    await status_msg.edit_text(
        "✅ Execution finished" if success else "⚠️ Execution finished with issues",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    try:
        destination.unlink()
    except FileNotFoundError:
        pass

@router.message(F.text)
async def handle_text_conversation(message: Message):
    if not await ensure_authorized(message):
        return
    thinking_msg = await send_status_message(message, "🤖 Thinking...")
    user_memory = await db_manager.recall(message.from_user.id)
    prompt = (
        "You are a helpful AI assistant. Use the user's memory for a personalized response.\n"
        f"**Memory Context**:\n{user_memory}\n\n**User Message**:\n{message.text}"
    )
    response = await llm_manager.generate(prompt)
    await answer_markdown(message, response)
    try:
        await thinking_msg.edit_text("✅ Ready", parse_mode=ParseMode.MARKDOWN_V2)
    except TelegramBadRequest:
        await thinking_msg.delete()

@router.callback_query()
async def handle_callback(query: CallbackQuery):
    if not is_user_authorized(query.from_user.id if query.from_user else None):
        await query.answer("Access denied", show_alert=True)
        return

    if not query.message:
        await query.answer("Unsupported context", show_alert=True)
        return

    if not query.data:
        await query.answer()
        return

    action, _, data = query.data.partition(":")

    if action == "analyze":
        details = await db_manager.get_execution_details(int(data))
        if not details:
            await query.answer("Execution record not found", show_alert=True)
            return
        await query.answer("Analyzing...", show_alert=False)
        code, result_json = details
        result = json.loads(result_json)
        error_context = result.get("stderr") or result.get("error") or "No diagnostics"
        await query.message.edit_reply_markup(reply_markup=None)
        thinking_msg = await send_status_message(query.message, "🔍 Debugging with AI...")
        prompt = (
            "Act as an expert debugger for the failed code.\n\n"
            f"**Code:**\n```\n{code}\n```\n\n"
            f"**Error:**\n```\n{error_context}\n```\n\n"
            "Provide a root cause analysis and a suggested fix."
        )
        analysis = await llm_manager.generate(prompt)
        await answer_markdown(query.message, f"*AI Debugger Analysis:*\n\n{analysis}")
        try:
            await thinking_msg.delete()
        except TelegramBadRequest:
            await thinking_msg.edit_text("✅ Analysis ready", parse_mode=ParseMode.MARKDOWN_V2)
    elif action == "demo":
        await query.answer()
        if data == "ai":
            await answer_markdown(query.message, "Try asking: `How do I read a JSON file in Python?`")
        else:
            await answer_markdown(query.message, "Try this: `/build a python script that prints the current time`")


async def configure_bot_commands() -> None:
    try:
        await bot.set_my_commands(BOT_COMMANDS, scope=BotCommandScopeAllPrivateChats())
        await bot.set_my_commands(GROUP_COMMANDS, scope=BotCommandScopeAllGroupChats())
    except TelegramBadRequest as exc:
        logger.error("Failed to configure bot commands: %s", exc)


async def configure_webhook() -> None:
    if not WEBHOOK_URL:
        logger.info("WEBHOOK_URL not provided. Ensure Telegram webhook is set via BotFather.")
        return
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.set_webhook(
            url=WEBHOOK_URL,
            secret_token=WEBHOOK_SECRET,
            allowed_updates=["message", "callback_query", "edited_message"],
        )
        logger.info("Webhook configured at %s", WEBHOOK_URL)
    except TelegramBadRequest as exc:
        logger.error("Failed to set webhook: %s", exc)

# --- Vercel ASGI Application ---
app = web.Application()

async def webhook_handler(request: web.Request) -> web.Response:
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        logger.warning("Invalid secret token."); return web.Response(status=403)
    try:
        update = Update.model_validate(await request.json(), context={"bot": bot})
        await dp.feed_update(bot=bot, update=update)
        return web.Response(status=200)
    except Exception as e:
        logger.error(f"Error processing update: {e}", exc_info=True)
        return web.Response(status=500)


async def health_handler(_: web.Request) -> web.Response:
    """Expose minimal health information for Vercel monitoring."""
    return web.json_response(
        {
            "status": "ok",
            "uptime_seconds": int((datetime.utcnow() - START_TIME).total_seconds()),
            "webhook_path": WEBHOOK_PATH,
            "authorized_users": list(AUTHORIZED_USERS) if AUTHORIZED_USERS else "*",
        }
    )

async def on_startup(app_instance: web.Application):
    logger.info("Initializing application resources...")
    WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
    await db_manager.init()
    if AUTHORIZED_USERS:
        logger.info("Authorized users: %s", ", ".join(str(uid) for uid in AUTHORIZED_USERS))
    elif ALLOW_ALL_USERS:
        logger.warning("ALLOW_ALL_USERS enabled. All Telegram users can access the bot.")
    else:
        logger.warning("No authorized users configured. All requests will be rejected until AUTHORIZED_USERS is set.")
    await configure_bot_commands()
    await configure_webhook()
    logger.info("Application resources initialized.")

async def on_shutdown(app_instance: web.Application):
    logger.info("Closing application resources...")
    await db_manager.close()
    if bot.session and not bot.session.closed:
        await bot.session.close()
    logger.info("Application resources closed.")

app.router.add_post(WEBHOOK_PATH, webhook_handler)
app.router.add_get("/api/health", health_handler)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)
