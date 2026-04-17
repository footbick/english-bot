import os
import io
import asyncio
import sqlite3
import random
import json
import logging

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, PollAnswer
from groq import Groq

# =========================
# LOGS
# =========================
logging.basicConfig(level=logging.INFO)

# =========================
# ENV
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)

user_sessions = {}

# =========================
# DB
# =========================
def init_db():
    conn = sqlite3.connect("english_coach.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS vocab (
        id INTEGER PRIMARY KEY,
        word TEXT,
        definition TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

# =========================
# AI QUEUE
# =========================
ai_queue = asyncio.Queue()
ai_semaphore = asyncio.Semaphore(2)


async def _ai_call(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()

    def call():
        fmt = {"type": "json_object"} if json_mode else None

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt[:3500]}
                    ],
                    response_format=fmt
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[GROQ ERROR] attempt {attempt}: {e}")

        return None

    async with ai_semaphore:
        return await loop.run_in_executor(None, call)


async def ai_worker():
    while True:
        job = await ai_queue.get()
        try:
            res = await _ai_call(job["prompt"], job["system"], job["json_mode"])
            job["future"].set_result(res)
        except Exception as e:
            print("[AI WORKER ERROR]", e)
            job["future"].set_result(None)
        finally:
            ai_queue.task_done()


async def ai_request(prompt, system_msg, json_mode=False):
    future = asyncio.get_event_loop().create_future()

    await ai_queue.put({
        "prompt": prompt,
        "system": system_msg,
        "json_mode": json_mode,
        "future": future
    })

    try:
        return await asyncio.wait_for(future, timeout=25)
    except asyncio.TimeoutError:
        return None


# =========================
# SAFE JSON
# =========================
def safe_json(text):
    try:
        return json.loads(text)
    except:
        import re
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
        return None


# =========================
# LOGIC
# =========================
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess:
        return

    if sess.get("step", 0) >= 10:
        await bot.send_message(user_id, f"🏁 Done! Score: {sess.get('score', 0)}/10")
        user_sessions.pop(user_id, None)
        return

    await grammar_q(user_id, sess)


async def grammar_q(user_id, sess):
    raw = await ai_request(
        "Create English grammar test (4 options). JSON: q, o, c",
        "Return JSON only",
        json_mode=True
    )

    data = safe_json(raw)
    if not data:
        await bot.send_message(user_id, "AI error, retrying...")
        return

    sess["correct"] = data["c"]

    await bot.send_poll(
        user_id,
        data["q"],
        data["o"][:4],
        type="quiz",
        correct_option_id=data["c"],
        is_anonymous=False
    )


# =========================
# HANDLERS
# =========================
@dp.message(F.text == "/start")
async def start(m: types.Message):
    kb = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Start test")]],
        resize_keyboard=True
    )
    await m.answer("Ready 🚀", reply_markup=kb)


@dp.message(F.text == "Start test")
async def start_test(m: types.Message):
    user_sessions[m.from_user.id] = {"step": 0, "score": 0}
    await send_next_step(m.from_user.id)


@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions:
        return

    sess = user_sessions[uid]

    if p.option_ids and p.option_ids[0] == sess.get("correct"):
        sess["score"] += 1

    sess["step"] += 1
    await send_next_step(uid)


# =========================
# MAIN
# =========================
async def main():
    asyncio.create_task(ai_worker())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())