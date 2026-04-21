import os, io, asyncio, random, json, logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func, desc, text
from sqlalchemy.orm import declarative_base, sessionmaker
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# --- CONFIG ---
logging.basicConfig(level=logging.INFO)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)
user_sessions = {}
processing_lock = asyncio.Lock()

# --- SAFE JSON ---
def safe_json(res):
    if not res:
        return None
    try:
        return json.loads(res)
    except:
        try:
            return json.loads(res[res.find("{"):res.rfind("}")+1])
        except:
            return None

# --- DB ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=30, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Vocab(Base):
    __tablename__ = "vocab"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger, index=True)
    word = Column(String)
    definition = Column(Text)
    category = Column(String)
    source = Column(String)

class User(Base):
    __tablename__ = "users"
    user_id = Column(BigInteger, primary_key=True)

def init_db():
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE vocab ADD COLUMN IF NOT EXISTS user_id BIGINT"))
            conn.commit()
        except:
            pass

# --- AI ---
async def ai_request(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    def call():
        fmt = {"type": "json_object"} if json_mode else None
        try:
            return client.chat.completions.create(
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format=fmt,
                timeout=25
            ).choices[0].message.content
        except:
            return None
    return await loop.run_in_executor(None, call)

# --- VOICE ---
async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    return await loop.run_in_executor(None, create_audio)

# --- CORE ---
async def send_next_step(user_id):
    async with processing_lock:
        sess = user_sessions.get(user_id)
        if not sess:
            return

        db = SessionLocal()
        try:
            topic = sess.get('grammar_topic', 'general')

            sys_prompt = """
Grammar Teacher (B2/C1).

Rules:
- Conditionals: use ALL types (0,1,2,3 + Mixed)
- Passive Voice: include Perfect, Continuous, Modal Passive
- Make questions varied and tricky
"""

            res = await ai_request(
                f"Topic: {topic}. JSON: {{\"q\":\".. ____ ..\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e_en\":\"...\",\"e_ru\":\"...\"}}",
                sys_prompt,
                True
            )

            data = safe_json(res)
            if not data:
                await bot.send_message(user_id, "⚠️ AI error")
                return

            sess['correct_id'] = data['c']
            sess['exp'] = data['e_en']

            await bot.send_poll(
                user_id,
                f"📝 {data['q']}",
                data['o'],
                type='quiz',
                correct_option_id=data['c'],
                is_anonymous=False
            )

        except:
            await bot.send_message(user_id, "⚠️ Error")

        finally:
            db.close()

# --- POLL ---
@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions:
        return

    sess = user_sessions[uid]

    if p.option_ids[0] == sess['correct_id']:
        sess['score'] += 1

    await bot.send_message(uid, f"💡 {sess.get('exp')}")
    sess['step'] += 1
    await send_next_step(uid)

# --- START ---
@dp.message(F.text == "/start")
async def start(m: types.Message):
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],
        [KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("Ready", reply_markup=kb)

# --- VOCAB ---
@dp.message(F.text == "📚 Vocabulary")
async def vocab(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Add", callback_data="voc_add")],
        [InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]
    ])
    await m.answer("Vocabulary:", reply_markup=kb)

# --- DELETE ---
@dp.callback_query(F.data.startswith("list_"))
async def list_words(cb: types.CallbackQuery):
    db = SessionLocal()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).limit(10).all()
    db.close()

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=w.word, callback_data=f"del_{w.id}")]
        for w in words
    ])

    await cb.message.edit_text("Delete word:", reply_markup=kb)

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid = int(cb.data.split('_')[1])

    db = SessionLocal()
    db.query(Vocab).filter(Vocab.id == wid).delete()
    db.commit()
    db.close()

    await cb.answer("Deleted")

    kb = cb.message.reply_markup.inline_keyboard
    new_kb = [row for row in kb if row[0].callback_data != f"del_{wid}"]

    await cb.message.edit_reply_markup(
        reply_markup=InlineKeyboardMarkup(inline_keyboard=new_kb)
    )

# --- ADD WORDS (FIXED) ---
@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]:
        return

    words = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]

    await m.answer(f"⏳ Processing {len(words)} words...")

    db = SessionLocal()
    added = 0

    chunk_size = 10

    for i in range(0, len(words), chunk_size):
        chunk = words[i:i+chunk_size]

        prompt = f"""
Define words:
{chunk}

JSON:
{{"items":[{{"w":"word","d":"definition","c":"word"}}]}}
"""

        res = await ai_request(prompt, "JSON ONLY", True)
        data = safe_json(res)

        if not data:
            for w in chunk:
                db.add(Vocab(user_id=m.from_user.id, word=w, definition="—", category="word"))
                added += 1
            continue

        for item in data.get("items", []):
            db.add(Vocab(
                user_id=m.from_user.id,
                word=item.get("w"),
                definition=item.get("d", ""),
                category=item.get("c", "word").lower().replace(" ", "_")
            ))
            added += 1

    db.commit()
    db.close()

    await m.answer(f"✅ Added {added}/{len(words)} words")

# --- MAIN ---
async def main():
    init_db()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
