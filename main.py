import os, io, asyncio, random, json, logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import *
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func
from sqlalchemy.orm import declarative_base, sessionmaker

# --- CONFIG ---
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not TELEGRAM_TOKEN:
    raise ValueError("No TELEGRAM_TOKEN")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)
user_sessions = {}

# --- DB ---
Base = declarative_base()
engine = None
SessionLocal = None

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine)
        logging.info("DB connected")
    except Exception as e:
        logging.error(e)

class Vocab(Base):
    __tablename__ = "vocab"
    id = Column(Integer, primary_key=True)
    word = Column(String)
    definition = Column(Text)
    category = Column(String)

if engine:
    Base.metadata.create_all(bind=engine)

# --- WEB ---
async def handle(request): return web.Response(text="OK")

async def start_web():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

# --- AI ---
async def ai_request(prompt, system="Teacher", json_mode=False):
    try:
        def call():
            fmt = {"type": "json_object"} if json_mode else None
            return client.chat.completions.create(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format=fmt
            ).choices[0].message.content
        return await asyncio.to_thread(call)
    except Exception as e:
        logging.error(e)
        return None

# --- VOICE ---
async def voice(text):
    def make():
        t = gTTS(text=text, lang="en")
        b = io.BytesIO()
        t.write_to_fp(b)
        b.seek(0)
        return b
    return await asyncio.to_thread(make)

# --- START ---
@dp.message(F.text == "/start")
async def start(m: types.Message):
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar")],
        [KeyboardButton(text="🎤 Speaking"), KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("Ready", reply_markup=kb)

# --- VOCAB MENU ---
@dp.message(F.text == "📚 Vocabulary")
async def vocab_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Words", callback_data="voc_word"),
         InlineKeyboardButton(text="Phrases", callback_data="voc_phrase")],
        [InlineKeyboardButton(text="Idioms", callback_data="voc_idiom")],
        [InlineKeyboardButton(text="Delete Words", callback_data="list")]
    ])
    await m.answer("Choose:", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def vocab_start(cb: CallbackQuery):
    cat = cb.data.replace("voc_", "")
    user_sessions[cb.from_user.id] = {"cat": cat}
    await send_word(cb.from_user.id)

async def send_word(uid):
    if not SessionLocal: return
    db = SessionLocal()
    sess = user_sessions[uid]
    word = db.query(Vocab).filter(Vocab.category == sess["cat"]).order_by(func.random()).first()
    db.close()
    if not word:
        await bot.send_message(uid, "No words")
        return

    prompt = f"""
Word: {word.word}
Definition: {word.definition}

Give:
- definition
- 3 similar wrong options

JSON:
{{"q":"...","o":["correct","w1","w2","w3"],"e":"explain"}}
"""
    res = await ai_request(prompt, json_mode=True)
    if not res:
        await bot.send_message(uid, "AI error")
        return

    data = json.loads(res)
    opts = data["o"]
    random.shuffle(opts)

    await bot.send_poll(uid, data["q"], opts,
                        type="quiz",
                        correct_option_id=opts.index(word.word))

# --- DELETE ---
@dp.callback_query(F.data == "list")
async def list_words(cb: CallbackQuery):
    db = SessionLocal()
    words = db.query(Vocab).limit(10).all()
    db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=w.word, callback_data=f"del_{w.id}")]
        for w in words
    ])
    await cb.message.answer("Tap to delete", reply_markup=kb)

@dp.callback_query(F.data.startswith("del_"))
async def delete(cb: CallbackQuery):
    wid = int(cb.data.split("_")[1])
    db = SessionLocal()
    db.query(Vocab).filter(Vocab.id == wid).delete()
    db.commit()
    db.close()
    await cb.message.edit_text("Deleted")

# --- PDF ---
@dp.message(F.document)
async def pdf(m: types.Message):
    file = await bot.get_file(m.document.file_id)
    f = await bot.download_file(file.file_path)
    reader = PdfReader(f)
    text = "".join([p.extract_text() or "" for p in reader.pages[:5]])
    words = list(set(text.split()))[:20]

    db = SessionLocal()
    for w in words:
        db.add(Vocab(word=w, definition="from pdf", category="word"))
    db.commit()
    db.close()

    await m.answer("PDF processed")

# --- GRAMMAR ---
@dp.message(F.text == "⚙️ Grammar")
async def grammar(m: types.Message):
    prompt = """
Create B2 grammar question.
English only.
JSON:
{"q":"...","o":["a","b","c","d"],"c":0,"e":"explanation"}
"""
    res = await ai_request(prompt, json_mode=True)
    data = json.loads(res)

    await m.answer_poll(data["q"], data["o"],
                        type="quiz",
                        correct_option_id=data["c"])

    await m.answer(data["e"])

# --- SPEAKING ---
@dp.message(F.text == "🎤 Speaking")
async def speaking(m: types.Message):
    res = await ai_request("Give 1 short speaking question", "Examiner")
    v = await voice(res)
    await m.answer(res)
    await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))

@dp.message(F.voice)
async def voice_answer(m: types.Message):
    await m.answer("Got your answer 👍")

# --- PROGRESS ---
@dp.message(F.text == "📊 My Progress")
async def progress(m: types.Message):
    if not SessionLocal:
        await m.answer("No DB")
        return
    db = SessionLocal()
    count = db.query(Vocab).count()
    db.close()
    await m.answer(f"Words in DB: {count}")

# --- MAIN ---
async def main():
    await start_web()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
