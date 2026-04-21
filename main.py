import os, io, asyncio, random, json, logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiogram.enums import ParseMode
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func, desc
from sqlalchemy.orm import declarative_base, sessionmaker

# --- CONFIG ---
logging.basicConfig(level=logging.INFO)
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")

# --- DB ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

class Vocab(Base):
    __tablename__ = "vocab"
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger)
    word = Column(String)
    definition = Column(Text)

Base.metadata.create_all(bind=engine)

# --- MEMORY ---
user_sessions = {}

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

# --- AI ---
async def ai_request(prompt, system_msg):
    loop = asyncio.get_event_loop()

    def call():
        for _ in range(2):
            try:
                return client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    timeout=20
                ).choices[0].message.content
            except:
                continue
        return None

    return await loop.run_in_executor(None, call)

# --- VOICE ---
async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create():
        tts = gTTS(text=text, lang='en')
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    return await loop.run_in_executor(None, create)

# --- START ---
@dp.message(F.text == "/start")
async def start(m: types.Message):
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar")],
        [KeyboardButton(text="🎤 Speaking"), KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("Ready", reply_markup=kb)

# --- VOCAB ADD ---
@dp.message(F.text)
async def add_words(m: types.Message):
    if m.text.startswith("/") or m.text in ["📚 Vocabulary","⚙️ Grammar","🎤 Speaking","📊 My Progress"]:
        return

    words = [w.strip() for w in m.text.replace(",", "\n").split("\n") if w.strip()]
    db = SessionLocal()
    added = 0

    for w in words:
        exists = db.query(Vocab).filter(Vocab.user_id == m.from_user.id, Vocab.word == w).first()
        if not exists:
            db.add(Vocab(user_id=m.from_user.id, word=w))
            added += 1

    db.commit()
    db.close()

    await m.answer(f"✅ Added {added} words")

# --- VOCAB TEST ---
@dp.message(F.text == "📚 Vocabulary")
async def vocab(m: types.Message):
    user_sessions[m.from_user.id] = {"type":"vocab","step":0,"score":0}
    await next_q(m.from_user.id)

# --- GRAMMAR ---
@dp.message(F.text == "⚙️ Grammar")
async def grammar(m: types.Message):
    user_sessions[m.from_user.id] = {"type":"grammar","step":0,"score":0}
    await next_q(m.from_user.id)

# --- SPEAKING ---
@dp.message(F.text == "🎤 Speaking")
async def speaking(m: types.Message):
    res = await ai_request("5 short topics JSON {topics:[]}", "JSON only")
    data = safe_json(res)
    topics = data["topics"] if data else ["Travel","Work","Life"]

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=t, callback_data=f"spk_{t}")]
        for t in topics
    ])
    await m.answer("Choose topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("spk_"))
async def spk_start(cb: types.CallbackQuery):
    topic = cb.data[4:]
    q = await ai_request(f"Ask 1 short question about {topic}", "Teacher max 1 sentence")

    user_sessions[cb.from_user.id] = {"type":"speaking","history":[q]}

    await cb.message.answer(q)
    v = await generate_voice(q)
    await bot.send_voice(cb.message.chat.id, BufferedInputFile(v.read(),"q.ogg"))

# --- VOICE ---
@dp.message(F.voice)
async def voice(m: types.Message):
    sess = user_sessions.get(m.from_user.id)
    if not sess or sess["type"] != "speaking":
        return

    file = await bot.get_file(m.voice.file_id)
    content = await bot.download_file(file.file_path)

    text = client.audio.transcriptions.create(
        file=("v.ogg", content.read()),
        model="whisper-large-v3"
    ).text

    resp = await ai_request(f"User: {text}. Reply short + ask question", "Teacher")

    await m.answer(resp)
    v = await generate_voice(resp)
    await bot.send_voice(m.chat.id, BufferedInputFile(v.read(),"r.ogg"))

# --- QUIZ ---
async def next_q(uid):
    sess = user_sessions.get(uid)
    if not sess:
        return

    if sess["type"] == "vocab":
        db = SessionLocal()
        word = db.query(Vocab).filter(Vocab.user_id==uid).order_by(func.random()).first()
        db.close()

        if not word:
            await bot.send_message(uid,"No words")
            return

        res = await ai_request(
            f"Word {word.word}. JSON {{q,options[4],answer}}",
            "Teacher JSON only"
        )

        data = safe_json(res)
        if not data:
            await bot.send_message(uid,"AI error")
            return

        sess["correct"] = data["answer"]

        await bot.send_poll(uid,data["q"],data["options"],type="quiz",
                            correct_option_id=data["answer"],
                            is_anonymous=False)

    else:
        res = await ai_request(
            "Grammar question JSON {q,options[4],answer}",
            "English grammar B2 JSON"
        )
        data = safe_json(res)
        if not data:
            await bot.send_message(uid,"AI error")
            return

        sess["correct"] = data["answer"]

        await bot.send_poll(uid,data["q"],data["options"],type="quiz",
                            correct_option_id=data["answer"],
                            is_anonymous=False)

# --- POLL ---
@dp.poll_answer()
async def poll(p: PollAnswer):
    uid = p.user.id
    sess = user_sessions.get(uid)
    if not sess:
        return

    if p.option_ids[0] == sess["correct"]:
        sess["score"] += 1

    sess["step"] += 1
    await next_q(uid)

# --- WEB ---
async def web_server():
    app = web.Application()
    app.router.add_get("/", lambda r: web.Response(text="OK"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner,"0.0.0.0",int(os.getenv("PORT",10000))).start()

# --- MAIN ---
async def main():
    asyncio.create_task(web_server())
    await bot.delete_webhook(drop_pending_updates=True)
    await asyncio.sleep(1)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
