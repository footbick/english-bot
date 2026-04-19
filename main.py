import os
import io
import asyncio
import random
import json
import logging

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq, RateLimitError
from gtts import gTTS

# SQLAlchemy для PostgreSQL
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func
from sqlalchemy.orm import declarative_base, sessionmaker

# =========================
# CONFIG & LOGS
# =========================
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)

user_sessions = {}

TOPIC_PROMPTS = {
    'pdf': "grammar patterns found in the provided context.",
    'passive': "Passive Voice structures (all tenses).",
    'conditionals': "Conditionals (Zero, 1st, 2nd, 3rd, Mixed).",
    'complex': "Complex Object (e.g., 'I want him to stay', 'I saw her crossing the street').",
    'participle': "Gerund and Participle (Participle I, Participle II, Perfect Participle).",
    'prepositions': "Prepositions (dependent prepositions, prepositions of time/place).",
    'general': "Mixed B2 English Grammar (General Mixed Practice)."
}

# =========================
# DATABASE (Supabase)
# =========================
Base = declarative_base()
# Настройка endpoint для Supabase через коннект
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "options": "-c endpoint=qmytswodftgctaojwiia",
        "sslmode": "require"
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    user_id = Column(BigInteger, primary_key=True)

class Vocab(Base):
    __tablename__ = "vocab"
    id = Column(Integer, primary_key=True, index=True)
    word = Column(String)
    definition = Column(Text)
    category = Column(String)
    source = Column(String)

Base.metadata.create_all(bind=engine)

# =========================
# RENDER WEB SERVER
# =========================
async def handle(request):
    return web.Response(text="English Coach is running!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

# =========================
# NOTIFICATIONS
# =========================
async def reminder_loop():
    while True:
        await asyncio.sleep(21600) # каждые 6 часов
        db = SessionLocal()
        users = db.query(User).all()
        for user in users:
            try:
                await bot.send_message(user.user_id, "⏰ English Practice Time!", parse_mode="Markdown")
            except:
                pass
        db.close()

# =========================
# AI TOOLS
# =========================
async def ai_request(prompt, system_msg, json_mode=False, user_id=None):
    loop = asyncio.get_event_loop()
    def call():
        fmt = {"type": "json_object"} if json_mode else None
        return client.chat.completions.create(
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", response_format=fmt
        ).choices[0].message.content
    try:
        return await loop.run_in_executor(None, call)
    except RateLimitError:
        if user_id: await bot.send_message(user_id, "⚠️ Groq Limit Reached.")
        return None
    except Exception as e:
        logging.error(f"AI Error: {e}")
        return None

async def generate_voice(text, lang='en'):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    return await loop.run_in_executor(None, create_audio)

# =========================
# CORE ENGINE
# =========================
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return

    if sess.get('is_exam') and sess['step'] >= 10:
        mistakes = ", ".join(sess.get('mistakes_words', [])) or "None"
        analysis = await ai_request(f"Mistakes: {mistakes}. Give tips in Russian.", "Teacher.", user_id=user_id)
        await bot.send_message(user_id, f"🏆 Exam Finished!\nScore: {sess['score']}/10\n\n{analysis}")
        user_sessions.pop(user_id, None)
        return

    q_type = random.choice(['vocab', 'grammar']) if sess.get('is_exam') else sess['type']
    db = SessionLocal()

    try:
        if q_type == 'vocab':
            exclude = sess.get('used_items', [])
            target = db.query(Vocab).filter(~Vocab.word.in_(exclude)).order_by(func.random()).first()
            if not target:
                q_type = 'grammar'
            else:
                dist = db.query(Vocab).filter(Vocab.id != target.id).order_by(func.random()).limit(3).all()
                options = list(set([target.definition] + [d.definition for d in dist]))
                random.shuffle(options)
                
                e_en = await ai_request(f"Explain word '{target.word}' for B2.", "Teacher.", user_id=user_id)
                e_ru = await ai_request(f"Translate to Russian: {e_en}", "Translator.", user_id=user_id)
                
                sess.setdefault('used_items', []).append(target.word)
                sess.update({'correct_id': options.index(target.definition), 'explanation': e_en, 'explanation_ru': e_ru, 'current_word': target.word})
                await bot.send_poll(user_id, f"Choose definition for: {target.word}", options[:4], type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # Grammar
        topic = TOPIC_PROMPTS.get(sess.get('grammar_topic', 'general'))
        history = " | ".join(sess.get('history', [])[-15:])
        prompt = f"Topic: {topic}. UNIQUE (not: {history}). JSON: {{\"q\":\"... with ____\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e_en\":\"...\",\"e_ru\":\"...\"}}"
        
        res = await ai_request(prompt, "Teacher. JSON ONLY.", json_mode=True, user_id=user_id)
        data = json.loads(res)
        sess.setdefault('history', []).append(data['q'])
        sess.update({'correct_id': data['c'], 'explanation': data['e_en'], 'explanation_ru': data['e_ru'], 'current_word': 'Grammar'})
        await bot.send_poll(user_id, data['q'], data['o'][:4], type='quiz', correct_option_id=data['c'], is_anonymous=False)
    finally:
        db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    else: sess.setdefault('mistakes_words', []).append(sess.get('current_word'))

    msg = f"💡 <b>Explanation:</b>\n{sess['explanation']}\n\n🇷🇺 <b>Перевод:</b> <tg-spoiler>{sess['explanation_ru']}</tg-spoiler>"
    await bot.send_message(uid, msg, parse_mode="HTML")
    sess['step'] += 1
    await asyncio.sleep(0.5); await send_next_step(uid)

# =========================
# HANDLERS (PDF, VOICE, ETC)
# =========================
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    db = SessionLocal()
    if not db.query(User).filter(User.user_id == m.from_user.id).first():
        db.add(User(user_id=m.from_user.id))
        db.commit()
    db.close()
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],
        [KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("🎯 English Coach Cloud Active!", reply_markup=kb)

@dp.message(F.document.mime_type == "application/pdf")
async def handle_pdf(m: types.Message):
    st = await m.answer("⏳ Analyzing PDF...")
    try:
        file = await bot.get_file(m.document.file_id)
        content = await bot.download_file(file.file_path)
        reader = PdfReader(io.BytesIO(content.read()))
        text = "".join([p.extract_text() for p in reader.pages[:3]])
        
        res = await ai_request(f"Extract 5 B2 terms from: {text[:3000]}", "JSON: {\"items\":[{\"w\":\"word\",\"d\":\"def\"}]}", json_mode=True, user_id=m.from_user.id)
        data = json.loads(res)
        
        db = SessionLocal()
        for i in data.get('items', []):
            db.add(Vocab(word=i['w'], definition=i['d'], category='pdf', source=m.document.file_name))
        db.commit(); db.close()
        await st.edit_text(f"✅ Added {len(data.get('items', []))} terms to Supabase.")
    except: await st.edit_text("❌ PDF error.")

@dp.message(F.voice)
async def handle_voice(m: types.Message):
    st = await m.answer("👂 Listening...")
    file = await bot.get_file(m.voice.file_id)
    content = await bot.download_file(file.file_path)
    trans = client.audio.transcriptions.create(file=("v.ogg", content.read()), model="whisper-large-v3", language="en").text
    f = await ai_request(f"Analyze: '{trans}'", "Teacher.", user_id=m.from_user.id)
    await st.edit_text(f"💬 <b>You:</b> {trans}\n\n<b>Feedback:</b> {f}", parse_mode="HTML")

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Mixed Practice", callback_data="gt_general")]])
    await m.answer("Choose topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    user_sessions[cb.from_user.id] = {'type':'grammar', 'score':0, 'step':0, 'grammar_topic': 'general', 'history': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.text == "📚 Vocabulary")
async def v_start(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'vocab', 'score':0, 'step':0, 'used_items': []}
    await send_next_step(m.from_user.id)

# =========================
# MAIN
# =========================
async def main():
    asyncio.create_task(start_web_server())
    asyncio.create_task(reminder_loop())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
