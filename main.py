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

# Библиотеки для работы с PostgreSQL
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

# =========================
# LOGS & CONFIG
# =========================
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL") # Твоя ссылка от Supabase

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)

user_sessions = {}

# =========================
# DATABASE SETUP (SQLAlchemy)
# =========================
Base = declarative_base()
# Замени старую строку engine = create_engine... на это:
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

# Создаем таблицы в Supabase, если их еще нет
Base.metadata.create_all(bind=engine)

# =========================
# WEB SERVER (FOR RENDER)
# =========================
async def handle(request):
    return web.Response(text="Bot is running and connected to Supabase!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    logging.info(f"Web server started on port {port}")
    await site.start()

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
        if user_id:
            try: await bot.send_message(user_id, "⚠️ Groq Limit Reached.")
            except: pass
        return None
    except Exception as e:
        logging.error(f"AI Error: {e}")
        return None

async def generate_voice(text, lang='en'):
    loop = asyncio.get_event_loop()
    def create_audio():
        clean = text.replace('*', '').replace('_', '')
        tts = gTTS(text=clean, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    return await loop.run_in_executor(None, create_audio)

# =========================
# CORE ENGINE
# =========================
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return

    if sess.get('is_exam') and sess['step'] >= 10:
        mistakes = ", ".join(sess.get('mistakes_words', [])) if sess.get('mistakes_words') else "None"
        analysis = await ai_request(f"Mistakes: {mistakes}. Give tips in Russian.", "Teacher.", user_id=user_id)
        if analysis:
            await bot.send_message(user_id, f"🏆 Exam Finished!\nScore: {sess['score']}/10\n\n{analysis}")
        user_sessions.pop(user_id, None)
        return

    q_type = random.choice(['vocab', 'grammar']) if sess.get('is_exam') else sess['type']
    q_num_header = f"Question {sess['step'] + 1}/10\n" if sess.get('is_exam') else ""

    db = SessionLocal()
    try:
        if q_type == 'vocab':
            exclude = sess.get('used_items', [])
            target = db.query(Vocab).filter(~Vocab.word.in_(exclude)).order_by(func.random()).first()
            
            if not target:
                q_type = 'grammar'
            else:
                dist = db.query(Vocab).filter(Vocab.definition != target.definition).order_by(func.random()).limit(3).all()
                options = list(set([target.definition] + [d.definition for d in dist]))
                random.shuffle(options)

                e_en = await ai_request(f"Explain word '{target.word}' for B2.", "Teacher.", user_id=user_id)
                e_ru = await ai_request(f"Translate to Russian: {e_en}", "Translator.", user_id=user_id)

                if not e_en or not e_ru: return

                sess.setdefault('used_items', []).append(target.word)
                sess.update({'correct_id': options.index(target.definition), 'current_word': target.word, 'explanation': e_en, 'explanation_ru': e_ru})
                await bot.send_poll(user_id, f"{q_num_header}Task: Choose definition\nWord: '{target.word}'", options[:4], type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # Grammar logic
        prompt = "Create B2 grammar sentence with ____. JSON: {\"q\":\"...\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e_en\":\"...\",\"e_ru\":\"...\"}"
        res_raw = await ai_request(prompt, "Teacher. JSON ONLY.", json_mode=True, user_id=user_id)
        data = json.loads(res_raw)
        sess.update({'correct_id': data['c'], 'explanation': data['e_en'], 'explanation_ru': data['e_ru'], 'current_word': 'Grammar'})
        await bot.send_poll(user_id, f"{q_num_header}{data['q']}", data['o'][:4], type='quiz', correct_option_id=data['c'], is_anonymous=False)
    
    except Exception as e:
        logging.error(f"Step error: {e}")
        await send_next_step(user_id)
    finally:
        db.close()

@dp.poll_answer()
async def poll_ans(poll_answer: PollAnswer):
    uid = poll_answer.user.id
    if uid in user_sessions:
        sess = user_sessions[uid]
        if poll_answer.option_ids[0] == sess['correct_id']: 
            sess['score'] += 1
        else: 
            sess.setdefault('mistakes_words', []).append(sess.get('current_word'))
        
        msg = f"💡 <b>Explanation:</b>\n{sess['explanation']}\n\n🇷🇺 <b>Перевод:</b> <tg-spoiler>{sess['explanation_ru']}</tg-spoiler>"
        try: await bot.send_message(uid, msg, parse_mode="HTML")
        except: pass
        
        sess['step'] += 1
        await asyncio.sleep(0.5)
        await send_next_step(uid)

# =========================
# HANDLERS
# =========================
@dp.message(F.text == "/start")
async def start(m: types.Message):
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
    await m.answer("🎯 English Coach connected to Supabase!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_q_start(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'vocab', 'score':0, 'step':0, 'is_exam': False, 'used_items': []}
    await send_next_step(m.from_user.id)

@dp.message(F.document.mime_type == "application/pdf")
async def pdf_h(m: types.Message):
    st = await m.answer("⏳ Analyzing PDF with Supabase...")
    try:
        file = await bot.get_file(m.document.file_id)
        content = await bot.download_file(file.file_path)
        reader = PdfReader(io.BytesIO(content.read()))
        text = "".join([p.extract_text() for p in reader.pages[:2]])
        
        res_raw = await ai_request(f"Extract 3 B2 terms from: {text[:2000]}", "JSON only: {\"items\":[{\"w\":\"word\",\"d\":\"def\"}]}", json_mode=True, user_id=m.from_user.id)
        data = json.loads(res_raw)
        
        db = SessionLocal()
        for i in data.get('items', []):
            db.add(Vocab(word=i['w'], definition=i['d'], category='pdf', source=m.document.file_name))
        db.commit()
        db.close()
        await st.edit_text(f"✅ Added {len(data.get('items', []))} terms to cloud storage.")
    except Exception as e:
        logging.error(e)
        await st.edit_text("❌ PDF error.")

# =========================
# MAIN
# =========================
async def main():
    asyncio.create_task(start_web_server())
    logging.info("Bot is starting with PostgreSQL support...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
