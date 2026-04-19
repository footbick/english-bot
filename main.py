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

# SQLAlchemy
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
    'passive': "Passive Voice structures (all tenses).",
    'conditionals': "Conditionals (Zero, 1st, 2nd, 3rd, Mixed).",
    'complex': "Complex Object (e.g., 'I want him to stay').",
    'participle': "Gerund and Participle.",
    'prepositions': "Prepositions (dependent prepositions).",
    'general': "Mixed B2 English Grammar.",
    'pdf': "Grammar and vocabulary from the uploaded PDF context."
}

# =========================
# DATABASE
# =========================
Base = declarative_base()
engine = create_engine(DATABASE_URL)
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
    return web.Response(text="Bot is running!")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 10000))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

# =========================
# AI & VOICE TOOLS
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
    except Exception as e:
        logging.error(f"AI Error: {e}")
        return None

async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    return await loop.run_in_executor(None, create_audio)

# =========================
# CORE LOGIC
# =========================
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return

    # Завершение экзамена
    if sess.get('is_exam') and sess['step'] >= 10:
        mistakes = ", ".join(sess.get('mistakes_words', [])) or "None"
        analysis = await ai_request(f"Mistakes: {mistakes}. Give tips in Russian.", "English Teacher.", user_id=user_id)
        await bot.send_message(user_id, f"🏆 Exam Finished!\nScore: {sess['score']}/10\n\n{analysis}")
        user_sessions.pop(user_id, None)
        return

    q_type = random.choice(['vocab', 'grammar']) if sess.get('is_exam') else sess['type']
    db = SessionLocal()
    
    try:
        if q_type == 'vocab':
            target = db.query(Vocab).order_by(func.random()).first()
            if not target:
                q_type = 'grammar'
            else:
                dist = db.query(Vocab).filter(Vocab.id != target.id).order_by(func.random()).limit(3).all()
                options = list(set([target.definition] + [d.definition for d in dist]))
                random.shuffle(options)
                
                e_en = await ai_request(f"Explain word '{target.word}' for B2 level.", "Teacher.", user_id=user_id)
                e_ru = await ai_request(f"Translate to Russian: {e_en}", "Translator.", user_id=user_id)
                
                sess.update({'correct_id': options.index(target.definition), 'explanation': e_en, 'explanation_ru': e_ru, 'current_word': target.word})
                await bot.send_poll(user_id, f"Choose definition for: {target.word}", options[:4], type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # Логика Грамматики
        topic_key = sess.get('grammar_topic', 'general')
        topic_desc = TOPIC_PROMPTS.get(topic_key, "Mixed B2 Grammar")
        
        prompt = f"Topic: {topic_desc}. JSON: {{\"q\":\"... with ____\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e_en\":\"...\",\"e_ru\":\"...\"}}"
        res = await ai_request(prompt, "Teacher. JSON ONLY.", json_mode=True, user_id=user_id)
        data = json.loads(res)
        
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
# HANDLERS
# =========================
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    db = SessionLocal()
    if not db.query(User).filter(User.user_id == m.from_user.id).first():
        db.add(User(user_id=m.from_user.id)); db.commit()
    db.close()
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],
        [KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("🎯 English Coach Cloud Active!", reply_markup=kb)

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive")],
        [InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],
        [InlineKeyboardButton(text="🔗 Complex Object", callback_data="gt_complex")],
        [InlineKeyboardButton(text="✨ Participle & Gerund", callback_data="gt_participle")],
        [InlineKeyboardButton(text="📍 Prepositions", callback_data="gt_prepositions")],
        [InlineKeyboardButton(text="🎲 Mixed Practice", callback_data="gt_general")]
    ])
    await m.answer("Choose topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    topic = cb.data.split("_")[1]
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': topic}
    await cb.message.answer(f"Starting {topic.upper()} practice!"); await send_next_step(cb.from_user.id)

@dp.message(F.text == "📚 Vocabulary")
async def v_start(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'vocab', 'step':0, 'score':0}
    await send_next_step(m.from_user.id)

@dp.message(F.text == "📊 My Progress")
async def ex_start(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'mistakes_words': []}
    await m.answer("🏆 Starting 10-Question Exam Mix!"); await send_next_step(m.from_user.id)

@dp.message(F.document.mime_type == "application/pdf")
async def handle_pdf(m: types.Message):
    st = await m.answer("⏳ Analyzing PDF...")
    try:
        file = await bot.get_file(m.document.file_id)
        content = await bot.download_file(file.file_path)
        reader = PdfReader(io.BytesIO(content.read()))
        text = "".join([p.extract_text() for p in reader.pages[:3]])
        res = await ai_request(f"Extract 5 B2 terms from: {text[:3000]}", "JSON: {\"items\":[{\"w\":\"word\",\"d\":\"def\"}]}", json_mode=True)
        data = json.loads(res)
        db = SessionLocal()
        for i in data.get('items', []):
            db.add(Vocab(word=i['w'], definition=i['d'], category='pdf', source=m.document.file_name))
        db.commit(); db.close()
        await st.edit_text(f"✅ Added {len(data.get('items', []))} terms to Supabase.")
    except: await st.edit_text("❌ PDF error.")

@dp.message(F.text == "🎤 Speaking Practice")
async def spk_start(m: types.Message):
    res = await ai_request("Suggest 3 B2 topics. JSON: {\"topics\":[\"Title\"]}", "JSON ONLY.", json_mode=True)
    topics = json.loads(res)['topics']
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=f"spk_{t[:15]}")] for t in topics])
    await m.answer("Choose a topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("spk_"))
async def spk_selected(cb: types.CallbackQuery):
    topic = cb.data[4:]
    q = await ai_request(f"Ask a B2 question about {topic}", "Teacher.")
    await cb.message.answer(f"🗣 <b>{topic}</b>\n{q}", parse_mode="HTML")
    v = await generate_voice(q); await bot.send_voice(cb.message.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))

@dp.message(F.voice)
async def voice_process(m: types.Message):
    file = await bot.get_file(m.voice.file_id)
    content = await bot.download_file(file.file_path)
    trans = client.audio.transcriptions.create(file=("v.ogg", content.read()), model="whisper-large-v3", language="en").text
    f = await ai_request(f"Analyze speech: '{trans}'", "Teacher.")
    await m.answer(f"💬 <b>You:</b> {trans}\n\n<b>Feedback:</b> {f}", parse_mode="HTML")

# =========================
# MAIN
# =========================
async def main():
    asyncio.create_task(start_web_server())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
