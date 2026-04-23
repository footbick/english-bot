import os, io, asyncio, random, json, logging, re
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func, desc, text
from sqlalchemy.orm import declarative_base, sessionmaker
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# --- 1. CONFIG ---
logging.basicConfig(level=logging.INFO)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

bot = Bot(token=TELEGRAM_TOKEN, default_properties=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)
user_sessions = {}

# --- 2. DB SETUP ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=15, max_overflow=10, pool_pre_ping=True, pool_recycle=300)
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

# --- 3. CORE TOOLS ---
def clean_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text
    except: return text

async def ai_request(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    # Используем 70b для ума, но с бронированным парсингом
    def call():
        try:
            res = client.chat.completions.create(
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"} if json_mode else None,
                timeout=40
            ).choices[0].message.content
            return json.loads(clean_json(res)) if json_mode else res
        except: return None
    return await loop.run_in_executor(None, call)

async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0); return buf
    return await loop.run_in_executor(None, create_audio)

# --- 4. THE STABLE ENGINE ---
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return
    db = SessionLocal()
    try:
        is_ex = sess.get('is_exam', False)
        if is_ex and sess['step'] >= 10:
            await bot.send_message(user_id, f"🏆 <b>Exam Result: {sess['score']}/10</b>")
            user_sessions.pop(user_id, None); return

        header = f"<b>Question {sess['step'] + 1}/10</b>\n\n" if is_ex else ""
        q_type = random.choice(['vocab', 'grammar']) if is_ex else sess['type']

        if q_type == 'vocab':
            # Выбираем случайное слово, которое мы еще не использовали в этой сессии
            target = db.query(Vocab).filter(Vocab.user_id == user_id, ~Vocab.id.in_(sess.get('used', []))).order_by(func.random()).first()
            
            if not target:
                if is_ex: q_type = 'grammar'
                else: await bot.send_message(user_id, "⚠️ Your dictionary is empty. Add words first!"); return
            
            if target:
                sess.setdefault('used', []).append(target.id)
                # Генерируем данные квиза только СЕЙЧАС (1 слово = быстро)
                data = await ai_request(f"Word: {target.word}. JSON: {{\"d\":\"def\",\"s\":\"syn\",\"o\":[\"{target.word}\",\"w1\",\"w2\",\"w3\"],\"e_ru\":\"перевод\",\"c\":\"cat\"}}", "Expert Teacher.", True)
                
                if not data or 'o' not in data: raise Exception("AI Fail")
                
                opts = data['o']
                if target.word not in opts: opts[0] = target.word
                random.shuffle(opts)
                
                # Сохраняем категорию и определение в базу для будущего
                target.definition = data['d']
                target.category = data.get('c', 'word')
                db.commit()

                sess.update({'correct_id': opts.index(target.word), 'exp': f"<b>{target.word}</b>\n{data['d']}\n\n🇷🇺 {data['e_ru']}"})
                await bot.send_message(user_id, f"{header}📖 <b>Definition:</b> {data['d']}\n🔗 <b>Synonyms:</b> {data['s']}")
                await bot.send_poll(user_id, "Choose correct word:", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # GRAMMAR
        topic = sess.get('grammar_topic', 'general')
        data = await ai_request(f"Topic: {topic}. JSON: {{\"q\":\".. ____ ..\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e\":\"rule\"}}", "Grammar Professor.", True)
        if not data: raise Exception("AI Fail")
        
        sess.update({'correct_id': data['c'], 'exp': data['e']})
        if header: await bot.send_message(user_id, header)
        await bot.send_poll(user_id, f"📝 Grammar: {topic}\n\n{data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)

    except:
        # Fallback если AI совсем не отвечает
        await bot.send_poll(user_id, "⚠️ AI timeout. Try this backup:\nI ___ English every day.", ["study","studies","studied","studying"], type='quiz', correct_option_id=0, is_anonymous=False)
        sess['correct_id'] = 0
        sess['exp'] = "Present Simple: I study."
    finally: db.close()

# --- 5. HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    db = SessionLocal()
    if not db.query(User).filter(User.user_id == m.from_user.id).first():
        db.add(User(user_id=m.from_user.id)); db.commit()
    db.close()
    kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],[KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],[KeyboardButton(text="📊 My Progress")]], resize_keyboard=True)
    await m.answer("🎯 Coach v8.0 (Ultra Stability) Active!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    db = SessionLocal(); count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔡 Start Learning", callback_data="voc_start")],
        [InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]
    ])
    await m.answer(f"Your Vocabulary ({count} items):", reply_markup=kb)

@dp.message(F.text == "📊 My Progress")
async def exam_mode(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'used': []}
    await m.answer("🏆 <b>Starting Exam (10 Questions)</b>"); await send_next_step(m.from_user.id)

@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📁 Upload PDF", "🎤 Speaking Practice", "📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]: return
    # ФИКС: Мгновенное сохранение без вызова AI
    lines = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]
    db = SessionLocal()
    for w in lines:
        db.add(Vocab(user_id=m.from_user.id, word=w))
    db.commit(); db.close()
    await m.answer(f"✅ Added {len(lines)} items! You can start practicing now.")

@dp.callback_query(F.data == "voc_start")
async def v_start(cb: types.CallbackQuery):
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.callback_query(F.data.startswith("list_"))
async def list_words(cb: types.CallbackQuery):
    off = int(cb.data.split('_')[1]); db = SessionLocal()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all(); db.close()
    if not words and off == 0: await cb.answer("Empty."); return
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    if len(words) == 8: kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+8}")])
    try: await cb.message.edit_text("Tap to delete:", reply_markup=kb)
    except: pass

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid, off = map(int, cb.data.split('_')[1:3]); db = SessionLocal()
    db.query(Vocab).filter(Vocab.id == wid).delete(); db.commit(); db.close()
    await cb.answer("Deleted."); await list_words(cb)

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    await bot.send_message(uid, f"💡 <b>Explanation:</b>\n{sess.get('exp')}")
    sess['step'] += 1; await asyncio.sleep(0.5); await send_next_step(uid)

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive"), InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],[InlineKeyboardButton(text="🔗 Complex Object", callback_data="gt_complex"), InlineKeyboardButton(text="✨ Participle", callback_data="gt_participle")],[InlineKeyboardButton(text="📍 Prepositions", callback_data="gt_prepositions"), InlineKeyboardButton(text="🎲 Mixed Practice", callback_data="gt_general")]])
    await m.answer("Choose Topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    topic = cb.data.split("_")[1]
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': topic, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.voice)
async def handle_voice(m: types.Message):
    if m.from_user.id not in user_sessions or user_sessions[m.from_user.id].get('type') != 'speaking': return
    st = await m.answer("👂 Listening..."); file = await bot.get_file(m.voice.file_id)
    content = await bot.download_file(file.file_path)
    trans = client.audio.transcriptions.create(file=("v.ogg", content.read()), model="whisper-large-v3", language="en").text
    await st.edit_text(f"💬 <b>You:</b> {trans}")
    history = user_sessions[m.from_user.id].get('history', [])
    resp = await ai_request(f"History: {history}. User: {trans}. Reply briefly.", "Teacher.")
    history.append(trans); history.append(resp)
    await m.answer(f"🗣 {resp}")
    v = await generate_voice(resp); await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="r.ogg"))

async def start_web_server():
    app = web.Application(); app.router.add_get("/", lambda r: web.Response(text="OK"))
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000))).start()

async def main():
    init_db(); asyncio.create_task(start_web_server())
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    scheduler.add_job(lambda: asyncio.create_task(send_reminder()), CronTrigger(hour='9,12,15,18', minute=0))
    scheduler.start()
    await bot.delete_webhook(drop_pending_updates=True); await dp.start_polling(bot)

if __name__ == "__main__": asyncio.run(main())
