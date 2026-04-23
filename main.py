import os, io, asyncio, random, json, logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiogram.enums import ParseMode
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

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)
user_sessions = {}
db_lock = asyncio.Lock()

# --- 2. DB SETUP ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=10, pool_pre_ping=True, pool_recycle=300)
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

async def send_reminder():
    db = SessionLocal()
    try:
        users = db.query(User.user_id).all()
        for u in users:
            try: 
                await bot.send_message(u.user_id, "🔔 <b>Time for English!</b>\nLet's do some practice.", parse_mode=ParseMode.HTML)
            except: pass
    finally: db.close()

# --- 3. TOOLS (Model: 70b-versatile + Retry) ---
async def ai_request(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    for attempt in range(3): # 3 попытки для исключения таймаутов
        def call():
            fmt = {"type": "json_object"} if json_mode else None
            try:
                return client.chat.completions.create(
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile", 
                    response_format=fmt, 
                    timeout=60 # Увеличенный таймаут для умной модели
                ).choices[0].message.content
            except: return None
        res = await loop.run_in_executor(None, call)
        if res: return res
        await asyncio.sleep(1.5)
    return None

async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0); return buf
    return await loop.run_in_executor(None, create_audio)

# --- 4. ENGINE ---
async def send_next_step(user_id):
    async with db_lock:
        sess = user_sessions.get(user_id)
        if not sess: return
        db = SessionLocal()
        try:
            is_ex = sess.get('is_exam', False)
            if is_ex and sess['step'] >= 10:
                await bot.send_message(user_id, f"🏆 <b>Exam Result: {sess['score']}/10</b>", parse_mode=ParseMode.HTML)
                user_sessions.pop(user_id, None); return

            header = f"<b>Question {sess['step'] + 1}/10</b>\n\n" if is_ex else ""
            q_type = random.choice(['vocab', 'grammar']) if is_ex else sess['type']

            if q_type == 'vocab':
                cat = sess.get('vocab_category', 'all')
                query = db.query(Vocab).filter(Vocab.user_id == user_id)
                if cat != 'all' and not is_ex: query = query.filter(Vocab.category == cat)
                target = query.filter(~Vocab.id.in_(sess.get('used', []))).order_by(func.random()).first()
                if not target:
                    if is_ex: q_type = 'grammar'
                    else: await bot.send_message(user_id, "⚠️ Category empty."); return
                
                if target:
                    sess.setdefault('used', []).append(target.id)
                    res = await ai_request(f"Word: {target.word}. JSON: {{\"d\":\"def\",\"s\":\"syn\",\"o\":[\"{target.word}\",\"w1\",\"w2\",\"w3\"],\"e_en\":\"rule\",\"e_ru\":\"перевод\"}}", "Expert Teacher.", True)
                    data = json.loads(res); opts = data['o']; random.shuffle(opts)
                    sess.update({'correct_id': opts.index(target.word), 'exp': f"{data['e_en']}\n\n🇷🇺 <b>Перевод:</b> <tg-spoiler>{data['e_ru']}</tg-spoiler>"})
                    await bot.send_message(user_id, f"{header}📖 <b>Definition:</b> {data['d']}\n🔗 <b>Synonyms:</b> {data['s']}", parse_mode=ParseMode.HTML)
                    await bot.send_poll(user_id, "Guess word:", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                    return

            topic = sess.get('grammar_topic', 'general')
            sys_instr = "You are a PhD English Professor. Ensure grammar logic is 100% correct. Explanation MUST be detailed (2-3 sentences)."
            res = await ai_request(f"Topic: {topic}. UNIQUE test. JSON: {{\"q\":\"sentence ____ rest\",\"o\":[\"correct\",\"w1\",\"w2\",\"w3\"],\"c\":0,\"e_en\":\"detailed eng rule\",\"e_ru\":\"подробный разбор\"}}", sys_instr, True)
            data = json.loads(res)
            sess.update({'correct_id': data['c'], 'exp': f"{data['e_en']}\n\n🇷🇺 <b>Разбор:</b> <tg-spoiler>{data['e_ru']}</tg-spoiler>"})
            if header: await bot.send_message(user_id, header, parse_mode=ParseMode.HTML)
            await bot.send_poll(user_id, f"📝 Grammar: {topic}\n\n{data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)
        except: await bot.send_message(user_id, "⚠️ AI timeout. Click the button again.")
        finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    await bot.send_message(uid, f"💡 <b>Explanation:</b>\n{sess.get('exp')}", parse_mode=ParseMode.HTML)
    sess['step'] += 1; await asyncio.sleep(0.5); await send_next_step(uid)

# --- 5. HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    db = SessionLocal()
    if not db.query(User).filter(User.user_id == m.from_user.id).first():
        db.add(User(user_id=m.from_user.id)); db.commit()
    db.close()
    kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],[KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],[KeyboardButton(text="📊 My Progress")]], resize_keyboard=True)
    await m.answer("🎯 Coach v6.5 Active!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    db = SessionLocal(); count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔡 Words", callback_data="voc_word"), InlineKeyboardButton(text="🗣 Phrases", callback_data="voc_phrase")],[InlineKeyboardButton(text="🧪 Phrasal Verbs", callback_data="voc_phrasal_verb"), InlineKeyboardButton(text="🎭 Idioms", callback_data="voc_idiom")],[InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]])
    await m.answer(f"Vocabulary (Total: {count}):", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    cat = cb.data.split("_")[1]
    if cat == "add": await cb.message.answer("Send any amount of words (one per line)."); return
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'vocab_category': cat, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.callback_query(F.data.startswith("list_"))
async def list_words(cb: types.CallbackQuery):
    off = int(cb.data.split('_')[1]); db = SessionLocal()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all(); db.close()
    if not words and off == 0: await cb.answer("Empty."); return
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    if len(words) == 8: kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+8}")])
    try: await cb.message.edit_text("Manage Dictionary:", reply_markup=kb)
    except: pass

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid, off = map(int, cb.data.split('_')[1:3]); db = SessionLocal()
    db.query(Vocab).filter(Vocab.id == wid).delete(); db.commit(); db.close()
    await cb.answer("Deleted."); await list_words(cb)

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive"), InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],[InlineKeyboardButton(text="🔗 Complex Object", callback_data="gt_complex"), InlineKeyboardButton(text="✨ Participle", callback_data="gt_participle")],[InlineKeyboardButton(text="📍 Prepositions", callback_data="gt_prepositions"), InlineKeyboardButton(text="🎲 Mixed Practice", callback_data="gt_general")]])
    await m.answer("Choose Topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    topic = cb.data.split("_")[1]
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': topic, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.text == "📊 My Progress")
async def exam_mode(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'used': []}
    await m.answer("🏆 <b>Starting Exam (10 Questions)</b>", parse_mode=ParseMode.HTML); await send_next_step(m.from_user.id)

@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📁 Upload PDF", "📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]: return
    lines = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]
    st = await m.answer(f"⏳ Processing {len(lines)} items..."); db = SessionLocal(); added = 0
    for w in lines:
        prompt = f"Define '{w}'. JSON: {{\"d\":\"def\", \"c\":\"word/phrase/phrasal_verb/idiom\"}}"
        res = await ai_request(prompt, "Vocab Teacher.", True)
        if res:
            try:
                data = json.loads(res); db.add(Vocab(user_id=m.from_user.id, word=w, definition=data['d'], category=data['c']))
                added += 1
                if added % 5 == 0: await st.edit_text(f"⏳ Progress: {added}/{len(lines)}...")
            except: continue
        await asyncio.sleep(0.1)
    db.commit(); db.close(); await st.edit_text(f"✅ Success! Added {added} items.")

async def start_web_server():
    app = web.Application(); app.router.add_get("/", lambda r: web.Response(text="OK"))
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000))).start()

async def main():
    init_db(); asyncio.create_task(start_web_server())
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    scheduler.add_job(send_reminder, CronTrigger(hour='9,12,15,18', minute=0))
    scheduler.start()
    await bot.delete_webhook(drop_pending_updates=True); await dp.start_polling(bot)

if __name__ == "__main__": asyncio.run(main())
