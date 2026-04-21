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
word_queue = asyncio.Queue()

# --- 2. DB SETUP ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=300)
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

# --- 3. BACKGROUND WORKER ---
async def process_word_queue():
    while True:
        user_id, word_id, word_text = await word_queue.get()
        db = SessionLocal()
        try:
            # Используем быструю модель для классификации
            res = client.chat.completions.create(
                messages=[{"role": "user", "content": f"Define '{word_text}'. JSON: {{\"d\":\"def\", \"c\":\"word/phrase/phrasal_verb/idiom\"}}"}],
                model="llama-3.1-8b-instant", response_format={"type": "json_object"}, timeout=20
            ).choices[0].message.content
            data = json.loads(res)
            db.query(Vocab).filter(Vocab.id == word_id).update({"definition": data['d'], "category": data['c']})
            db.commit()
        except Exception as e:
            logging.error(f"Worker Error: {e}")
        finally:
            db.close()
            word_queue.task_done()
            await asyncio.sleep(1)

# --- 4. TOOLS ---
async def ai_request(prompt, system_msg, json_mode=False, speed_mode=True):
    """speed_mode=True использует быструю модель 8b, False - мощную 70b"""
    model = "llama-3.1-8b-instant" if speed_mode else "llama-3.3-70b-versatile"
    loop = asyncio.get_event_loop()
    def call():
        fmt = {"type": "json_object"} if json_mode else None
        try:
            return client.chat.completions.create(
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                model=model, response_format=fmt, timeout=25
            ).choices[0].message.content
        except: return None
    return await loop.run_in_executor(None, call)

async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0); return buf
    return await loop.run_in_executor(None, create_audio)

# --- 5. ENGINE ---
async def send_next_step(user_id):
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
            # Фильтруем слова, у которых уже есть определение
            query = db.query(Vocab).filter(Vocab.user_id == user_id, Vocab.definition != None)
            if cat != 'all' and not is_ex: query = query.filter(Vocab.category == cat)
            target = query.filter(~Vocab.id.in_(sess.get('used', []))).order_by(func.random()).first()
            
            if not target:
                if is_ex: q_type = 'grammar'
                else: await bot.send_message(user_id, "⚠️ Category empty or words are still being processed."); return
            
            if target:
                sess.setdefault('used', []).append(target.id)
                # Быстрая модель для квизов
                res = await ai_request(f"Word: {target.word}. JSON: {{\"d\":\"def\",\"s\":\"syn\",\"o\":[\"{target.word}\",\"w1\",\"w2\",\"w3\"],\"e_en\":\"eng\",\"e_ru\":\"рус\"}}", "Teacher.", True, True)
                data = json.loads(res); opts = data['o']; random.shuffle(opts)
                sess.update({'correct_id': opts.index(target.word), 'exp': f"{data['e_en']}\n\n🇷🇺 <b>Перевод:</b> <tg-spoiler>{data['e_ru']}</tg-spoiler>"})
                await bot.send_message(user_id, f"{header}📖 <b>Definition:</b> {data['d']}\n🔗 <b>Synonyms:</b> {data['s']}", parse_mode=ParseMode.HTML)
                await bot.send_poll(user_id, "Guess word:", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # GRAMMAR (8b model for speed)
        topic = sess.get('grammar_topic', 'general')
        res = await ai_request(f"Topic: {topic}. JSON: {{\"q\":\".. ____ ..\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e_en\":\"eng\",\"e_ru\":\"рус\"}}", "Grammar Teacher.", True, True)
        data = json.loads(res)
        sess.update({'correct_id': data['c'], 'exp': f"{data['e_en']}\n\n🇷🇺 <b>Разбор:</b> <tg-spoiler>{data['e_ru']}</tg-spoiler>"})
        if header: await bot.send_message(user_id, header, parse_mode=ParseMode.HTML)
        await bot.send_poll(user_id, f"📝 Grammar: {topic}\n\n{data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)
    except Exception as e:
        logging.error(f"Engine Error: {e}")
        await bot.send_message(user_id, "⚠️ AI was a bit slow. Please try clicking the button again.")
    finally: db.close()

# --- 6. HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    db = SessionLocal()
    if not db.query(User).filter(User.user_id == m.from_user.id).first():
        db.add(User(user_id=m.from_user.id)); db.commit()
    db.close()
    kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],[KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],[KeyboardButton(text="📊 My Progress")]], resize_keyboard=True)
    await m.answer("🎯 Coach v7.1 Active!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    db = SessionLocal(); count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id, Vocab.definition != None).count(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔡 Words", callback_data="voc_word"), InlineKeyboardButton(text="🗣 Phrases", callback_data="voc_phrase")],[InlineKeyboardButton(text="🧪 Phrasal Verbs", callback_data="voc_phrasal_verb"), InlineKeyboardButton(text="🎭 Idioms", callback_data="voc_idiom")],[InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]])
    await m.answer(f"Vocabulary (Ready: {count}):", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    cat = cb.data.split("_")[1]
    if cat == "add": await cb.message.answer("Send words (one per line)."); return
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'vocab_category': cat, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📁 Upload PDF", "🎤 Speaking Practice", "📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]: return
    lines = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]
    db = SessionLocal()
    for w in lines:
        new_v = Vocab(user_id=m.from_user.id, word=w, source="Manual")
        db.add(new_v); db.commit()
        await word_queue.put((m.from_user.id, new_v.id, w))
    db.close()
    await m.answer(f"✅ Received {len(lines)} items. Processing in background...")

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

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    await bot.send_message(uid, f"💡 <b>Explanation:</b>\n{sess.get('exp')}", parse_mode=ParseMode.HTML)
    sess['step'] += 1; await asyncio.sleep(0.5); await send_next_step(uid)

@dp.message(F.text == "🎤 Speaking Practice")
async def spk_menu(m: types.Message):
    st = await m.answer("⏳ Generating topics..."); 
    # Здесь можно оставить 70b для качества тем
    res = await ai_request("5 catchy topics. JSON: {\"topics\":[\"T1\",\"T2\",\"T3\",\"T4\",\"T5\"]}", "JSON ONLY.", True, False)
    topics = json.loads(res)['topics']
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=f"spk_st_{t[:20]}")] for t in topics])
    await st.delete(); await m.answer("Pick a topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("spk_st_"))
async def spk_init(cb: types.CallbackQuery):
    topic = cb.data[7:]; 
    q = await ai_request(f"Conversation about {topic}. Max 2 sentences.", "Teacher.", False, False)
    user_sessions[cb.from_user.id] = {'type': 'speaking', 'history': [q]}
    await cb.message.answer(f"🗣 <b>Topic: {topic}</b>\n{q}", parse_mode=ParseMode.HTML)
    v = await generate_voice(q); await bot.send_voice(cb.message.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))
    await cb.answer()

@dp.message(F.voice)
async def handle_voice(m: types.Message):
    if m.from_user.id not in user_sessions or user_sessions[m.from_user.id].get('type') != 'speaking': return
    st = await m.answer("👂 Listening..."); file = await bot.get_file(m.voice.file_id)
    content = await bot.download_file(file.file_path)
    trans = client.audio.transcriptions.create(file=("v.ogg", content.read()), model="whisper-large-v3", language="en").text
    await st.edit_text(f"💬 <b>You:</b> {trans}", parse_mode=ParseMode.HTML)
    history = user_sessions[m.from_user.id].get('history', [])
    resp = await ai_request(f"History: {history}. User: {trans}. Reply briefly and ask question.", "Teacher.", False, False)
    history.append(trans); history.append(resp)
    await m.answer(f"🗣 {resp}")
    v = await generate_voice(resp); await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="r.ogg"))

@dp.message(F.text == "📊 My Progress")
async def exam_mode(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'used': []}
    await m.answer("🏆 <b>Starting Exam (10 Questions)</b>", parse_mode=ParseMode.HTML); await send_next_step(m.from_user.id)

async def start_web_server():
    app = web.Application(); app.router.add_get("/", lambda r: web.Response(text="OK"))
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000))).start()

async def main():
    init_db(); asyncio.create_task(start_web_server())
    asyncio.create_task(process_word_queue())
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    scheduler.add_job(send_reminder, CronTrigger(hour='9,12,15,18', minute=0))
    scheduler.start()
    await bot.delete_webhook(drop_pending_updates=True); await dp.start_polling(bot)

if __name__ == "__main__": asyncio.run(main())
