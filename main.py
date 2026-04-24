import os, io, asyncio, random, json, logging, re
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func, desc
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
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=10, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Vocab(Base):
    __tablename__ = "vocab"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger, index=True)
    word = Column(String)
    definition = Column(Text)
    category = Column(String) 

class User(Base):
    __tablename__ = "users"
    user_id = Column(BigInteger, primary_key=True)

def init_db():
    Base.metadata.create_all(bind=engine)

# --- 3. PROMPT ARCHITECTURE (DIVERSITY FIX) ---
BASE_PROMPT = """
You are a strict English grammar test generator.
RULES:
- Generate ONE unique test question. Only ONE correct answer.
- NO HTML tags. NO labels.
- EXTREME DIVERSITY REQUIRED: Use contexts like gardening, car repair, office life, medical checkups, hiking, theater, architecture.
- STRICTLY FORBIDDEN TOPICS: Astronauts, Space, Lottery, Tropical island, Beach, Buying a house. Do not use them.
- Question, answer and explanation MUST match perfectly.

Return JSON ONLY:
{
"q": "sentence with ____",
"o": ["a","b","c","d"],
"c": 0,
"e": "short explanation in English",
"ru": "короткий разбор на русском"
}
"""

TOPIC_RULES = {
    "conditionals": "Use ONLY one type: zero/first/second/third/mixed. STRICT: If+Past=2nd (use 'were'), If+Present=1st. NO space context.",
    "passive": "Use Passive Voice (be + V3). Test various tenses.",
    "complex": "Use Complex Object. Example: I expect them to arrive.",
    "participle": "Test V-ing vs V3 in advanced sentences.",
    "prepositions": "Test prepositions in common verb/adjective combinations.",
    "general": "Mix all grammar rules randomly."
}

# --- 4. CORE TOOLS ---
def clean_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text
    except: return text

async def ai_request(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    for attempt in range(2):
        def call():
            try:
                res = client.chat.completions.create(
                    messages=[{"role": "system", "content": system_msg + " NO HTML."}, {"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    response_format={"type": "json_object"} if json_mode else None,
                    timeout=45
                ).choices[0].message.content
                return json.loads(clean_json(res)) if json_mode else res
            except: return None
        result = await loop.run_in_executor(None, call)
        if result: return result
    return None

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
            await bot.send_message(user_id, f"🏆 Exam Result: {sess['score']}/10")
            user_sessions.pop(user_id, None); return

        header = f"Question {sess['step'] + 1}/10\n\n" if is_ex else ""
        q_type = random.choice(['vocab', 'grammar']) if is_ex else sess['type']

        if q_type == 'vocab':
            target = db.query(Vocab).filter(Vocab.user_id == user_id, ~Vocab.id.in_(sess.get('used', []))).order_by(func.random()).first()
            if not target:
                if is_ex: q_type = 'grammar'
                else: await bot.send_message(user_id, "⚠️ Your dictionary is empty."); return
            
            if target:
                sess.setdefault('used', []).append(target.id)
                prompt = f"Create quiz for word '{target.word}'. JSON: {{\"d\":\"clear definition\",\"o\":[\"{target.word}\",\"s1\",\"s2\",\"s3\"],\"ru\":\"перевод\"}}"
                data = await ai_request(prompt, "English Teacher.", True)
                opts = data.get('o', [target.word, "word_b", "word_c", "word_d"])
                if target.word not in opts: opts[0] = target.word
                random.shuffle(opts)
                sess.update({'correct_id': opts.index(target.word), 'exp': data['d'], 'ru': data['ru'], 'word': target.word})
                await bot.send_poll(user_id, f"🎯 {header}{data['d']}", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # GRAMMAR
        sess.pop('word', None) # ПРАВКА: Убираем hold on (очистка слова из сессии)
        topic = sess.get('grammar_topic', 'general')
        final_prompt = BASE_PROMPT + "\n" + TOPIC_RULES.get(topic, TOPIC_RULES["general"])
        final_prompt += f"\nDiversity Seed: {random.randint(1, 999999)}"
        
        data = await ai_request(final_prompt, "Strict Grammar Master.", True)
        if not data:
            await bot.send_message(user_id, "⚠️ AI timeout. Trying again..."); await send_next_step(user_id); return

        sess.update({'correct_id': data['c'], 'exp': data['e'], 'ru': data['ru']})
        await bot.send_poll(user_id, f"🎓 {header}{data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)

    except:
        await bot.send_message(user_id, "⚠️ AI error. Please try again.")
    finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    
    word_info = f"{sess['word']}\n" if 'word' in sess else ""
    await bot.send_message(uid, f"💡 {word_info}{sess.get('exp')}")
    await bot.send_message(uid, f"🇷🇺 <tg-spoiler>{sess.get('ru')}</tg-spoiler>", parse_mode=ParseMode.HTML, disable_notification=True)
    
    sess['step'] += 1; await asyncio.sleep(0.5); await send_next_step(uid)

# --- 6. HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    db = SessionLocal()
    if not db.query(User).filter(User.user_id == m.from_user.id).first():
        db.add(User(user_id=m.from_user.id)); db.commit()
    db.close()
    kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],[KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],[KeyboardButton(text="📊 My Progress")]], resize_keyboard=True)
    await m.answer("🎯 Coach v10.6 Active!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    db = SessionLocal(); count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔤 Start Learning", callback_data="voc_start_learning")],[InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]])
    await m.answer(f"📦 Vocabulary ({count} items):", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    if cb.data == "voc_add": await cb.message.answer("⌨️ Send words (one per line)."); await cb.answer(); return
    # ПРАВКА 1: Починен запуск обучения
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.callback_query(F.data.startswith("list_"))
async def list_words(cb: types.CallbackQuery):
    off = int(cb.data.split('_')[1]); db = SessionLocal()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all(); db.close()
    if not words and off == 0: await cb.answer("Empty."); return
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    if len(words) == 8: kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+8}")])
    if off > 0: kb.inline_keyboard.append([InlineKeyboardButton(text="⬅️ Back", callback_data=f"list_{max(0, off-8)}")])
    try: await cb.message.edit_text("🗑 Tap to delete:", reply_markup=kb)
    except: pass

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid, off = map(int, cb.data.split('_')[1:3])
    db = SessionLocal(); db.query(Vocab).filter(Vocab.id == wid).delete(); db.commit()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    if len(words) == 8: kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+8}")])
    if off > 0: kb.inline_keyboard.append([InlineKeyboardButton(text="⬅️ Back", callback_data=f"list_{max(0, off-8)}")])
    await cb.message.edit_reply_markup(reply_markup=kb); await cb.answer("Deleted")

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive"), InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],[InlineKeyboardButton(text="🔗 Complex Object", callback_data="gt_complex"), InlineKeyboardButton(text="✨ Participle", callback_data="gt_participle")],[InlineKeyboardButton(text="📍 Prepositions", callback_data="gt_prepositions"), InlineKeyboardButton(text="🎲 Mixed Practice", callback_data="gt_general")]])
    await m.answer("📝 Choose Topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': cb.data.split("_")[1], 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.text == "📊 My Progress")
async def exam_mode(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'used': []}
    await m.answer("🏆 Starting Exam"); await send_next_step(m.from_user.id)

@dp.message(F.text == "🎤 Speaking Practice")
async def spk_menu(m: types.Message):
    res = await ai_request("5 catchy topics", "JSON: {\"topics\":[\"T1\"]}", True)
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=f"spk_st_{t[:20]}")] for t in res.get('topics', [])])
    await m.answer("🗣 Pick a topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("spk_st_"))
async def spk_init(cb: types.CallbackQuery):
    topic = cb.data[7:]; q = await ai_request(f"Start conversation about {topic}. Strictly 2-3 sentences.", "Teacher.", False)
    user_sessions[cb.from_user.id] = {'type': 'speaking', 'history': [q]}
    await cb.message.answer(f"🗣 Topic: {topic}\n{q}")
    v = await generate_voice(q); await bot.send_voice(cb.message.chat.id, BufferedInputFile(v.read(), filename="q.ogg")); await cb.answer()

@dp.message(F.voice)
async def handle_voice(m: types.Message):
    if m.from_user.id not in user_sessions or user_sessions[m.from_user.id].get('type') != 'speaking': return
    file = await bot.get_file(m.voice.file_id); content = await bot.download_file(file.file_path)
    trans = client.audio.transcriptions.create(file=("v.ogg", content.read()), model="whisper-large-v3", language="en").text
    resp = await ai_request(f"User: {trans}. 2-3 sentences.", "Teacher.", False)
    user_sessions[m.from_user.id]['history'].append(trans)
    await m.answer(f"🗣 {resp}"); v = await generate_voice(resp)
    await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="r.ogg"))

@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📁 Upload PDF", "🎤 Speaking Practice", "📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]: return
    lines = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]
    db = SessionLocal()
    for w in lines:
        db.query(Vocab).filter(Vocab.user_id == m.from_user.id, Vocab.word == w).delete()
        db.add(Vocab(user_id=m.from_user.id, word=w))
    db.commit(); db.close(); await m.answer(f"✅ Added {len(lines)} items.")

async def start_web_server():
    app = web.Application(); app.router.add_get("/", lambda r: web.Response(text="OK"))
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000))).start()

async def main():
    init_db(); asyncio.create_task(start_web_server())
    await bot.delete_webhook(drop_pending_updates=True); await dp.start_polling(bot)

if __name__ == "__main__": asyncio.run(main())
