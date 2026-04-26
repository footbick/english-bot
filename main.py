import os, io, asyncio, random, json, logging, re
from difflib import SequenceMatcher
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

# --- 3. ACADEMIC GRAMMAR RULES (STRICT) ---
BASE_PROMPT = """
You are a strict English grammar examiner. 
STRICT RULES:
- Generate ONE gap-fill question: a sentence with '____'.
- Provide exactly 3 options for the answer.
- Only ONE answer is correct.
- NO HTML TAGS. NO MARKDOWN.
- EXTREME DIVERSITY: Use science, art, law, and tech contexts.
- RETURN JSON ONLY: {"q": "sentence...", "o": ["correct", "wrong1", "wrong2"], "c": 0, "e": "English explanation", "ru": "разбор на русском"}
"""

TOPIC_RULES = {
    "conditionals": """
    RULES for Conditionals:
    - Zero: If+Present -> Present.
    - 1st: If+Present -> will+base.
    - 2nd: If+Past -> would+base. STRICT: Use 'were' for all persons, NEVER 'was'.
    - 3rd: If+Past Perfect -> would have+V3.
    - Mixed: If+Past Perfect -> would+base (3->2).
    """,
    "passive": "RULES for Passive Voice: Use 'be + V3'. Test tense markers (already, since, yesterday).",
    "complex": "RULES for Complex Object: verb + object + to-infinitive (want/expect) or bare-infinitive (make/let/see).",
    "participle": "RULES for Participle: V-ing (active action) vs V3 (passive/completed action).",
    "prepositions": "RULES for Prepositions: Fixed collocations, phrasal verbs, and time/place prepositions.",
    "general": "RULES for General: Randomly pick one advanced B2-C1 grammar concept."
}

def is_grammatically_correct(data, topic):
    try:
        q, correct = data['q'].lower(), data['o'][data['c']].lower()
        if topic == "conditionals" and "if" in q:
            if "was" in correct and ("were" in q or "would" in q): return False
        if len(data['o']) != 3: return False
        return True
    except: return False

# --- 4. CORE TOOLS ---
def clean_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text
    except: return text

async def ai_request(prompt, sys_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    def call():
        try:
            res = client.chat.completions.create(
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"} if json_mode else None,
                timeout=45
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

# --- 5. THE ENGINE ---
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return
    db = SessionLocal()
    try:
        is_ex = sess.get('is_exam', False)
        if is_ex and sess['step'] >= 10:
            await bot.send_message(user_id, f"🏆 Exam Result: {sess['score']}/10")
            user_sessions.pop(user_id, None); return

        q_type = random.choice(['vocab', 'grammar']) if is_ex else sess['type']

        if q_type == 'vocab':
            target = db.query(Vocab).filter(Vocab.user_id == user_id, ~Vocab.id.in_(sess.get('used', []))).order_by(func.random()).first()
            if not target:
                if is_ex: q_type = 'grammar'
                else: await bot.send_message(user_id, "⚠️ Vocabulary empty."); return
            
            if target:
                sess.setdefault('used', []).append(target.id)
                prompt = f"Explain '{target.word}'. JSON: {{\"d\":\"definition\", \"o\":[\"{target.word}\", \"w2\", \"w3\"], \"ru\":\"перевод\"}}"
                data = await ai_request(prompt, "English Teacher.", True)
                opts = data.get('o', [target.word, "opt2", "opt3"])[:3]
                if target.word not in opts: opts[0] = target.word
                random.shuffle(opts)
                sess.update({'correct_id': opts.index(target.word), 'exp': data['d'], 'ru': data['ru'], 'word': target.word})
                await bot.send_poll(user_id, f"🎯 {data['d']}", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # GRAMMAR ENGINE
        topic = sess.get('grammar_topic', 'general')
        sys_msg = BASE_PROMPT + TOPIC_RULES.get(topic, TOPIC_RULES["general"])
        prompt = f"Create one B2 grammar test on topic '{topic}'. Use random seed {random.randint(1,99999)}"
        
        data = None
        for _ in range(4):
            candidate = await ai_request(prompt, "Academic Grammar Examiner.", True)
            if candidate and is_grammatically_correct(candidate, topic):
                data = candidate; break
        
        if not data:
            await bot.send_message(user_id, "⚠️ AI error. Trying again..."); await send_next_step(user_id); return

        sess.update({'correct_id': data['c'], 'exp': data['e'], 'ru': data['ru']})
        sess.pop('word', None)
        await bot.send_poll(user_id, f"🎓 {data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)

    except:
        await bot.send_message(user_id, "⚠️ System timeout.")
    finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    
    word_label = f"{sess['word']}\n" if 'word' in sess else ""
    # ПРАВКА: Объяснение + Скрытый перевод
    await bot.send_message(uid, f"💡 {word_label}{sess.get('exp')}")
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
    await m.answer("🎯 Coach v12.0 Active!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    db = SessionLocal(); count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔤 Start Learning", callback_data="voc_start")],[InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]])
    await m.answer(f"📦 Vocabulary ({count} items):", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    if cb.data == "voc_add": await cb.message.answer("⌨️ Send words."); await cb.answer(); return
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid, off = map(int, cb.data.split('_')[1:3])
    db = SessionLocal(); db.query(Vocab).filter(Vocab.id == wid).delete(); db.commit()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    await cb.message.edit_reply_markup(reply_markup=kb); await cb.answer("Deleted")

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive"), InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],
        [InlineKeyboardButton(text="🔗 Complex Object", callback_data="gt_complex"), InlineKeyboardButton(text="✨ Participle", callback_data="gt_participle")],
        [InlineKeyboardButton(text="📍 Prepositions", callback_data="gt_prepositions"), InlineKeyboardButton(text="🎲 Mixed Practice", callback_data="gt_general")]
    ])
    await m.answer("📝 Choose Topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': cb.data.split("_")[1], 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.voice)
async def handle_voice(m: types.Message):
    if m.from_user.id not in user_sessions or user_sessions[m.from_user.id].get('type') != 'speaking': return
    file = await bot.get_file(m.voice.file_id); content = await bot.download_file(file.file_path)
    trans = client.audio.transcriptions.create(file=("v.ogg", content.read()), model="whisper-large-v3", language="en").text
    resp = await ai_request(f"User: {trans}. Strictly 2 sentences.", "Teacher.", False)
    await m.answer(f"🗣 {resp}"); v = await generate_voice(resp)
    await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="r.ogg"))

# --- 7. RUN ---
async def start_web_server():
    app = web.Application(); app.router.add_get("/", lambda r: web.Response(text="OK"))
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000))).start()

async def main():
    init_db(); asyncio.create_task(start_web_server())
    await bot.delete_webhook(drop_pending_updates=True); await dp.start_polling(bot)

if __name__ == "__main__": asyncio.run(main())
