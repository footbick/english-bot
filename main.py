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

# --- 3. REFACTORED GRAMMAR SYSTEM ---
BASE_PROMPT = """
You are a professional English Grammar Examiner. 
STRICT RULES:
- Generate ONE unique 'fill-in-the-blank' question. Use '____' for the gap.
- Provide EXACTLY 3 answer options.
- Only ONE answer is correct. Distractors must be plausible.
- No HTML tags in JSON values.
- RETURN JSON ONLY: {"q": "sentence with ____", "o": ["correct", "wrong1", "wrong2"], "c": 0, "e": "concise English rule", "ru": "разбор на русском"}
"""

TOPIC_RULES = {
    "conditionals": """
    RULES (Conditionals):
    - Zero: If + Present -> Present.
    - 1st: If + Present -> will + verb.
    - 2nd: If + Past Simple -> would + verb. (Use 'were' for all persons).
    - 3rd: If + Past Perfect -> would have + V3.
    - Mixed (3->2): If + Past Perfect -> would + base.
    - Do not mix non-standard tenses.
    """,
    "passive": """
    RULES (Passive Voice):
    - Structure: (be) + Past Participle (V3).
    - Focus on tense-specific auxiliaries (am/is/are/was/were/been/being).
    - Avoid Active Voice answers.
    """,
    "complex": """
    RULES (Complex Object):
    - Structure: Verb + Object + (to) Infinitive or Participle.
    - Test verbs like: want, expect, see, hear, let, make.
    - Example: I want him to do it / I saw her dancing.
    """,
    "participle": """
    RULES (Participle Clauses):
    - Test V-ing (active meaning) vs V3 (passive meaning).
    - Context: sentences where participle replaces a relative clause.
    """,
    "prepositions": """
    RULES (Prepositions):
    - Test dependent prepositions (Verb/Adj + Prep).
    - Examples: depend on, interested in, proud of, look forward to.
    """,
    "general": "Mix all 5 topics above randomly with equal focus on advanced levels."
}

# --- 4. VALIDATION & ANTI-DUPLICATE ---
def normalize_question(q):
    return re.sub(r'[^a-zA-Z\s]', '', q.lower()).strip()

def is_similar(q1, q2):
    return SequenceMatcher(None, normalize_question(q1), normalize_question(q2)).ratio() > 0.6

def is_duplicate(sess, q):
    history = sess.get('history', [])
    return any(is_similar(q, h) for h in history)

def add_to_history(sess, q):
    history = sess.get('history', [])
    history.append(q)
    if len(history) > 20: history.pop(0)
    sess['history'] = history

def is_valid_conditional(data):
    try:
        q, correct = data['q'].lower(), data['o'][data['c']].lower()
        if "would have" in q and "had" in q and "have" not in correct and "____" in q:
            if "yesterday" in q: return "have" in correct # 3rd cond check
        if "was" in correct and ("would" in q or "were" in q): return False
        return True
    except: return False

# --- 5. CORE TOOLS ---
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

# --- 6. THE ENGINE ---
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
                prompt = f"Explain word '{target.word}'. JSON: {{\"d\":\"meaning\", \"o\":[\"{target.word}\", \"w2\", \"w3\"], \"ru\":\"перевод\"}}"
                data = await ai_request(prompt, "Expert Teacher.", True)
                opts = data.get('o', [target.word, "word_b", "word_c"])
                if target.word not in opts: opts[0] = target.word
                random.shuffle(opts)
                sess.update({'correct_id': opts.index(target.word), 'exp': data['d'], 'ru': data['ru'], 'word': target.word})
                await bot.send_poll(user_id, f"🎯 {header}{data['d']}", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # GRAMMAR BLOCK (Strict Rules + 3 Options + Validation)
        topic = sess.get('grammar_topic', 'general')
        final_prompt = BASE_PROMPT + TOPIC_RULES.get(topic, TOPIC_RULES["general"])
        final_prompt += f"\nVariability Seed: {random.randint(1, 1000000)}"
        
        data = None
        for _ in range(4):
            candidate = await ai_request(final_prompt, "Strict Grammar Examiner.", True)
            if not candidate: continue
            if len(candidate.get('o', [])) != 3: continue # Ensure 3 options
            if is_duplicate(sess, candidate['q']): continue
            if topic == "conditionals" and not is_valid_conditional(candidate): continue
            
            add_to_history(sess, candidate['q'])
            data = candidate; break
        
        if not data:
            await bot.send_message(user_id, "⚠️ AI error. Retrying..."); await send_next_step(user_id); return

        sess.update({'correct_id': data['c'], 'exp': data['e'], 'ru': data['ru']})
        sess.pop('word', None)
        await bot.send_poll(user_id, f"🎓 {header}{data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)

    except:
        await bot.send_message(user_id, "⚠️ System timeout. Please try again.")
    finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    
    word_label = f"{sess['word']}\n" if 'word' in sess else ""
    # ФИКС: Объяснение и СКРЫТЫЙ перевод в одном сообщении
    explanation = f"💡 {word_label}{sess.get('exp')}\n\n🇷🇺 <tg-spoiler>{sess.get('ru')}</tg-spoiler>"
    await bot.send_message(uid, explanation, parse_mode=ParseMode.HTML, disable_notification=True)
    
    sess['step'] += 1; await asyncio.sleep(0.5); await send_next_step(uid)

# --- 7. HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    db = SessionLocal()
    if not db.query(User).filter(User.user_id == m.from_user.id).first():
        db.add(User(user_id=m.from_user.id)); db.commit()
    db.close()
    kb = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],[KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],[KeyboardButton(text="📊 My Progress")]], resize_keyboard=True)
    await m.answer("🎯 Coach v11.5 Active!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    db = SessionLocal(); count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔤 Start Learning", callback_data="voc_start")],[InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]])
    await m.answer(f"📦 Vocabulary ({count} items):", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    if cb.data == "voc_add": await cb.message.answer("⌨️ Send words."); await cb.answer(); return
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'used': [], 'history': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.callback_query(F.data.startswith("list_"))
async def list_words(cb: types.CallbackQuery):
    off = int(cb.data.split('_')[1]); db = SessionLocal()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    if len(words) == 8: kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+8}")])
    if off > 0: kb.inline_keyboard.append([InlineKeyboardButton(text="⬅️ Back", callback_data=f"list_{max(0, off-8)}")])
    try: await cb.message.edit_text("🗑 Tap to delete:", reply_markup=kb)
    except: pass

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid, off = map(int, cb.data.split('_')[1:3]); db = SessionLocal(); db.query(Vocab).filter(Vocab.id == wid).delete(); db.commit()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    if len(words) == 8: kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+8}")])
    if off > 0: kb.inline_keyboard.append([InlineKeyboardButton(text="⬅️ Back", callback_data=f"list_{max(0, off-8)}")])
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
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': cb.data.split("_")[1], 'used': [], 'history': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.document)
async def handle_pdf(m: types.Message):
    if not m.document.file_name.endswith('.pdf'): return
    st = await m.answer("⏳ Processing PDF..."); file = await bot.get_file(m.document.file_id)
    content = await bot.download_file(file.file_path); reader = PdfReader(io.BytesIO(content.read()))
    text_data = "".join([p.extract_text() for p in reader.pages[:3]])
    res = await ai_request(f"Extract 10 words: {text_data[:2000]}", "JSON: {\"items\":[\"w1\"]}", True)
    items = res.get('items', []); db = SessionLocal()
    for i in items:
        db.query(Vocab).filter(Vocab.user_id == m.from_user.id, Vocab.word == i).delete()
        db.add(Vocab(user_id=m.from_user.id, word=i))
    db.commit(); db.close(); await st.edit_text(f"✅ Added {len(items)} items from PDF.")

@dp.message(F.text == "📊 My Progress")
async def exam_mode(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'used': [], 'history': []}
    await m.answer("🏆 Starting Exam"); await send_next_step(m.from_user.id)

@dp.message(F.voice)
async def handle_voice(m: types.Message):
    if m.from_user.id not in user_sessions or user_sessions[m.from_user.id].get('type') != 'speaking': return
    file = await bot.get_file(m.voice.file_id); content = await bot.download_file(file.file_path)
    trans = client.audio.transcriptions.create(file=("v.ogg", content.read()), model="whisper-large-v3", language="en").text
    resp = await ai_request(f"User: {trans}. Strictly 2 sentences.", "Teacher.", False)
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
