import os, io, asyncio, random, json, logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func, desc, text
from sqlalchemy.orm import declarative_base, sessionmaker

# --- 1. CONFIG ---
logging.basicConfig(level=logging.INFO)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)
user_sessions = {}
processing_lock = asyncio.Lock()

# --- 2. DB SETUP ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Vocab(Base):
    __tablename__ = "vocab"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger, index=True)
    word = Column(String)
    definition = Column(Text)
    category = Column(String) 
    source = Column(String)

def init_db():
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try: conn.execute(text("ALTER TABLE vocab ADD COLUMN IF NOT EXISTS user_id BIGINT")); conn.commit()
        except: pass

# --- 3. TOOLS ---
async def ai_request(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    def call():
        fmt = {"type": "json_object"} if json_mode else None
        try:
            return client.chat.completions.create(
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", response_format=fmt, timeout=12
            ).choices[0].message.content
        except: return None
    return await loop.run_in_executor(None, call)

async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0); return buf
    return await loop.run_in_executor(None, create_audio)

# --- 4. ENGINE ---
async def send_next_step(user_id):
    async with processing_lock:
        sess = user_sessions.get(user_id)
        if not sess: return
        db = SessionLocal()
        try:
            # ЛОГИКА ЭКЗАМЕНА (MY PROGRESS)
            is_ex = sess.get('is_exam', False)
            if is_ex and sess['step'] >= 10:
                await bot.send_message(user_id, f"🏆 <b>Exam Result: {sess['score']}/10</b>\nKeep practicing!", parse_mode="HTML")
                user_sessions.pop(user_id, None); return

            header = f"<b>Question {sess['step'] + 1}/10</b>\n\n" if is_ex else ""
            q_type = random.choice(['vocab', 'grammar']) if is_ex else sess['type']

            if q_type == 'vocab':
                cat = sess.get('vocab_category', 'all')
                query = db.query(Vocab).filter(Vocab.user_id == user_id)
                if cat != 'all' and not is_ex: query = query.filter(Vocab.category == cat)
                target = query.filter(~Vocab.id.in_(sess.get('used', []))).order_by(func.random()).first()
                
                if not target:
                    if is_ex: q_type = 'grammar' # Если слов нет, идем в грамматику
                    else: await bot.send_message(user_id, "⚠️ Category empty."); return
                else:
                    sess.setdefault('used', []).append(target.id)
                    res = await ai_request(f"Word: {target.word}. B2 def & 2 synonyms. JSON: {{\"d\":\"..\",\"s\":\"..\",\"o\":[\"{target.word}\",\"w1\",\"w2\",\"w3\"],\"e\":\"..\"}}", "Teacher.", True)
                    data = json.loads(res); opts = data['o']; random.shuffle(opts)
                    sess.update({'correct_id': opts.index(target.word), 'exp': data['e']})
                    await bot.send_message(user_id, f"{header}📖 <b>Definition:</b> {data['d']}\n🔗 <b>Synonyms:</b> {data['s']}", parse_mode="HTML")
                    await bot.send_poll(user_id, "Guess the word:", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                    return

            # ГРАММАТИКА
            topic = sess.get('grammar_topic', 'general')
            res = await ai_request(f"Topic: {topic}. B2. JSON: {{\"q\":\".. ____ ..\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e\":\"..\"}}", "Grammar Guru.", True)
            data = json.loads(res)
            sess.update({'correct_id': data['c'], 'exp': data['e']})
            await bot.send_poll(user_id, f"{header}📝 Grammar: {topic}\n\n{data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)
        except: await bot.send_message(user_id, "⚠️ Error. Try again.")
        finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    await bot.send_message(uid, f"💡 <b>Explanation:</b>\n{sess.get('exp')}", parse_mode="HTML")
    sess['step'] += 1; await asyncio.sleep(0.5); await send_next_step(uid)

# --- 5. HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],
        [KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("🎯 English Coach v5.2 Ready!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    db = SessionLocal(); count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count(); db.close()
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔡 Words", callback_data="voc_word"), InlineKeyboardButton(text="🗣 Phrases", callback_data="voc_phrase")],
        [InlineKeyboardButton(text="🧪 Phrasal Verbs", callback_data="voc_phrasal_verb"), InlineKeyboardButton(text="🎭 Idioms", callback_data="voc_idiom")],
        [InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]
    ])
    await m.answer(f"Vocabulary (Total: {count}):", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    cat = cb.data.split("_")[1]
    if cat == "add": await cb.message.answer("Send words (e.g. Apple, Banana)"); return
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'vocab_category': cat, 'used': []}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive"), InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],
        [InlineKeyboardButton(text="🔗 Complex Object", callback_data="gt_complex"), InlineKeyboardButton(text="✨ Participle", callback_data="gt_participle")],
        [InlineKeyboardButton(text="📍 Prepositions", callback_data="gt_prepositions"), InlineKeyboardButton(text="🎲 Mixed", callback_data="gt_general")]
    ])
    await m.answer("Choose Topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    topic = cb.data.split("_")[1]
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': topic}
    await send_next_step(cb.from_user.id); await cb.answer()

@dp.message(F.text == "🎤 Speaking Practice")
async def spk_start(m: types.Message):
    st = await m.answer("⏳ Generating topic..."); q = await ai_request("Ask 1 short B2 question.", "Short question only.")
    await st.delete(); await m.answer(f"🗣 <b>Question:</b>\n{q}", parse_mode="HTML")
    v = await generate_voice(q); await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))

@dp.message(F.text == "📊 My Progress")
async def exam_mode(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'used': []}
    await m.answer("🏆 <b>Starting Exam (10 Questions)</b>", parse_mode="HTML"); await send_next_step(m.from_user.id)

@dp.message(F.document)
async def handle_pdf(m: types.Message):
    st = await m.answer("⏳ Processing PDF..."); file = await bot.get_file(m.document.file_id)
    content = await bot.download_file(file.file_path); reader = PdfReader(io.BytesIO(content.read()))
    text = "".join([p.extract_text() for p in reader.pages[:2]])
    res = await ai_request(f"Extract 5 items. JSON: {{\"items\":[{{\"w\":\"word\",\"d\":\"def\",\"c\":\"word/phrase/idiom\"}}]}}. Text: {text[:1500]}", "JSON.", True)
    items = json.loads(res).get('items', []); db = SessionLocal()
    for i in items: db.add(Vocab(user_id=m.from_user.id, word=i['w'], definition=i['d'], category=i.get('c', 'word')))
    db.commit(); db.close(); await st.edit_text(f"✅ Added {len(items)} items.")

@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📁 Upload PDF", "🎤 Speaking Practice", "📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]: return
    words = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]
    db = SessionLocal()
    for w in words:
        res = await ai_request(f"Define '{w}'. JSON: {{\"d\":\"def\",\"c\":\"word/phrase/idiom\"}}", "JSON.", True)
        data = json.loads(res); db.add(Vocab(user_id=m.from_user.id, word=w, definition=data['d'], category=data['c']))
    db.commit(); db.close(); await m.answer(f"✅ Added {len(words)} words.")

async def main():
    init_db(); asyncio.create_task(web._run_app(web.Application(), port=10000))
    await bot.delete_webhook(drop_pending_updates=True); await dp.start_polling(bot)

if __name__ == "__main__": asyncio.run(main())
