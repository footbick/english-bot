import os, io, asyncio, random, json, logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func, desc
from sqlalchemy.orm import declarative_base, sessionmaker

# --- CONFIG ---
logging.basicConfig(level=logging.INFO)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
client = Groq(api_key=GROQ_API_KEY)
user_sessions = {}

# --- DB SETUP ---
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Vocab(Base):
    __tablename__ = "vocab"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger)
    word = Column(String)
    definition = Column(Text)
    category = Column(String) # word, phrase, phrasal_verb, collocation, idiom
    source = Column(String)

Base.metadata.create_all(bind=engine)

# --- WEB SERVER ---
async def handle(request): return web.Response(text="Bot Active")
async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000)))
    await site.start()

# --- AI & VOICE ---
async def ai_request(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    def call():
        fmt = {"type": "json_object"} if json_mode else None
        return client.chat.completions.create(
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", response_format=fmt
        ).choices[0].message.content
    try: return await loop.run_in_executor(None, call)
    except: return None

async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0); return buf
    return await loop.run_in_executor(None, create_audio)

# --- CORE ENGINE ---
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return
    db = SessionLocal()
    try:
        # ПРОВЕРКА ЭКЗАМЕНА
        if sess.get('is_exam') and sess['step'] >= 10:
            await bot.send_message(user_id, f"🏁 Exam Finished! Score: {sess['score']}/10")
            user_sessions.pop(user_id, None); return

        q_type = random.choice(['vocab', 'grammar']) if sess.get('is_exam') else sess['type']

        if q_type == 'vocab':
            cat = sess.get('vocab_category')
            exclude = sess.get('used_items', [])
            query = db.query(Vocab).filter(Vocab.user_id == user_id)
            if cat and cat != "all": query = query.filter(Vocab.category == cat)
            
            target = query.filter(~Vocab.id.in_(exclude)).order_by(func.random()).first()
            if not target:
                if not exclude:
                    await bot.send_message(user_id, f"⚠️ Category '{cat}' is empty. Add words first!")
                    return
                sess['used_items'] = []; target = query.order_by(func.random()).first()

            sess.setdefault('used_items', []).append(target.id)
            prompt = (f"Word: '{target.word}'. Define for B2 level. "
                      f"JSON: {{\"def\":\"definition\", \"syn\":\"synonyms\", \"o\":[\"{target.word}\",\"w1\",\"w2\",\"w3\"], \"e_en\":\"explain\", \"e_ru\":\"объяснение\"}}")
            res = await ai_request(prompt, "Teacher. JSON ONLY. No Unicode bugs.", json_mode=True)
            data = json.loads(res)
            opts = data['o']; random.shuffle(opts)
            sess.update({'correct_id': opts.index(target.word), 'exp': f"{data['e_en']}\n\n🇷🇺 {data['e_ru']}"})
            await bot.send_message(user_id, f"📖 <b>{cat.upper()}</b>\n\nDef: {data['def']}\nSyn: {data['syn']}", parse_mode="HTML")
            await bot.send_poll(user_id, "Choose correct:", opts, type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
            return

        # GRAMMAR
        topic = sess.get('grammar_topic', 'general')
        prompt = f"Topic: {topic}. B2 level. Provide detailed English explanation and Russian translation. JSON: {{\"q\":\"... ____ ...\", \"o\":[\"a\",\"b\",\"c\",\"d\"], \"c\":0, \"e_en\":\"detailed explanation\", \"e_ru\":\"подробное объяснение\"}}"
        res = await ai_request(prompt, "Grammar Guru. JSON ONLY.", json_mode=True)
        data = json.loads(res)
        sess.update({'correct_id': data['c'], 'exp': f"{data['e_en']}\n\n🇷🇺 {data['e_ru']}"})
        await bot.send_poll(user_id, f"📝 Grammar: {topic}\n\n{data['q']}", data['o'], type='quiz', correct_option_id=data['c'], is_anonymous=False)
    finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    await bot.send_message(uid, f"💡 <b>Explanation:</b>\n{sess['exp']}", parse_mode="HTML")
    sess['step'] += 1; await asyncio.sleep(1); await send_next_step(uid)

# --- HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],
        [KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("🎯 English Coach v3.0 Ready!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔡 Words", callback_data="voc_word"), InlineKeyboardButton(text="🗣 Phrases", callback_data="voc_phrase")],
        [InlineKeyboardButton(text="🧪 Phrasal Verbs", callback_data="voc_phrasal_verb"), InlineKeyboardButton(text="🎭 Idioms", callback_data="voc_idiom")],
        [InlineKeyboardButton(text="➕ Add New", callback_data="voc_add"), InlineKeyboardButton(text="🗑 Delete", callback_data="list_0")]
    ])
    await m.answer("Vocabulary Section:", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    cat = cb.data.split("_")[1]
    if cat == "add": 
        await cb.message.answer("Send word or list (e.g. 'Apple, Banana')"); return
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'vocab_category': cat, 'used_items': []}
    await cb.message.answer(f"Starting {cat.upper()}..."); await send_next_step(cb.from_user.id)

@dp.callback_query(F.data.startswith("list_"))
async def list_words(cb: types.CallbackQuery):
    off = int(cb.data.split('_')[1]); db = SessionLocal()
    words = db.query(Vocab).filter(Vocab.user_id == cb.from_user.id).order_by(desc(Vocab.id)).limit(8).offset(off).all()
    db.close()
    if not words and off == 0: await cb.answer("Dictionary empty."); return
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    if len(words) == 8: kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+8}")])
    await cb.message.edit_text("Tap to delete (word will disappear):", reply_markup=kb)

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid, off = map(int, cb.data.split('_')[1:3]); db = SessionLocal()
    db.query(Vocab).filter(Vocab.id == wid).delete(); db.commit(); db.close()
    await list_words(cb) # Обновляем текущее сообщение

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive")],
        [InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],
        [InlineKeyboardButton(text="🎲 Mixed", callback_data="gt_general")]
    ])
    await m.answer("Choose Topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    topic = cb.data.split("_")[1]
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': topic}
    await send_next_step(cb.from_user.id)

@dp.message(F.text == "🎤 Speaking Practice")
async def spk_start(m: types.Message):
    res = await ai_request("Suggest 1 catchy B2 topic title.", "JSON: {\"t\":\"Title\"}", json_mode=True)
    topic = json.loads(res)['t']
    q = await ai_request(f"Ask 1 short B2 question about {topic}", "Short question only.")
    await m.answer(f"🗣 <b>Topic: {topic}</b>\n\n{q}", parse_mode="HTML")
    v = await generate_voice(q); await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))

@dp.message(F.document.mime_type == "application/pdf")
async def handle_pdf(m: types.Message):
    st = await m.answer("⏳ Reading PDF..."); file = await bot.get_file(m.document.file_id)
    content = await bot.download_file(file.file_path); reader = PdfReader(io.BytesIO(content.read()))
    text = "".join([p.extract_text() for p in reader.pages[:3]])
    res = await ai_request(f"Extract 5 B2 items. JSON: {{\"items\":[{{\"w\":\"word\",\"d\":\"def\",\"c\":\"word/phrase/idiom\"}}]}}. Text: {text[:2000]}", "JSON ONLY.", json_mode=True)
    items = json.loads(res).get('items', []); db = SessionLocal()
    for i in items: db.add(Vocab(user_id=m.from_user.id, word=i['w'], definition=i['d'], category=i['c'], source="PDF"))
    db.commit(); db.close(); await st.edit_text(f"✅ Added {len(items)} items from PDF.")

@dp.message(F.text == "📊 My Progress")
async def show_progress(m: types.Message):
    db = SessionLocal()
    count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count()
    cats = db.query(Vocab.category, func.count(Vocab.id)).filter(Vocab.user_id == m.from_user.id).group_by(Vocab.category).all()
    db.close()
    stats = "\n".join([f"• {c[0].upper()}: {c[1]}" for c in cats])
    await m.answer(f"📈 <b>Personal Stats:</b>\n\nTotal Vocabulary: {count}\n\nBreakdown:\n{stats}", parse_mode="HTML")

@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📁 Upload PDF", "🎤 Speaking Practice", "📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]: return
    words = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]
    db = SessionLocal()
    for w in words:
        res = await ai_request(f"Define '{w}'. JSON: {{\"d\":\"def\",\"c\":\"word/phrase/idiom\"}}", "JSON ONLY.", json_mode=True)
        data = json.loads(res)
        db.add(Vocab(user_id=m.from_user.id, word=w, definition=data['d'], category=data['c'], source="Manual"))
    db.commit(); db.close(); await m.answer(f"✅ Added {len(words)} items.")

async def main():
    asyncio.create_task(start_web_server()); await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
