import os, io, asyncio, random, json, logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, PollAnswer, BufferedInputFile
from aiohttp import web
from PyPDF2 import PdfReader
from groq import Groq
from gtts import gTTS
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Text, func
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
    category = Column(String) # 'word', 'phrase', 'phrasal_verb', 'pdf'
    source = Column(String)

Base.metadata.create_all(bind=engine)

# --- WEB SERVER (RENDER) ---
async def handle(request): return web.Response(text="Bot is Active")
async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000)))
    await site.start()

# --- AI & TOOLS ---
async def ai_request(prompt, system_msg, json_mode=False):
    loop = asyncio.get_event_loop()
    def call():
        fmt = {"type": "json_object"} if json_mode else None
        return client.chat.completions.create(
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", response_format=fmt
        ).choices[0].message.content
    try: return await loop.run_in_executor(None, call)
    except Exception as e: return None

async def generate_voice(text):
    loop = asyncio.get_event_loop()
    def create_audio():
        tts = gTTS(text=text.replace('*', ''), lang='en')
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0)
        return buf
    return await loop.run_in_executor(None, create_audio)

# --- ENGINE ---
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return
    
    db = SessionLocal()
    q_num = sess['step'] + 1
    total = 10 if sess.get('is_exam') else "∞"
    header = f"📍 Question {q_num}/{total}\n\n"

    if sess.get('is_exam') and sess['step'] >= 10:
        mistakes = ", ".join(sess.get('mistakes_words', [])) or "None"
        analysis = await ai_request(f"Mistakes: {mistakes}. Feedback in RU.", "B2 Teacher.")
        await bot.send_message(user_id, f"🏆 Exam Finished!\nScore: {sess['score']}/10\n\n{analysis}")
        user_sessions.pop(user_id, None); return

    q_type = random.choice(['vocab', 'grammar']) if sess.get('is_exam') else sess['type']
    
    try:
        if q_type == 'vocab':
            cat_filter = sess.get('vocab_category')
            query = db.query(Vocab)
            if cat_filter: query = query.filter(Vocab.category == cat_filter)
            target = query.order_by(func.random()).first()
            
            if not target: q_type = 'grammar' # Fallback
            else:
                dist = db.query(Vocab).filter(Vocab.id != target.id).order_by(func.random()).limit(3).all()
                options = list(set([target.definition] + [d.definition for d in dist]))
                random.shuffle(options)
                
                # Запрос подробного объяснения
                expl = await ai_request(f"Explain word/phrase '{target.word}'. Why '{target.definition}' is correct? Context for B2. RU translation.", "Teacher. JSON: {\"en\":\"...\",\"ru\":\"...\"}", json_mode=True)
                data = json.loads(expl)
                
                sess.update({'correct_id': options.index(target.definition), 'explanation': data['en'], 'explanation_ru': data['ru'], 'current_word': target.word})
                await bot.send_poll(user_id, f"{header}Choose correct definition:\n👉 '{target.word}'", options[:4], type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        # GRAMMAR
        topic = sess.get('grammar_topic', 'general')
        prompt = f"Topic: {topic}. B2 Level. UNIQUE. JSON: {{\"q\":\"Sentence with ____\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e_en\":\"Detailed explanation why it is correct and why others are wrong.\",\"e_ru\":\"Перевод и объяснение на русском.\"}}"
        res = await ai_request(prompt, "B2 Teacher. JSON ONLY.", json_mode=True)
        data = json.loads(res)
        
        sess.update({'correct_id': data['c'], 'explanation': data['e_en'], 'explanation_ru': data['e_ru'], 'current_word': 'Grammar'})
        await bot.send_poll(user_id, f"{header}{data['q']}", data['o'][:4], type='quiz', correct_option_id=data['c'], is_anonymous=False)
    finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    if p.option_ids[0] == sess['correct_id']: sess['score'] += 1
    else: sess.setdefault('mistakes_words', []).append(sess.get('current_word'))

    msg = f"💡 <b>Deep Explanation:</b>\n{sess['explanation']}\n\n🇷🇺 <b>RU:</b> <tg-spoiler>{sess['explanation_ru']}</tg-spoiler>"
    await bot.send_message(uid, msg, parse_mode="HTML")
    sess['step'] += 1
    await asyncio.sleep(1); await send_next_step(uid)

# --- HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],
        [KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("🎯 Welcome back! Your English Coach is ready.", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔡 Single Words", callback_data="voc_word")],
        [InlineKeyboardButton(text="🗣 Phrases", callback_data="voc_phrase")],
        [InlineKeyboardButton(text="🧪 Phrasal Verbs", callback_data="voc_phrasal_verb")],
        [InlineKeyboardButton(text="📄 From My PDF", callback_data="voc_pdf")],
        [InlineKeyboardButton(text="🚀 Infinite Mix", callback_data="voc_mix")]
    ])
    await m.answer("Choose what you want to practice:", reply_markup=kb)

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    cat = cb.data.split("_")[1] if "mix" not in cb.data else None
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'vocab_category': cat}
    await cb.message.edit_text(f"Starting {cat if cat else 'Mix'} practice...")
    await send_next_step(cb.from_user.id)

@dp.message(F.text == "⚙️ Grammar Test")
async def g_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Passive Voice", callback_data="gt_passive")],
        [InlineKeyboardButton(text="❓ Conditionals", callback_data="gt_conditionals")],
        [InlineKeyboardButton(text="🔗 Complex Object", callback_data="gt_complex")],
        [InlineKeyboardButton(text="🎲 Mixed Practice", callback_data="gt_general")]
    ])
    await m.answer("Choose grammar topic:", reply_markup=kb)

@dp.callback_query(F.data.startswith("gt_"))
async def g_start(cb: types.CallbackQuery):
    topic = cb.data.split("_")[1]
    user_sessions[cb.from_user.id] = {'type':'grammar', 'step':0, 'score':0, 'grammar_topic': topic}
    await cb.message.answer(f"Starting {topic.upper()}..."); await send_next_step(cb.from_user.id)

@dp.message(F.text == "🎤 Speaking Practice")
async def spk_start(m: types.Message):
    # Добавлен random seed в промпт для разнообразия
    res = await ai_request(f"Suggest 3 diverse and unusual B2 topics for discussion. Random seed: {random.randint(1,999)}. JSON: {{\"topics\":[\"Title\"]}}", "JSON ONLY.", json_mode=True)
    topics = json.loads(res)['topics']
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=f"spk_{t[:20]}")] for t in topics])
    await m.answer("Choose a fresh topic for today:", reply_markup=kb)

@dp.callback_query(F.data.startswith("spk_"))
async def spk_selected(cb: types.CallbackQuery):
    topic = cb.data[4:]; q = await ai_request(f"Ask 1 deep B2 question about {topic}", "Teacher.")
    await cb.message.answer(f"🗣 <b>Topic: {topic}</b>\n{q}", parse_mode="HTML")
    v = await generate_voice(q); await bot.send_voice(cb.message.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))

@dp.message(F.document.mime_type == "application/pdf")
async def handle_pdf(m: types.Message):
    st = await m.answer("⏳ Analyzing PDF... Please wait.")
    try:
        file = await bot.get_file(m.document.file_id)
        content = await bot.download_file(file.file_path)
        reader = PdfReader(io.BytesIO(content.read()))
        text = "".join([p.extract_text() for p in reader.pages[:5]])
        
        prompt = f"Extract 5-8 useful B2 items (words, phrases, phrasal verbs). JSON: {{\"items\":[{{\"w\":\"...\",\"d\":\"...\",\"c\":\"word/phrase/phrasal_verb\"}}]}}. Text: {text[:3500]}"
        res = await ai_request(prompt, "JSON ONLY.", json_mode=True)
        data = json.loads(res)
        
        db = SessionLocal()
        for i in data.get('items', []):
            db.add(Vocab(word=i['w'], definition=i['d'], category=i['c'], source=m.document.file_name))
        db.commit(); db.close()
        await st.edit_text(f"✅ Successfully processed {len(data.get('items', []))} items from PDF. They are added to your personal library!")
    except: await st.edit_text("❌ Error processing PDF.")

@dp.message(F.text == "📊 My Progress")
async def ex_start(m: types.Message):
    user_sessions[m.from_user.id] = {'type':'mix', 'step':0, 'score':0, 'is_exam': True, 'mistakes_words': []}
    await m.answer("🏆 <b>Final Exam Mode</b>\n10 questions from all topics. Good luck!", parse_mode="HTML")
    await send_next_step(m.from_user.id)

@dp.message(F.text == "📁 Upload PDF")
async def pdf_btn(m: types.Message): await m.answer("📂 Send your PDF file (text-based) to extract new vocabulary.")

# --- MAIN ---
async def main():
    asyncio.create_task(start_web_server())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
