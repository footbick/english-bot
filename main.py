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
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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

# --- WEB SERVER ---
async def handle(request): return web.Response(text="Bot Active")
async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 10000)))
    await site.start()

# --- AI TOOLS ---
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

# --- ENGINE ---
async def send_next_step(user_id):
    sess = user_sessions.get(user_id)
    if not sess: return
    db = SessionLocal()
    try:
        header = f"📍 Question {sess['step'] + 1}/10\n\n" if sess.get('is_exam') else ""
        if sess.get('is_exam') and sess['step'] >= 10:
            user_sessions.pop(user_id, None); await bot.send_message(user_id, "🏆 Exam Finished!"); return

        q_type = random.choice(['vocab', 'grammar']) if sess.get('is_exam') else sess['type']

        if q_type == 'vocab':
            cat = sess.get('vocab_category')
            query = db.query(Vocab)
            if cat and cat != "all": query = query.filter(Vocab.category == cat)
            target = query.order_by(func.random()).first()
            
            if not target: q_type = 'grammar'
            else:
                prompt = (f"Word: '{target.word}'. Create a B2 definition and 2 synonyms. "
                          f"DO NOT use the word itself. Provide 3 wrong options. "
                          f"JSON: {{\"def\":\"...\",\"syn\":\"...\",\"o\":[\"{target.word}\",\"w1\",\"w2\",\"w3\"],\"e_en\":\"...\",\"e_ru\":\"...\"}}")
                res = await ai_request(prompt, "English Teacher. JSON ONLY.", json_mode=True)
                data = json.loads(res)
                options = data['o']; random.shuffle(options)
                sess.update({'correct_id': options.index(target.word), 'explanation': data['e_en'], 'explanation_ru': data['e_ru'], 'current_word': target.word})
                await bot.send_message(user_id, f"{header}📖 <b>Definition:</b> {data['def']}\n🔗 <b>Synonyms:</b> {data['syn']}", parse_mode="HTML")
                await bot.send_poll(user_id, "Guess the word:", options[:4], type='quiz', correct_option_id=sess['correct_id'], is_anonymous=False)
                return

        topic = sess.get('grammar_topic', 'general')
        prompt = f"Topic: {topic}. B2 level. JSON: {{\"q\":\"... with ____\",\"o\":[\"a\",\"b\",\"c\",\"d\"],\"c\":0,\"e_en\":\"...\",\"e_ru\":\"...\"}}"
        res = await ai_request(prompt, "Grammar Teacher. JSON ONLY.", json_mode=True)
        data = json.loads(res)
        sess.update({'correct_id': data['c'], 'explanation': data['e_en'], 'explanation_ru': data['e_ru']})
        await bot.send_poll(user_id, f"{header}{data['q']}", data['o'][:4], type='quiz', correct_option_id=data['c'], is_anonymous=False)
    finally: db.close()

@dp.poll_answer()
async def handle_poll(p: PollAnswer):
    uid = p.user.id
    if uid not in user_sessions: return
    sess = user_sessions[uid]
    await bot.send_message(uid, f"💡 <b>Explanation:</b>\n{sess['explanation']}\n\n🇷🇺 <b>RU:</b> <tg-spoiler>{sess['explanation_ru']}</tg-spoiler>", parse_mode="HTML")
    sess['step'] += 1; await asyncio.sleep(1); await send_next_step(uid)

# --- HANDLERS ---
@dp.message(F.text == "/start")
async def cmd_start(m: types.Message):
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📁 Upload PDF"), KeyboardButton(text="🎤 Speaking Practice")],
        [KeyboardButton(text="📚 Vocabulary"), KeyboardButton(text="⚙️ Grammar Test")],
        [KeyboardButton(text="📊 My Progress")]
    ], resize_keyboard=True)
    await m.answer("🎯 Coach Active!", reply_markup=kb)

@dp.message(F.text == "📚 Vocabulary")
async def v_menu(m: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔡 Words", callback_data="voc_word"), InlineKeyboardButton(text="🗣 Phrases", callback_data="voc_phrase")],
        [InlineKeyboardButton(text="🧪 Phrasal Verbs", callback_data="voc_phrasal_verb"), InlineKeyboardButton(text="📄 PDF Words", callback_data="voc_pdf")],
        [InlineKeyboardButton(text="🔗 Collocations", callback_data="voc_collocation"), InlineKeyboardButton(text="🎭 Idioms", callback_data="voc_idiom")],
        [InlineKeyboardButton(text="➕ Add New Words", callback_data="voc_add_help")],
        [InlineKeyboardButton(text="🗑 View / Delete Words", callback_data="list_0")]
    ])
    await m.answer("Vocabulary Management:", reply_markup=kb)

@dp.callback_query(F.data == "voc_add_help")
async def v_add_help(cb: types.CallbackQuery):
    await cb.message.answer("📝 Чтобы добавить слова, просто отправь их боту списком через запятую или каждое с новой строки.\n\nНапример:\n`Advantage, Challenge, Get over`")
    await cb.answer()

@dp.callback_query(F.data.startswith("voc_"))
async def v_start(cb: types.CallbackQuery):
    cat = cb.data.split("_")[1]
    if cat == "add": return
    user_sessions[cb.from_user.id] = {'type':'vocab', 'step':0, 'score':0, 'vocab_category': cat}
    await cb.message.answer(f"Starting {cat.replace('_', ' ')}..."); await send_next_step(cb.from_user.id)

@dp.callback_query(F.data.startswith("list_"))
async def list_words(cb: types.CallbackQuery):
    off = int(cb.data.split('_')[1]); db = SessionLocal()
    words = db.query(Vocab).order_by(Vocab.id.desc()).limit(10).offset(off).all(); db.close()
    if not words: await cb.answer("No more words."); return
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=f"❌ {w.word}", callback_data=f"del_{w.id}_{off}")] for w in words])
    kb.inline_keyboard.append([InlineKeyboardButton(text="Next ➡️", callback_data=f"list_{off+10}")])
    await cb.message.edit_text("Tap to delete:", reply_markup=kb)

@dp.callback_query(F.data.startswith("del_"))
async def del_word(cb: types.CallbackQuery):
    wid = int(cb.data.split('_')[1]); db = SessionLocal()
    db.query(Vocab).filter(Vocab.id == wid).delete(); db.commit(); db.close(); await list_words(cb)

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
    await cb.message.answer(f"Practice: {topic.upper()}"); await send_next_step(cb.from_user.id)

@dp.message(F.text == "🎤 Speaking Practice")
async def spk_start(m: types.Message):
    res = await ai_request("3 B2 topics. JSON: {\"topics\":[\"Title\"]}", "JSON ONLY.", json_mode=True)
    topics = json.loads(res)['topics']
    kb = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=f"spk_{t[:20]}")] for t in topics])
    await m.answer("Topics:", reply_markup=kb)

@dp.callback_query(F.data.startswith("spk_"))
async def spk_selected(cb: types.CallbackQuery):
    topic = cb.data[4:]; q = await ai_request(f"Ask about {topic}", "Teacher.")
    await cb.message.answer(f"🗣 <b>{topic}</b>\n{q}", parse_mode="HTML")
    v = await generate_voice(q); await bot.send_voice(cb.message.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))

@dp.message(F.text)
async def manual_add(m: types.Message):
    if m.text.startswith("/") or m.text in ["📁 Upload PDF", "🎤 Speaking Practice", "📚 Vocabulary", "⚙️ Grammar Test", "📊 My Progress"]: return
    words = [i.strip() for i in m.text.replace(',', '\n').split('\n') if i.strip()]
    db = SessionLocal()
    for w in words:
        res = await ai_request(f"Define '{w}' for B2.", "Short definition.")
        if res: db.add(Vocab(word=w, definition=res, category='word', source='manual'))
    db.commit(); db.close(); await m.answer(f"✅ Added {len(words)} items.")

async def main():
    asyncio.create_task(start_web_server()); await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
