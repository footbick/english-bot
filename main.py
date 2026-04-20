# --- ИСПРАВЛЕННЫЙ PROGRESS ---
@dp.message(F.text == "📊 My Progress")
async def show_progress(m: types.Message):
    db = SessionLocal()
    try:
        # Считаем общее количество слов пользователя
        count = db.query(Vocab).filter(Vocab.user_id == m.from_user.id).count()
        # Считаем по категориям для наглядности
        words = db.query(Vocab).filter(Vocab.user_id == m.from_user.id, Vocab.category == 'word').count()
        phrases = db.query(Vocab).filter(Vocab.user_id == m.from_user.id, Vocab.category == 'phrase').count()
        
        text = (f"📊 <b>Ваш прогресс:</b>\n\n"
                f"Всего в словаре: {count}\n"
                f"📝 Слова: {words}\n"
                f"🗣 Фразы: {phrases}\n\n"
                f"Продолжайте обучение! 🔥")
        await m.answer(text, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Progress Error: {e}")
        await m.answer("❌ Ошибка при получении статистики.")
    finally:
        db.close()

# --- ИСПРАВЛЕННЫЙ SPEAKING (с защитой от сбоев) ---
@dp.message(F.text == "🎤 Speaking Practice")
async def spk_start(m: types.Message):
    status = await m.answer("⏳ Генерирую тему для обсуждения...")
    try:
        # Упрощенный запрос к AI, чтобы снизить шанс ошибки
        topic_res = await ai_request("Give me one interesting B2 English discussion topic title.", "Text only, max 5 words.")
        question_res = await ai_request(f"Ask one short open-ended question about {topic_res}", "English only.")
        
        if not topic_res or not question_res:
            raise Exception("AI empty response")

        await status.delete()
        await m.answer(f"🗣 <b>Topic: {topic_res}</b>\n\n{question_res}", parse_mode="HTML")
        
        # Озвучка
        v = await generate_voice(question_res)
        await bot.send_voice(m.chat.id, BufferedInputFile(v.read(), filename="q.ogg"))
    except Exception as e:
        logging.error(f"Speaking Error: {e}")
        await status.edit_text("⚠️ Не удалось запустить практику. Попробуйте еще раз.")

# --- ИСПРАВЛЕННЫЙ PDF (с уведомлениями о каждом шаге) ---
@dp.message(F.document)
async def handle_pdf(m: types.Message):
    if not m.document.file_name.lower().endswith(".pdf"):
        return

    status = await m.answer("📥 Загружаю файл...")
    try:
        file = await bot.get_file(m.document.file_id)
        content = await bot.download_file(file.file_path)
        
        await status.edit_text("📄 Читаю содержимое PDF...")
        reader = PdfReader(io.BytesIO(content.read()))
        # Берем текст только первых 2 страниц, чтобы не перегружать AI
        full_text = ""
        for i in range(min(len(reader.pages), 2)):
            full_text += reader.pages[i].extract_text()
        
        if len(full_text) < 20:
            await status.edit_text("❌ Не удалось извлечь текст из PDF (возможно, это скан/картинка).")
            return

        await status.edit_text("🤖 AI выбирает полезные слова...")
        prompt = (f"Extract 5 useful B2 English words or phrases from this text. "
                  f"Return ONLY a JSON object: {{\"items\": [ {{\"w\":\"word\",\"d\":\"definition\",\"c\":\"category\"}} ] }}. "
                  f"Text: {full_text[:1500]}")
        
        res = await ai_request(prompt, "You are a vocabulary extractor. JSON ONLY.", json_mode=True)
        data = json.loads(res)
        items = data.get('items', [])

        if not items:
            await status.edit_text("AI не нашел подходящих слов в этом фрагменте.")
            return

        await status.edit_text(f"💾 Сохраняю {len(items)} слов в базу...")
        db = SessionLocal()
        for i in items:
            # Маппинг категорий, чтобы они попадали в фильтры бота
            cat = i.get('c', 'word').lower()
            if 'phrase' in cat: cat = 'phrase'
            elif 'idiom' in cat: cat = 'idiom'
            else: cat = 'word'
            
            db.add(Vocab(
                user_id=m.from_user.id,
                word=i['w'],
                definition=i['d'],
                category=cat,
                source="PDF"
            ))
        db.commit()
        db.close()
        
        await status.edit_text(f"✅ Готово! Добавлено слов: {len(items)}. Теперь они появятся в разделе Vocabulary.")
    except Exception as e:
        logging.error(f"PDF Error: {e}")
        await status.edit_text("❌ Произошла ошибка при обработке файла.")
