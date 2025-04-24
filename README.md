# 🧠 Call Intelligence System using Groq LLM & MongoDB

This project is a smart call analysis pipeline that leverages **Groq's blazing-fast LLaMA-3 model** to turn raw transcription data into **structured, actionable insights**.

The core idea? You already have hours of customer calls. Now it's time to *understand them*—at scale.

---

## 🛠️ What This Project Does

This Python-based system:
- Pulls transcribed phone calls directly from **MongoDB**
- Uses **Groq LLaMA-3** via Langchain to:
  - Summarize the call
  - Extract key details (name, phone number, location, DOB, etc.)
  - Detect mentions of **rescheduling** (e.g., “next Monday at 7”)
  - Analyze **sentiment**
  - Assess **customer interest**
- Writes all results back to the database, enhancing each call record with valuable metadata

---

## 🧰 Tools & Libraries

- **Groq LLaMA-3** (`langchain-groq`) — the powerhouse brain
- **MongoDB + pymongo** — handles data persistence
- **SentenceTransformers** — for embeddings (ready for future vector search)
- **dotenv, logging, threading** — production-level polish

---

## ✅ How to Use

1. **Set Up API Keys**
   - Create a `.env` file with:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Processor**
   ```bash
   python main.py
   ```

   It’ll fetch transcriptions, analyze them via Groq, and store results—all auto-magically.

---

## 🔒 Processed File Tracking

To avoid re-processing, filenames are logged in `processed_llm_files.txt`. Smart and simple.

---

## 📂 Folder Layout

```
📁 data/                  # Optional local text data
🧠 main.py                # Core LLM logic
📝 processed_llm_files.txt
🔐 .env
📦 requirements.txt
```

---

Need this exported into an actual `README.md` file? I can generate that too. Just say the word.
