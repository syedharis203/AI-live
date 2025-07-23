from flask import Flask, request, render_template, jsonify, session, redirect
import os
import uuid
import pdfplumber
import requests
import json
import random
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Pinecone Setup ---
pc = Pinecone(api_key="pcsk_dH9vJ_3JrrNAHeGANYsmWDtv6gy6nXWkCuHBRh2dRXFs7ewn31ifjDYtnWWqzHaGkGwyW")
index_name = "reg"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(index_name)

# --- Gemini Setup ---
genai.configure(api_key="AIzaSyCpQvZUGclZCnL18Wh0fz_mqQDT6-_CBLE")

# --- Helper Functions ---
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"PDF parse warning on page {page.page_number}:", e)
    return text

def ollama_chat(prompt):
    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text.strip()

def embed_text(text):
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embedding = response["embedding"]
        if not any(embedding):
            print("⚠️ Skipping zero vector embedding.")
            return None
        return embedding
    except Exception as e:
        print("Embedding error:", str(e))
        return None

def extract_skills_with_ollama(resume_text, job_title):
    prompt = f"""
    Analyze this resume and extract the candidate's key technical skills.
    Group them into categories like Frontend, Backend, DevOps, Data, AI/ML, Other.

    Resume:\n{resume_text}
    Job Title (if any): {job_title}

    Output format:
    {{
        "Frontend": [...],
        "Backend": [...],
        "DevOps": [...],
        "Data": [...],
        "AI/ML": [...],
        "Other": [...]
    }}
    """
    try:
        response = ollama_chat(prompt)
        cleaned = response.strip().strip(" ")
        if cleaned.startswith("json"):
            cleaned = cleaned.replace("json", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"error": str(e)}

def generate_question_from_skill(skill):
    embed = embed_text(skill)
    if embed is None:
        return "Sorry, couldn't generate a question due to missing embedding."
    namespace = session.get("namespace", "interview")
    pinecone_results = pinecone_index.query(
        vector=embed,
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )
    context_chunks = [m['metadata']['text'] for m in pinecone_results['matches'] if 'text' in m['metadata']]
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI interviewer conducting a professional technical interview.

Ask one concise, relevant, and well-structured question based on the candidate's experience with {skill}. The question should be framed in a formal and polite tone, but still feel approachable.

Do **not** mention the candidate’s name or background. Do **not** explain or summarize the skill. Do **not** include any preamble, markdown, or additional context.

Start with a polite phrase like “Thank you for sharing.” or “Great, let’s continue.” to maintain a warm but professional tone.

Use the following reference material to inspire your question:
{context_text}

Return only the question as a plain string.
    """
    return ollama_chat(prompt)

def get_initial_greeting():
    prompt = "You're an HR interviewer. Greet the candidate warmly and ask them to introduce themselves."
    return ollama_chat(prompt)

def evaluate_answer(answer_text):
    eval_prompt = f"""
    Evaluate if the following answer was written by an AI or a human.
    Return a JSON like: {{"score": 85, "label": "AI-like"}}.
    Answer:\n{answer_text}
    """
    try:
        response = ollama_chat(eval_prompt)
        cleaned = response.strip().strip(" ")
        if cleaned.startswith("json"):
            cleaned = cleaned.replace("json", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"score": 0, "label": "Unknown"}

def calculate_summary_score(transcript):
    if not transcript:
        return {"avg_score": 0, "ai_count": 0, "human_count": 0}
    total_score = 0
    ai_like = 0
    human_like = 0
    for entry in transcript:
        eval = entry.get("evaluation", {})
        score = eval.get("score", 0)
        label = eval.get("label", "").lower()
        total_score += score
        if "ai" in label:
            ai_like += 1
        elif "human" in label:
            human_like += 1
    avg = round(total_score / len(transcript), 1)
    return {"avg_score": avg, "ai_count": ai_like, "human_count": human_like}

# --- Routes ---
@app.route('/')
def home():
    return render_template('combined.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    file = request.files['resume']
    job_title = request.form.get("job_title", "").strip()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath) if file.filename.endswith('.pdf') else open(filepath, 'r', encoding='utf-8').read()
    embedding = embed_text(text)
    if embedding is None:
        return jsonify({"status": "error", "message": "Embedding failed (empty or all-zero vector)."}), 400

    namespace_id = str(uuid.uuid4())
    session['namespace'] = namespace_id

    pinecone_index.upsert(
        vectors=[{
            "id": file.filename,
            "values": embedding,
            "metadata": {"text": text, "source": "resume", "job_title": job_title}
        }],
        namespace=namespace_id
    )

    extracted_skills = extract_skills_with_ollama(text, job_title)
    session['skills'] = extracted_skills
    session['transcript'] = []
    session['asked_skills'] = []
    session['intro_done'] = False
    session['question_count'] = 0
    return jsonify({"status": "ok"})

@app.route('/next_question', methods=['POST'])
def next_question():
    data = request.json
    user_answer = data.get("answer", "")
    transcript = session.get('transcript', [])
    asked_skills = session.get('asked_skills', [])
    skill_dict = session.get('skills', {})
    intro_done = session.get('intro_done', False)
    question_count = session.get('question_count', 0)

    evaluation = None
    if 'last_question' in session and user_answer:
        evaluation = evaluate_answer(user_answer)
        transcript.append({
            "q": session['last_question'],
            "a": user_answer,
            "evaluation": evaluation
        })
        session['transcript'] = transcript

    if not intro_done:
        q = get_initial_greeting()
        session['last_question'] = q
        session['intro_done'] = True
        return jsonify({"done": False, "question": q})

    if question_count >= 5:
        summary = calculate_summary_score(session.get('transcript', []))
        return jsonify({"done": True, "message": "Thanks for taking the interview!", "summary": summary})

    flat_skills = [(cat, skill) for cat, lst in skill_dict.items() for skill in lst if skill not in asked_skills]
    if not flat_skills:
        summary = calculate_summary_score(session.get('transcript', []))
        return jsonify({"done": True, "message": "Thanks for taking the interview!", "summary": summary})

    cat, next_skill = random.choice(flat_skills)
    asked_skills.append(next_skill)
    session['asked_skills'] = asked_skills
    q = generate_question_from_skill(next_skill)
    session['last_question'] = q
    session['question_count'] = question_count + 1

    return jsonify({"done": False, "question": q, "evaluation": evaluation})

@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    if request.method == 'POST':
        skill = request.form['skill']
        question = request.form['question']
        embed = embed_text(question)
        if embed:
            pinecone_index.upsert(
                vectors=[{
                    "id": f"{skill}-{random.randint(1000,9999)}",
                    "values": embed,
                    "metadata": {"text": question, "source": "admin-upload"}
                }],
                namespace="interview"
            )
        return redirect('/admin')

    return '''
    <h3>Admin Panel - Upload Question</h3>
    <form method="POST">
      Skill: <input type="text" name="skill"><br><br>
      Question:<br>
      <textarea name="question" rows="5" cols="50"></textarea><br><br>
      <input type="submit" value="Add">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
