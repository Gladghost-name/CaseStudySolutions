# Case Study Solutions

## LocalBuka Conversational Assistant Prototype & Recommendation Logic

### ✓ Recommendation / Search Logic (Core Task)

A semantic search algorithm leveraging the `BAAI/bge-m3` Sentence Transformer. It embeds dynamic user preference profiles (including price tolerance, dietary limits, and spice preferences) and computes cosine similarity against pre-computed restaurant metadata vectors, returning a mathematically ranked threshold of top recommendations

#### How to Run

1. **Install the Required Dependencies:**
   Make sure you have `torch`, `sentence-transformers`, and `pandas` installed in your Python environment.

   ```bash
   pip install torch sentence-transformers pandas
   ```

2. **Execute the Script**:
   Run the file using your terminal:

   ```bash
   python run_recommendation.py
   ```

----

### ✓ Conversational Assistant Prototype

  A generative AI 🤖 chat interface powered by the `gemini-2.5-flash model`. The assistant is given a system instruction to anchor its responses exclusively to a proprietary JSON dataset of local restaurants and dishes, ensuring accurate guidance from user free-text message.
**How to Run**

1. **Install the Required Dependencies:**
   You will need Streamlit and the modern Google GenAI SDK installed in your environment.

   ```bash
   pip install streamlit google-genai
   ```

2. **Launch the App:**
Open your terminal, navigate to your localbuka_project folder, and start the Streamlit server:

```bash
streamlit run app.py
```

----
**Project Structure:**

Ensure your Python scripts and the JSON files are in the exact same directory, otherwise the app will crash trying to find the restaurant data.

```txt
case_study_solutions/
│
├── restaurants.json  
└── dishes.json
└── conversational_assistant.py
└── recommendation_task.py                 
```
