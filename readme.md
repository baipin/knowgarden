# Knowledge Garden 

## Introduction
This is an AI agent that can sort and give report automatically according to your daily thoughts(short inputs).

## Applications
Nowadays, many people read articles, listen to podcasts, and browse news every day, but this knowledge is fragmented and forgotten after a few months. The system acts like an "AI gardener", helping to organize, connect, and foster new ideas.

## Construction

```
knowledge-garden-ai/
├── backend/                # Backend service powered by FastAPI/Flask
│   ├── main.py             # Entry point: API route definitions and orchestration
│   ├── agents/             # Core AI logic modules
│   │   ├── ingestion.py    # Agent 1: Content processing and summarization
│   │   ├── synthesis.py    # Agent 2: Relationship discovery and graph linking
│   │   └── growth.py       # Agent 3: Spaced repetition and output suggestions
│   ├── database.py         # Data persistence layer (SQLite/SQLAlchemy)
│   └── requirements.txt    # Python dependencies
├── frontend/               # Frontend application (Root for web files)
│   ├── index.html          # Homepage: Inspiration capture and input
│   ├── garden.html         # Visualization: Knowledge graph and card grid
│   ├── assets/             # Static assets (Custom CSS, JS logic)
│   └── components/         # Reusable MDUI 2 web components
└── README.md               # Project documentation and setup guide
```

## Three Agent Roles:

- Ingestion Agent: Helps users input/summarize what they've learned today (supports links or text). → Automatic summary.

- Connection & Synthesis Agent: Discovers hidden connections between knowledge points, verifying/expanding them with code or search. (Uses simple embeddings or rules + code execution to create mind maps).

- Growth & Reminder Agent: Generates a "knowledge growth plan" (next review time, application scenarios, output ideas such as short articles).