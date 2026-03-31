# Knowledge Garden 

## Introduction
This is an AI agent that can sort and give report automatically according to your daily thoughts(short inputs).

## Applications
Nowadays, many people read articles, listen to podcasts, and browse news every day, but this knowledge is fragmented and forgotten after a few months. The system acts like an "AI gardener", helping to organize, connect, and foster new ideas.

## Construction

```
knowledge-garden-ai/
├── api/                # Backend service powered by FastAPI/Flask
│   ├── main.py             # Entry point: API route definitions and orchestration
│   ├── agents/             # Core AI logic modules
│   │   ├── agent1.py    # Agent 1 ingestion: Content processing and summarization
│   │   ├── agent2.py    # Agent 2 synthesis: Relationship discovery and graph linking
│   │   └── agent3.py       # Agent 3 growth: Spaced repetition and output suggestions
│   └── requirements.txt    # Python dependencies
├── ./                 # Frontend application (Root for web files)
│   ├── index.html          # Homepage: Inspiration capture and input
│   ├── garden.html         # Visualization: Knowledge graph and card grid
│   ├── js/             # Static assets (Custom CSS, JS logic)
│   │   └──  lang.js    # JS file for multi-language
└── README.md               # Project documentation and setup guide
```

## Deployment
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fbaipin%2Fknowgarden&env=DEEPSEEK_BASE_URL,DEEPSEEK_API_KEY&envDefaults=%7B%22DEEPSEEK_BASE_URL%22%3A%22*input%20your%20openai%20url%2C%20not%20limited%20to%20deepseek%2C%20any%20model%20is%20ok.*%22%2C%22DEEPSEEK_API_KEY%22%3A%22*input%20your%20openai%20api%2C%20not%20limited%20to%20deepseek%2C%20any%20model%20is%20ok.*%22%7D&demo-title=Knowledge%20Garden&demo-url=https%3A%2F%2Fkg.baipon.com%2F)  
Or you can clone this repos and deploy it on your own server, you should set the following environment variables, or change it in `api/main.py`.
|  Environment Variables Keys   | Description  |
|  ----  | ----  |
| DEEPSEEK_BASE_URL  | input your openai url, not limited to deepseek, any model is ok. |
| DEEPSEEK_API_KEY  | input your openai api, not limited to deepseek, any model is ok. |

## Three Agent Roles:

- Ingestion Agent: Helps users input/summarize what they've learned today (supports links or text). → Automatic summary.

- Connection & Synthesis Agent: Discovers hidden connections between knowledge points, verifying/expanding them with code or search. (Uses simple embeddings or rules + code execution to create mind maps).

- Growth & Reminder Agent: Generates a "knowledge growth plan" (next review time, application scenarios, output ideas such as short articles).