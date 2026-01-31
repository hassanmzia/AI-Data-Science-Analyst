# AI Data Science Analyst

A professional-grade **multi-agent AI system** for data science and data analytics, built with Django, PostgreSQL, Node.js, React/TypeScript, and Docker Compose.

Based on the LangChain Data Science Assistant project by Mohammed Z Hassan.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (React/TypeScript)                   │
│                        Port: 3050                                │
│  ┌──────────┬──────────┬──────────┬───────────┬───────────────┐ │
│  │Dashboard │Assistant │ Datasets │ Analysis  │ Archive       │ │
│  │          │(Chat AI) │ (Upload/ │ (EDA/ML/  │ (Templates/   │ │
│  │          │          │  Import) │  SQL/Viz) │  Reuse)       │ │
│  └──────────┴──────────┴──────────┴───────────┴───────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼                               ▼
┌──────────────────┐           ┌──────────────────────┐
│  Django Backend   │           │  MCP/A2A Server      │
│    Port: 8050     │◄─────────►│    Port: 4050        │
│                   │           │                      │
│ • REST API        │           │ • MCP Protocol       │
│ • WebSocket       │           │ • A2A Protocol       │
│ • Celery Tasks    │           │ • Tool Registry      │
│ • Agent Orchest.  │           │ • Multi-Agent Mgmt   │
└───────┬───────────┘           └──────────────────────┘
        │
  ┌─────┼─────┬──────────┐
  ▼     ▼     ▼          ▼
┌────┐ ┌────┐ ┌───────┐ ┌──────────┐
│ PG │ │Redis│ │ChromaDB│ │ Celery   │
│5450│ │6350 │ │ 6340  │ │ Workers  │
└────┘ └────┘ └───────┘ └──────────┘
```

## Features

### From Original Notebook
- **Natural Language EDA** — Explore DataFrames using plain English (row counts, column info, missing values, statistics)
- **Data Visualization** — KDE plots, correlation heatmaps, distribution charts via natural language
- **Hypothesis Testing** — T-tests, chi-square, ANOVA from text descriptions
- **ML Model Building** — Logistic Regression, Random Forest, XGBoost, SVM, Neural Networks
- **SQL Database Querying** — Natural language to SQL translation
- **RAG Document Q&A** — Upload PDFs and ask questions with retrieval-augmented generation
- **Web Search Fallback** — SerpAPI integration for current events queries

### Additional Professional Features
- **Multi-Agent System** — Specialized agents (Data Analyst, Data Scientist, SQL Expert, ML Engineer, RAG Assistant) orchestrated via MCP and A2A protocols
- **Dataset Management** — Upload files, import from Kaggle, URLs, or external databases
- **Project Management** — Organize analyses into projects with AI-generated summaries
- **Project Archive** — Save old projects for future reuse, convert to reusable templates
- **Real-time Updates** — WebSocket support for streaming analysis results
- **Background Processing** — Celery workers for async ML training and analysis
- **API Documentation** — Auto-generated Swagger/OpenAPI docs at `/api/docs/`
- **Database Connections** — Connect to PostgreSQL, MySQL, SQLite, MS SQL databases
- **Export** — Export datasets and results to CSV, Excel, JSON, Parquet

## Services & Ports

| Service        | Port | Description                    |
|----------------|------|--------------------------------|
| Frontend       | 3050 | React/TypeScript UI            |
| Backend API    | 8050 | Django REST API + WebSocket    |
| MCP/A2A Server | 4050 | Model Context Protocol server  |
| PostgreSQL     | 5450 | Primary database               |
| Redis          | 6350 | Cache + Celery broker          |
| ChromaDB       | 6340 | Vector store for RAG           |
| Celery Flower  | 5550 | Task monitoring UI             |
| pgAdmin        | 5460 | Database admin UI              |

**Access the app at:** `http://172.168.1.95:3050`

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### Setup

1. **Clone and configure:**
   ```bash
   cd AI-Data-Science-Analyst
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start all services:**
   ```bash
   docker compose up -d --build
   ```

3. **Access the app:**
   - Main UI: http://172.168.1.95:3050
   - API Docs: http://172.168.1.95:8050/api/docs/
   - pgAdmin: http://172.168.1.95:5460
   - Flower: http://172.168.1.95:5550

4. **Create Django admin user (optional):**
   ```bash
   docker compose exec backend python manage.py createsuperuser
   ```

## Tech Stack

- **Backend:** Django 5.1, Django REST Framework, Celery, Daphne (ASGI)
- **Frontend:** React 18, TypeScript, Tailwind CSS, Zustand, React Query
- **AI/ML:** LangChain, OpenAI GPT, ChromaDB, FAISS, scikit-learn, XGBoost, PyTorch, TensorFlow
- **Protocols:** MCP (Model Context Protocol), A2A (Agent-to-Agent)
- **Database:** PostgreSQL 16, Redis 7
- **Infrastructure:** Docker Compose, Nginx

## Project Structure

```
├── docker-compose.yml          # Multi-container orchestration
├── .env.example                # Environment configuration template
├── backend/                    # Django backend
│   ├── core/                   # Django project settings
│   ├── datasets/               # Dataset & document management
│   ├── analysis/               # EDA, visualization, ML, hypothesis testing
│   ├── agents/                 # Multi-agent orchestrator & chat
│   ├── projects/               # Project management
│   └── archive/                # Project archive & templates
├── mcp-server/                 # Node.js MCP/A2A server
│   └── src/
│       ├── agents/             # Agent manager
│       ├── protocols/          # MCP & A2A routers
│       └── tools/              # Tool registry
└── frontend/                   # React/TypeScript frontend
    └── src/
        ├── pages/              # Page components
        ├── services/           # API clients
        ├── store/              # Zustand state management
        └── types/              # TypeScript types
```

## Author

Mohammed Z Hassan — Based on GA NLP Week 12 Final Project: LangChain Data Science Assistant
