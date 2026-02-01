# AI Data Science & Analyst Assistant

A professional-grade **multi-agent AI system** for data science and data analytics. Built with Django, React/TypeScript, Node.js (MCP/A2A), PostgreSQL, Redis, ChromaDB, and Docker Compose. Supports **GPU-accelerated deep learning** on NVIDIA RTX 5090 (Blackwell sm_120) with CUDA 12.8.

Built by Mohammed Z Hassan.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Multi-Agent System](#multi-agent-system)
- [MCP & A2A Protocols](#mcp--a2a-protocols)
- [ML/DL Training Engine](#mldl-training-engine)
- [RAG (Document Q&A)](#rag-document-qa)
- [Services & Ports](#services--ports)
- [Quick Start](#quick-start)
- [Environment Configuration](#environment-configuration)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [WebSocket Endpoints](#websocket-endpoints)
- [Frontend Pages](#frontend-pages)
- [Database Schema](#database-schema)
- [GPU Setup](#gpu-setup)
- [Docker Infrastructure](#docker-infrastructure)
- [Monitoring & Operations](#monitoring--operations)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

---

## Architecture Overview

```
                              ┌──────────────────┐
                              │   Web Browser     │
                              │   (User)          │
                              └────────┬──────────┘
                                       │ HTTPS
                              ┌────────▼──────────┐
                              │   Nginx (3050)     │
                              │   Reverse Proxy    │
                              │   Static Assets    │
                              └──┬──────────┬──────┘
                                 │          │
               ┌─────────────────┘          └────────────────────┐
               │ REST API / WebSocket                  MCP / A2A │
               ▼                                                 ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│   Django Backend (8050)  │◄────────────►│  MCP/A2A Server (4050)   │
│                          │  HTTP calls  │                          │
│  ┌────────────────────┐  │              │  ┌────────────────────┐  │
│  │ Django REST API     │  │              │  │ MCP Protocol       │  │
│  │ (drf-spectacular)   │  │              │  │ (tools/resources)  │  │
│  ├────────────────────┤  │              │  ├────────────────────┤  │
│  │ Django Channels     │  │              │  │ A2A Protocol       │  │
│  │ (WebSocket)         │  │              │  │ (agent cards/tasks)│  │
│  ├────────────────────┤  │              │  ├────────────────────┤  │
│  │ Agent Orchestrator  │  │              │  │ Tool Registry      │  │
│  │ (LangChain)         │  │              │  │ (8+ built-in)      │  │
│  ├────────────────────┤  │              │  ├────────────────────┤  │
│  │ 5 Django Apps       │  │              │  │ Agent Manager      │  │
│  │ datasets│analysis   │  │              │  │ (6 specialists)    │  │
│  │ agents │projects    │  │              │  └────────────────────┘  │
│  │ archive             │  │              └──────────────────────────┘
│  └────────────────────┘  │
└──────┬───────────────────┘
       │ Task Queue
       ▼
┌──────────────────────────┐     ┌───────────────┐   ┌───────────────┐
│  Celery Worker (GPU)     │────►│ PostgreSQL 16  │   │ ChromaDB      │
│  4 concurrent workers    │     │ (Port 5450)    │   │ (Port 6340)   │
│  Queues: default,        │     │ UUID, pg_trgm  │   │ Vector Store  │
│  ml_tasks, analysis_tasks│     └───────────────┘   │ RAG Embeddings│
├──────────────────────────┤                          └───────────────┘
│  Celery Beat             │     ┌───────────────┐   ┌───────────────┐
│  (Scheduled tasks)       │────►│ Redis 7       │   │ File Storage  │
├──────────────────────────┤     │ (Port 6350)   │   │ media_data    │
│  Flower (Port 5550)      │     │ Cache/Broker  │   │ model_data    │
│  (Task monitoring)       │     └───────────────┘   └───────────────┘
└──────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│  NVIDIA RTX 5090 (Blackwell sm_120) │ 32 GB VRAM │ CUDA 12.8      │
│  PyTorch Nightly cu128 │ TensorFlow 2.18 │ XGBoost GPU              │
└─────────────────────────────────────────────────────────────────────┘
```

A draw.io interactive architecture diagram is available at [`architecture-diagram.drawio`](./architecture-diagram.drawio).

---

## Features

### Core Data Science Capabilities

| Feature | Description |
|---------|-------------|
| **Natural Language EDA** | Explore DataFrames using plain English — row counts, column info, missing values, statistics, data profiling |
| **Data Visualization** | 11+ chart types — KDE, histogram, scatter, bar, line, heatmap, box, violin, pie, pair plot, custom Plotly |
| **ML Model Training** | 9+ traditional ML models with scikit-learn and XGBoost (GPU-accelerated) |
| **DL Model Training** | 10 deep learning architectures with PyTorch and TensorFlow on CUDA 12.8 |
| **Hypothesis Testing** | T-test, chi-square, ANOVA, Mann-Whitney, Pearson/Spearman correlation, Shapiro-Wilk |
| **SQL Querying** | Natural language to SQL translation against connected databases |
| **RAG Document Q&A** | Upload PDFs/DOCX/PPTX, semantic search with ChromaDB, answer questions with sources |
| **Web Search** | SerpAPI integration for real-time information retrieval |

### Platform Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent System** | 6 specialized AI agents orchestrated via MCP and A2A protocols |
| **Dataset Management** | Upload files, import from Kaggle, URLs, or external databases (PostgreSQL, MySQL, SQLite, MS SQL) |
| **Project Management** | Organize analyses into projects with notes, findings, and AI-generated summaries |
| **Project Archive** | Save completed projects, convert to reusable templates with methodology preservation |
| **Real-time Updates** | WebSocket streaming for chat and analysis progress |
| **Background Processing** | Celery workers with 3 dedicated queues for async ML training and analysis |
| **API Documentation** | Auto-generated OpenAPI/Swagger docs at `/api/docs/` |
| **Export** | Export datasets and results to CSV, Excel, JSON, Parquet |
| **GPU Acceleration** | NVIDIA CUDA 12.8 for PyTorch, TensorFlow, and XGBoost training |

---

## Multi-Agent System

The system uses 6 specialized AI agents, each with dedicated capabilities and tools:

| Agent | Capabilities | Tools |
|-------|-------------|-------|
| **Data Analyst** | EDA, statistics, data profiling, data cleaning | `eda_analysis`, `get_dataset_info`, `create_visualization` |
| **Data Scientist** | ML modeling, hypothesis testing, feature engineering, prediction | `train_ml_model`, `hypothesis_test`, `eda_analysis`, `create_visualization` |
| **SQL Expert** | Natural language to SQL, database analysis, data extraction | `sql_query`, `get_dataset_info` |
| **Visualization Expert** | Chart creation, dashboard design, visual analytics | `create_visualization`, `get_dataset_info`, `eda_analysis` |
| **RAG Assistant** | Document search, question answering, semantic retrieval | `query_documents`, `search_documents` |
| **ML Engineer** | Model architecture, hyperparameter tuning, optimization | `train_ml_model`, `train_dl_model`, `hypothesis_test`, `get_dataset_info` |

### Agent Orchestration Flow

1. User sends a natural language query via the Assistant page
2. The **AgentOrchestrator** in Django routes the query to the appropriate specialist agent
3. The agent selects tools via the **MCP protocol** from the Tool Registry
4. Tools call Django REST API endpoints to perform operations
5. Long-running tasks (ML/DL training) are dispatched to **Celery workers** (with GPU)
6. Results are streamed back to the user via **WebSocket**

---

## MCP & A2A Protocols

### Model Context Protocol (MCP)

The MCP server exposes tools, resources, and prompts following the MCP specification:

```
POST /mcp/tools/list         → Discover available tools
POST /mcp/tools/call         → Execute a tool with arguments
POST /mcp/resources/list     → List available data resources
POST /mcp/resources/read     → Read resource content
POST /mcp/prompts/list       → List prompt templates
POST /mcp/prompts/get        → Get prompt with parameters
```

**Built-in Tools (8+):**
- `eda_analysis` — Run exploratory data analysis on a dataset
- `create_visualization` — Generate Plotly charts from natural language
- `train_ml_model` — Train traditional ML models (scikit-learn/XGBoost)
- `train_dl_model` — Train deep learning models (PyTorch/TensorFlow)
- `hypothesis_test` — Run statistical hypothesis tests
- `sql_query` — Execute SQL queries on connected databases
- `get_dataset_info` — Retrieve dataset metadata and preview
- `query_documents` — RAG query against indexed documents

### Agent-to-Agent Protocol (A2A)

```
GET  /a2a/agents                    → List all agents with capabilities
GET  /a2a/agents/{agentId}/card     → Get agent capability card
POST /a2a/agents/{agentId}/tasks    → Create a task for an agent
GET  /a2a/tasks/{taskId}            → Get task status and result
POST /a2a/tasks/{taskId}/cancel     → Cancel a running task
```

Agent cards include capabilities, skills, supported modes, and endpoint information for agent discovery and collaboration.

---

## ML/DL Training Engine

### Machine Learning (scikit-learn + XGBoost)

| Model Type | Task | GPU |
|------------|------|-----|
| Logistic Regression | Classification | CPU |
| Random Forest | Classification / Regression | CPU |
| Decision Tree | Classification / Regression | CPU |
| K-Nearest Neighbors | Classification | CPU |
| Support Vector Machine | Classification | CPU |
| Gradient Boosting | Classification / Regression | CPU |
| Linear Regression | Regression | CPU |
| XGBoost | Classification / Regression | **GPU** (`tree_method='gpu_hist'`) |
| Neural Network (sklearn MLP) | Classification / Regression | CPU |

**ML Pipeline:** LangChain Pandas Agent generates training code automatically from natural language, executes it, extracts metrics via regex, and persists the model.

### Deep Learning (PyTorch + TensorFlow)

| Architecture | Task Types | Framework |
|--------------|-----------|-----------|
| CNN (Convolutional Neural Network) | Image/Text Classification | PyTorch / TensorFlow |
| RNN (Recurrent Neural Network) | Sequence Prediction | PyTorch / TensorFlow |
| LSTM (Long Short-Term Memory) | Sequence Prediction, Text | PyTorch / TensorFlow |
| GRU (Gated Recurrent Unit) | Sequence Prediction | PyTorch / TensorFlow |
| Transformer | Text Classification, Sequence | PyTorch / TensorFlow |
| Autoencoder | Anomaly Detection | PyTorch / TensorFlow |
| GAN (Generative Adversarial Network) | Generative | PyTorch / TensorFlow |
| MLP (Multi-Layer Perceptron) | Classification / Regression | PyTorch / TensorFlow |
| ResNet | Image Classification | PyTorch / TensorFlow |
| Custom | Any | PyTorch / TensorFlow |

**DL Pipeline:**
1. User selects architecture, framework, epochs, batch size, and learning rate
2. LLM (GPT-4o) generates a complete Python training script
3. Code is executed via `exec()` with a single-namespace dict (resolves class scoping issues)
4. Training runs on GPU (CUDA 12.8) with real-time status updates
5. Metrics are extracted, model is saved, and results are displayed

**Configurable Parameters:**
- Framework: PyTorch (recommended) or TensorFlow
- Epochs: 1–1000 (default: 50)
- Batch Size: 1–512 (default: 32)
- Learning Rate: 0.00001–1.0 (default: 0.001)
- Task Type: Classification, Regression, Image Classification, Text Classification, Sequence Prediction, Anomaly Detection, Generative

### Metrics Tracked

Classification: Accuracy, Precision, Recall, F1 Score, AUC-ROC
Regression: MSE, MAE, RMSE, R-squared
All: Training Loss, Validation Loss, Training History

---

## RAG (Document Q&A)

### Supported Document Types
- PDF (via PyPDF)
- DOCX (via python-docx)
- TXT (plain text)
- PPTX (via python-pptx)
- Markdown

### Ingestion Pipeline
1. **Upload** — Document uploaded via the Documents page
2. **Parse** — Text extracted using format-specific parsers
3. **Chunk** — Text split into overlapping chunks for context preservation
4. **Embed** — Chunks embedded using `text-embedding-3-large` (OpenAI)
5. **Store** — Embeddings stored in ChromaDB vector collections

### Query Pipeline
1. **User Query** — Natural language question
2. **Embed Query** — Same embedding model for consistency
3. **Vector Search** — Cosine similarity search in ChromaDB
4. **Context Build** — Top-K relevant chunks retrieved with metadata
5. **LLM Answer** — GPT-4o generates an answer grounded in the retrieved context

### Use Cases
- Research papers: Ask questions about methodology, findings, and conclusions
- Technical documentation: Query API docs, find configurations
- Financial reports: Extract key metrics, compare across documents
- Legal documents: Search contract clauses, compare terms

---

## Services & Ports

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| **Frontend** | ds-frontend | 3050 | React 18 + Nginx reverse proxy |
| **Backend API** | ds-backend | 8050 | Django 5.1 + Daphne ASGI server |
| **MCP/A2A Server** | ds-mcp-server | 4050 | Node.js + Express + WebSocket |
| **PostgreSQL** | ds-postgres | 5450 | Primary relational database |
| **Redis** | ds-redis | 6350 | Cache, Celery broker, Channel layers |
| **ChromaDB** | ds-chromadb | 6340 | Vector store for RAG embeddings |
| **Celery Worker** | ds-celery-worker | — | 4 concurrent workers with GPU |
| **Celery Beat** | ds-celery-beat | — | Scheduled task scheduler |
| **Flower** | ds-flower | 5550 | Celery task monitoring UI |
| **pgAdmin** | ds-pgadmin | 5460 | Database administration UI |

---

## Quick Start

### Prerequisites

- **Docker** & **Docker Compose v2**
- **NVIDIA Container Toolkit** (for GPU support)
- **NVIDIA GPU** with CUDA 12.8 compatible driver (570+)
- **OpenAI API key**
- 32+ GB RAM recommended
- 50+ GB disk space (Docker images + data)

### Setup

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd AI-Data-Science-Analyst
   cp .env.example .env
   ```

2. **Edit `.env` with your API keys:**
   ```bash
   # Required
   OPENAI_API_KEY=sk-your-openai-api-key

   # Optional
   SERPAPI_API_KEY=your-serpapi-key
   KAGGLE_USERNAME=your-kaggle-username
   KAGGLE_KEY=your-kaggle-key
   ```

3. **Start all services:**
   ```bash
   docker compose up -d --build
   ```

4. **Wait for services to be healthy** (backend runs migrations automatically):
   ```bash
   docker compose ps
   # All services should show "healthy" or "running"
   ```

5. **Access the application:**
   - **Main UI:** http://172.168.1.95:3050
   - **API Documentation:** http://172.168.1.95:8050/api/docs/
   - **Django Admin:** http://172.168.1.95:8050/admin/
   - **Celery Flower:** http://172.168.1.95:5550
   - **pgAdmin:** http://172.168.1.95:5460

6. **Create Django admin user (optional):**
   ```bash
   docker compose exec backend python manage.py createsuperuser
   ```

### Verify GPU Support

```bash
# Check GPU visibility inside the container
docker compose exec backend python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

# Expected output:
# CUDA: True
# GPU: NVIDIA GeForce RTX 5090
```

---

## Environment Configuration

Create a `.env` file from `.env.example` with the following variables:

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o and embeddings | `sk-...` |

### Optional - External APIs

| Variable | Description | Default |
|----------|-------------|---------|
| `SERPAPI_API_KEY` | SerpAPI key for web search | — |
| `KAGGLE_USERNAME` | Kaggle username for dataset import | — |
| `KAGGLE_KEY` | Kaggle API key | — |

### Django Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DJANGO_SECRET_KEY` | Django secret key | `change-me-to-a-random-secret-key` |
| `DJANGO_DEBUG` | Debug mode | `True` |
| `DJANGO_ALLOWED_HOSTS` | Allowed hosts | `172.168.1.95,localhost,127.0.0.1,0.0.0.0` |

### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_DB` | Database name | `datascience_db` |
| `POSTGRES_USER` | Database user | `ds_admin` |
| `POSTGRES_PASSWORD` | Database password | `ds_secure_password_2024` |

### Service Ports

| Variable | Description | Default |
|----------|-------------|---------|
| `FRONTEND_PORT` | Frontend port | `3050` |
| `BACKEND_PORT` | Backend API port | `8050` |
| `MCP_PORT` | MCP server port | `4050` |
| `POSTGRES_EXTERNAL_PORT` | PostgreSQL external port | `5450` |
| `REDIS_EXTERNAL_PORT` | Redis external port | `6350` |
| `CHROMA_PORT_EXTERNAL` | ChromaDB external port | `6340` |
| `FLOWER_PORT` | Flower monitoring port | `5550` |
| `PGADMIN_PORT` | pgAdmin port | `5460` |

### Frontend Build Args

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://172.168.1.95:8050` |
| `VITE_MCP_URL` | MCP server URL | `http://172.168.1.95:4050` |
| `VITE_WS_URL` | WebSocket URL | `ws://172.168.1.95:8050/ws` |

---

## Technology Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| Django | 5.1.3 | Web framework |
| Django REST Framework | 3.15.2 | REST API |
| Daphne | 4.1.2 | ASGI server (HTTP + WebSocket) |
| Django Channels | 4.2.0 | WebSocket support |
| Celery | 5.4.0 | Async task processing |
| django-celery-beat | 2.7.0 | Scheduled tasks |
| drf-spectacular | 0.28.0 | OpenAPI documentation |
| django-cors-headers | 4.6.0 | CORS support |
| django-filter | 24.3 | API filtering |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.3.1 | UI framework |
| TypeScript | 5.7.2 | Type safety |
| Vite | 6.0.3 | Build tool |
| Tailwind CSS | 3.4.16 | Styling |
| Zustand | 5.0.1 | State management |
| React Query | 5.61.3 | Server state + caching |
| React Router | 7.0.2 | Client-side routing |
| Plotly.js | — | Interactive charts |
| Axios | — | HTTP client |

### AI / Machine Learning

| Technology | Version | Purpose |
|------------|---------|---------|
| OpenAI GPT-4o | — | LLM for code generation, chat, RAG |
| LangChain | 0.3.9 | Agent framework, tool calling |
| LangChain Experimental | 0.3.3 | Pandas DataFrame agent |
| scikit-learn | 1.5.2 | Traditional ML models |
| XGBoost | 2.1.3 | Gradient boosting (GPU) |
| PyTorch | Nightly cu128 | Deep learning (GPU) |
| TensorFlow | 2.18.0 | Deep learning (GPU) |
| ChromaDB | 0.5.23 | Vector store for RAG |
| text-embedding-3-large | — | Document embeddings |

### Data Science

| Technology | Version | Purpose |
|------------|---------|---------|
| Pandas | 2.2.3 | DataFrames |
| NumPy | 1.26.4 | Numerical computing |
| SciPy | 1.14.1 | Scientific computing |
| Matplotlib | 3.9.2 | Static charts |
| Seaborn | 0.13.2 | Statistical visualizations |
| Plotly | 5.24.1 | Interactive charts |
| Statsmodels | 0.14.4 | Statistical models |

### Infrastructure

| Technology | Version | Purpose |
|------------|---------|---------|
| Docker Compose | v2 | Container orchestration |
| NVIDIA CUDA | 12.8 | GPU compute |
| PostgreSQL | 16 | Relational database |
| Redis | 7 | Cache + message broker |
| Nginx | Alpine | Reverse proxy |
| Rocky Linux | 9 | Container base OS |

---

## Project Structure

```
AI-Data-Science-Analyst/
├── docker-compose.yml                 # 10 service orchestration
├── .env.example                       # Environment template
├── architecture-diagram.drawio        # Draw.io architecture diagram
├── Technical_Architecture.pptx        # Architecture PowerPoint slides
├── README.md                          # This file
│
├── docker/
│   └── postgres/
│       └── init.sql                   # DB init (uuid-ossp, pg_trgm, schemas)
│
├── backend/                           # Django 5.1 Backend
│   ├── Dockerfile                     # CUDA 12.8 + Python 3.12 + PyTorch
│   ├── requirements.txt               # 74 Python dependencies
│   ├── manage.py                      # Django CLI
│   │
│   ├── core/                          # Django project config
│   │   ├── settings.py               # Apps, middleware, databases, channels
│   │   ├── urls.py                   # Root URL patterns
│   │   ├── asgi.py                   # Daphne ASGI application
│   │   ├── wsgi.py                   # WSGI application
│   │   └── celery_app.py            # Celery config + task routing
│   │
│   ├── datasets/                      # Dataset & Document management
│   │   ├── models.py                 # Dataset, DatabaseConnection, Document
│   │   ├── views.py                  # Upload, import, preview, statistics, export
│   │   ├── services.py              # DatasetService, DocumentService
│   │   ├── serializers.py           # REST serializers
│   │   └── urls.py                  # API routes
│   │
│   ├── analysis/                      # Analysis & ML/DL engine
│   │   ├── models.py                 # AnalysisSession, Visualization, MLModel, HypothesisTest
│   │   ├── views.py                  # run_eda, run_ml, run_dl, run_visualization, etc.
│   │   ├── tasks.py                  # Celery tasks for async processing
│   │   ├── serializers.py           # Analysis serializers
│   │   └── urls.py                  # API routes
│   │
│   ├── agents/                        # Multi-agent orchestration
│   │   ├── models.py                 # Conversation, Message, AgentConfig
│   │   ├── views.py                  # ConversationViewSet with chat action
│   │   ├── services.py              # AgentOrchestrator (6 agent handlers)
│   │   ├── consumers.py             # WebSocket consumers (ChatConsumer, AnalysisConsumer)
│   │   ├── routing.py               # WebSocket URL routing
│   │   ├── serializers.py           # Chat serializers
│   │   └── urls.py                  # API routes
│   │
│   ├── projects/                      # Project management
│   │   ├── models.py                 # Project, ProjectNote
│   │   ├── views.py                  # ProjectViewSet
│   │   └── urls.py                  # API routes
│   │
│   └── archive/                       # Archive & templates
│       ├── models.py                 # ArchivedProject, ArchivedAnalysis, ProjectTemplate
│       ├── views.py                  # Archive viewsets
│       └── urls.py                  # API routes
│
├── mcp-server/                        # Node.js MCP/A2A Server
│   ├── Dockerfile                     # Node 20-alpine
│   ├── package.json                   # 13 prod + 7 dev dependencies
│   ├── tsconfig.json                 # TypeScript config
│   └── src/
│       ├── index.ts                  # Express server (4050) + WebSocket
│       ├── agents/
│       │   └── agent-manager.ts      # 6 agent definitions + task management
│       ├── protocols/
│       │   ├── mcp-router.ts        # MCP protocol endpoints
│       │   └── a2a-router.ts        # A2A protocol endpoints
│       ├── tools/
│       │   └── registry.ts          # 8+ tool definitions + execution
│       └── utils/
│           └── logger.ts            # Winston logger
│
└── frontend/                          # React 18 Frontend
    ├── Dockerfile                     # Multi-stage: Node build -> Nginx
    ├── nginx.conf                    # Reverse proxy, SPA routing, gzip
    ├── package.json                  # 15 prod + 8 dev dependencies
    ├── vite.config.ts               # Vite build config
    ├── tailwind.config.cjs          # Tailwind CSS config
    └── src/
        ├── index.tsx                # React entry point
        ├── App.tsx                  # Root component with React Router
        ├── types/
        │   └── index.ts            # 180+ lines of TypeScript interfaces
        ├── pages/
        │   ├── Dashboard.tsx       # Overview metrics & quick actions
        │   ├── AssistantPage.tsx   # Multi-agent AI chat interface
        │   ├── DatasetsPage.tsx    # Dataset upload, import, management
        │   ├── AnalysisPage.tsx    # EDA, ML, visualization, SQL
        │   ├── VisualizationPage.tsx # Chart gallery & creation
        │   ├── MLModelsPage.tsx    # ML/DL model training & management
        │   ├── DocumentsPage.tsx   # RAG document upload & query
        │   ├── ProjectsPage.tsx    # Project organization
        │   ├── ArchivePage.tsx     # Archive & templates
        │   └── SettingsPage.tsx    # Configuration
        ├── components/
        │   └── PlotlyChart.tsx     # Plotly visualization component
        ├── services/
        │   └── api.ts              # Axios API client (100+ lines)
        └── store/
            └── index.ts            # Zustand state management
```

---

## API Reference

The full API documentation is auto-generated and available at `/api/docs/` (Swagger UI) when the backend is running.

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/datasets/datasets/` | List all datasets |
| `POST` | `/api/datasets/datasets/upload/` | Upload a dataset file |
| `POST` | `/api/datasets/datasets/import_kaggle/` | Import from Kaggle |
| `POST` | `/api/datasets/datasets/import_url/` | Import from URL |
| `POST` | `/api/datasets/datasets/import_database/` | Import from external DB |
| `GET` | `/api/datasets/datasets/{id}/preview/` | Preview first N rows |
| `GET` | `/api/datasets/datasets/{id}/statistics/` | Descriptive statistics |
| `GET` | `/api/datasets/datasets/{id}/column_info/` | Column metadata |
| `POST` | `/api/datasets/datasets/{id}/export/` | Export to CSV/Excel/JSON/Parquet |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analysis/sessions/run_eda/` | Run exploratory data analysis |
| `POST` | `/api/analysis/sessions/run_visualization/` | Generate visualization |
| `POST` | `/api/analysis/sessions/run_ml/` | Train ML model |
| `POST` | `/api/analysis/sessions/run_dl/` | Train DL model (GPU) |
| `POST` | `/api/analysis/sessions/run_hypothesis_test/` | Run hypothesis test |
| `POST` | `/api/analysis/sessions/run_sql/` | Execute SQL query |
| `GET` | `/api/analysis/sessions/{id}/status_check/` | Check task status |
| `GET` | `/api/analysis/models/` | List trained models |
| `POST` | `/api/analysis/models/{id}/predict/` | Make predictions |

### Agents & Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/agents/conversations/` | List conversations |
| `POST` | `/api/agents/conversations/` | Create conversation |
| `POST` | `/api/agents/conversations/{id}/chat/` | Send message |
| `POST` | `/api/agents/conversations/{id}/archive/` | Archive conversation |

### Documents (RAG)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/datasets/documents/upload/` | Upload document for RAG |
| `POST` | `/api/datasets/documents/{id}/query/` | Query document with RAG |

### Projects & Archive

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/api/projects/projects/` | List/create projects |
| `GET/POST` | `/api/archive/archived-projects/` | List/create archives |
| `GET/POST` | `/api/archive/templates/` | List/create templates |

---

## WebSocket Endpoints

| Endpoint | Purpose |
|----------|---------|
| `ws://.../ws/chat/{conversation_id}/` | Real-time AI chat with streaming responses |
| `ws://.../ws/analysis/{session_id}/` | Real-time analysis progress updates |

Events: `connection_established`, `message`, `typing`, `analysis_update`, `error`

---

## Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| **Dashboard** | `/` | Overview metrics, recent activity, quick actions |
| **AI Assistant** | `/assistant` | Multi-agent chat interface with conversation history |
| **Datasets** | `/datasets` | Upload, import (Kaggle/URL/DB), preview, statistics |
| **Analysis** | `/analysis` | Run EDA, ML, DL, hypothesis tests, SQL queries |
| **Visualization** | `/visualization` | Create and browse interactive Plotly charts |
| **ML Models** | `/ml-models` | Train ML/DL models, configure GPU, view metrics |
| **Documents** | `/documents` | Upload documents for RAG, query with semantic search |
| **Projects** | `/projects` | Organize work into projects with notes |
| **Archive** | `/archive` | Browse archived projects and reusable templates |
| **Settings** | `/settings` | Application configuration |

---

## Database Schema

### Key Models

**datasets app:**
- `Dataset` — File metadata, row/column counts, statistics, preview data
- `DatabaseConnection` — External database connection configs
- `Document` — RAG document metadata, embedding status, chunk count

**analysis app:**
- `AnalysisSession` — EDA/ML/DL/SQL session with status, results, code
- `Visualization` — Plotly chart config, rendered images
- `MLModel` — Trained model metadata, metrics, hyperparameters, model file
- `HypothesisTest` — Test type, statistic, p-value, conclusion

**agents app:**
- `Conversation` — Chat session with agent type and context
- `Message` — Individual messages with role, content, tool calls
- `AgentConfig` — Agent settings, model, temperature, tools

**projects app:**
- `Project` — Project metadata, datasets, documents, AI summary
- `ProjectNote` — Notes, findings, todos, decisions

**archive app:**
- `ArchivedProject` — Preserved project with methodology and findings
- `ArchivedAnalysis` — Individual analysis results
- `ProjectTemplate` — Reusable workflow templates

All models use **UUID primary keys** for security and distributed compatibility.

---

## GPU Setup

### Hardware Requirements

- NVIDIA GPU with Compute Capability 5.0+ (recommended: 7.0+)
- NVIDIA Driver 570+ for CUDA 12.8
- NVIDIA Container Toolkit installed

### Current Configuration

| Component | Version | Notes |
|-----------|---------|-------|
| GPU | NVIDIA RTX 5090 | Blackwell architecture (sm_120) |
| VRAM | 32 GB | Sufficient for large DL models |
| Driver | 570.86.10 | Supports CUDA 12.8 |
| CUDA | 12.8 | Latest stable |
| cuDNN | Runtime | Bundled in Docker image |
| Base Image | `nvidia/cuda:12.8.0-cudnn-runtime-rockylinux9` | Rocky Linux 9 |
| Python | 3.12 | Required by PyTorch cu128 |
| PyTorch | Nightly cu128 | Only version supporting sm_120 |
| TensorFlow | 2.18.0 | CUDA 12.8 compatible |

### NVIDIA Container Toolkit Setup

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.8.0-base-rockylinux9 nvidia-smi
```

### GPU Allocation in Docker Compose

The `backend` and `celery-worker` services have GPU access:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Why PyTorch Nightly?

The RTX 5090 uses the **Blackwell architecture (sm_120)**. Stable PyTorch releases only support up to sm_90 (Ada Lovelace). The **nightly cu128 build** is required for sm_120 support. This will change once PyTorch releases a stable version with Blackwell support.

---

## Docker Infrastructure

### Docker Compose Services

10 containers orchestrated on the `ds-analyst-network` bridge network:

| Service | Base Image | Build | GPU |
|---------|-----------|-------|-----|
| postgres | `postgres:16-alpine` | Pre-built | No |
| pgadmin | `dpage/pgadmin4:latest` | Pre-built | No |
| redis | `redis:7-alpine` | Pre-built | No |
| chromadb | `chromadb/chroma:latest` | Pre-built | No |
| backend | `nvidia/cuda:12.8.0-cudnn-runtime-rockylinux9` | Custom | **Yes** |
| celery-worker | Same as backend | Custom | **Yes** |
| celery-beat | Same as backend | Custom | No |
| flower | Same as backend | Custom | No |
| mcp-server | `node:20-alpine` | Custom | No |
| frontend | `node:20-alpine` → `nginx:alpine` | Multi-stage | No |

### Persistent Volumes

| Volume | Purpose |
|--------|---------|
| `postgres_data` | PostgreSQL database files |
| `pgadmin_data` | pgAdmin configuration |
| `redis_data` | Redis RDB snapshots |
| `chroma_data` | ChromaDB vector embeddings |
| `media_data` | Uploads, exports, visualizations, documents |
| `model_data` | Trained ML/DL model files |

### Service Dependencies & Startup Order

```
postgres (healthy) ─┐
redis (healthy) ────┼──► backend (healthy) ──► celery-worker
chromadb (started) ─┘         │                celery-beat
                              │                flower
redis (healthy) ─────────────►│
chromadb (started) ───────────┼──► mcp-server
                              │
backend ──────────────────────┼──► frontend
mcp-server ───────────────────┘
```

### Healthchecks

| Service | Check | Interval |
|---------|-------|----------|
| PostgreSQL | `pg_isready` | 10s |
| Redis | `redis-cli ping` | 10s |
| Backend | HTTP `/api/health/` | 10s (30s startup) |

### Celery Task Queues

| Queue | Purpose |
|-------|---------|
| `default` | General agent tasks |
| `ml_tasks` | ML/DL model training |
| `analysis_tasks` | EDA and visualization |

---

## Monitoring & Operations

### Access Points

| Service | URL |
|---------|-----|
| Frontend UI | http://172.168.1.95:3050 |
| API Documentation | http://172.168.1.95:8050/api/docs/ |
| Django Admin | http://172.168.1.95:8050/admin/ |
| Celery Flower | http://172.168.1.95:5550 |
| pgAdmin | http://172.168.1.95:5460 |

### Common Commands

```bash
# Start all services
docker compose up -d --build

# View logs
docker compose logs -f backend
docker compose logs -f celery-worker

# Check service health
docker compose ps

# Run Django management commands
docker compose exec backend python manage.py createsuperuser
docker compose exec backend python manage.py makemigrations
docker compose exec backend python manage.py migrate

# Check GPU in container
docker compose exec backend python -c "import torch; print(torch.cuda.is_available())"
docker compose exec celery-worker nvidia-smi

# Restart specific service
docker compose restart backend
docker compose restart celery-worker

# Stop all services
docker compose down

# Stop and remove volumes (data loss!)
docker compose down -v
```

### Backend Auto-Startup

The backend container automatically runs on startup:
1. `python manage.py makemigrations --noinput`
2. `python manage.py migrate --noinput`
3. `python manage.py collectstatic --noinput`
4. `daphne -b 0.0.0.0 -p 8050 core.asgi:application`

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA driver on host
nvidia-smi

# Verify Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-rockylinux9 nvidia-smi

# If "could not select device driver nvidia":
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### PyTorch sm_120 Warning

If you see `NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible`, ensure you're using PyTorch **nightly cu128**, not a stable release:

```bash
# In Dockerfile, use:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Backend Health Check Failing

```bash
# Check backend logs
docker compose logs backend

# The health check has a 30s start_period and 30 retries
# Wait for migrations to complete
docker compose exec backend python -c "import django; django.setup(); print('OK')"
```

### Celery Tasks Not Processing

```bash
# Check worker status
docker compose logs celery-worker

# Verify Redis connectivity
docker compose exec redis redis-cli ping

# Monitor via Flower
open http://172.168.1.95:5550
```

### ChromaDB Connection Issues

```bash
# ChromaDB doesn't use a health check (disabled)
# Verify it's running
docker compose logs chromadb
curl http://172.168.1.95:6340/api/v1/heartbeat
```

### Frontend Blank Page

```bash
# Check Nginx logs
docker compose logs frontend

# Verify build args are correct in docker-compose.yml
# VITE_API_URL should point to backend
# VITE_WS_URL should use ws:// protocol
```

---

## Author

**Mohammed Z Hassan**

A production-grade, GPU-accelerated, multi-agent data science platform with MCP/A2A protocol support, real-time WebSocket communication, and containerized deployment.
