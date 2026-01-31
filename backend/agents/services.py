import json
import logging
import re
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Multi-agent orchestrator that routes requests to specialized agents."""

    def __init__(self):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

    def process_message(self, conversation, message, dataset_id=None, document_id=None):
        """Process a user message and route to appropriate agent."""
        assistant_type = conversation.assistant_type

        # Build conversation history
        history = self._build_history(conversation)

        # Route to appropriate agent
        if assistant_type == 'data_analyst' or assistant_type == 'general':
            return self._handle_data_analyst(conversation, message, history,
                                            dataset_id, document_id)
        elif assistant_type == 'data_scientist':
            return self._handle_data_scientist(conversation, message, history, dataset_id)
        elif assistant_type == 'sql_expert':
            return self._handle_sql_expert(conversation, message, history, dataset_id)
        elif assistant_type == 'ml_engineer':
            return self._handle_ml_engineer(conversation, message, history, dataset_id)
        elif assistant_type == 'rag_assistant':
            return self._handle_rag_assistant(conversation, message, history, document_id)
        else:
            return self._handle_general(conversation, message, history,
                                       dataset_id, document_id)

    def _build_history(self, conversation, limit=20):
        """Build message history for context."""
        messages = conversation.messages.order_by('-created_at')[:limit]
        history = []
        for msg in reversed(messages):
            history.append({
                'role': msg.role,
                'content': msg.content,
            })
        return history

    def _handle_data_analyst(self, conversation, message, history,
                             dataset_id=None, document_id=None):
        """Handle data analyst queries - EDA, stats, general analysis."""
        from langchain_experimental.agents import create_pandas_dataframe_agent

        context_parts = []
        code_blocks = []
        visualizations = []

        # If we have a dataset, use DataFrame agent
        if dataset_id or conversation.dataset_id:
            did = dataset_id or conversation.dataset_id
            from datasets.models import Dataset
            from datasets.services import DatasetService
            dataset = Dataset.objects.get(id=did)
            df = DatasetService._read_dataframe(dataset)

            agent = create_pandas_dataframe_agent(
                self.llm, df,
                verbose=True,
                allow_dangerous_code=True,
                agent_type='openai-tools',
                prefix=f"""You are an expert data analyst assistant.
You are analyzing a dataset called "{dataset.name}" with {len(df)} rows and {len(df.columns)} columns.
Columns: {list(df.columns)}

Previous conversation:
{self._format_history(history)}

Provide thorough, insightful analysis. Include statistics, patterns, and actionable insights.""",
            )

            response = agent.invoke(message)
            output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

            return {
                'content': output,
                'agent_name': 'Data Analyst Agent',
                'tool_calls': [],
                'code_blocks': code_blocks,
                'visualizations': visualizations,
                'metadata': {'dataset': str(did)},
            }

        # Without dataset - use general LLM
        return self._handle_general(conversation, message, history, dataset_id, document_id)

    def _handle_data_scientist(self, conversation, message, history, dataset_id=None):
        """Handle data science queries - ML, statistics, modeling."""
        from langchain_experimental.agents import create_pandas_dataframe_agent

        if dataset_id or conversation.dataset_id:
            did = dataset_id or conversation.dataset_id
            from datasets.models import Dataset
            from datasets.services import DatasetService
            dataset = Dataset.objects.get(id=did)
            df = DatasetService._read_dataframe(dataset)

            agent = create_pandas_dataframe_agent(
                self.llm, df,
                verbose=True,
                allow_dangerous_code=True,
                agent_type='openai-tools',
                prefix=f"""You are an expert data scientist assistant.
Dataset: "{dataset.name}" ({len(df)} rows x {len(df.columns)} cols)
Columns: {list(df.columns)}

You can perform: EDA, feature engineering, hypothesis testing, ML modeling,
statistical analysis, and provide data-driven insights.

Previous conversation:
{self._format_history(history)}""",
            )

            response = agent.invoke(message)
            output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

            return {
                'content': output,
                'agent_name': 'Data Scientist Agent',
                'tool_calls': [],
                'code_blocks': [],
                'visualizations': [],
                'metadata': {'dataset': str(did)},
            }

        return self._handle_general(conversation, message, history)

    def _handle_sql_expert(self, conversation, message, history, dataset_id=None):
        """Handle SQL queries using natural language."""
        from langchain_community.utilities import SQLDatabase
        from langchain_community.agent_toolkits import SQLDatabaseToolkit
        from langchain.agents import create_sql_agent

        if dataset_id or conversation.dataset_id:
            did = dataset_id or conversation.dataset_id
            from datasets.models import Dataset
            from datasets.services import DatasetService
            dataset = Dataset.objects.get(id=did)
            df = DatasetService._read_dataframe(dataset)

            import sqlite3
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            conn = sqlite3.connect(tmp.name)
            table_name = dataset.name.replace(' ', '_').replace('-', '_').lower()
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()

            db = SQLDatabase.from_uri(f"sqlite:///{tmp.name}")
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type='openai-tools',
                prefix=f"""You are an expert SQL analyst.
You're querying a database with table "{table_name}".

Previous conversation:
{self._format_history(history)}""",
            )

            response = agent.invoke(message)
            output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

            return {
                'content': output,
                'agent_name': 'SQL Expert Agent',
                'tool_calls': [],
                'code_blocks': [],
                'visualizations': [],
                'metadata': {'dataset': str(did)},
            }

        return self._handle_general(conversation, message, history)

    def _handle_ml_engineer(self, conversation, message, history, dataset_id=None):
        """Handle ML engineering tasks."""
        return self._handle_data_scientist(conversation, message, history, dataset_id)

    def _handle_rag_assistant(self, conversation, message, history, document_id=None):
        """Handle RAG-based document Q&A."""
        if document_id:
            from datasets.models import Document
            from datasets.services import DocumentService
            document = Document.objects.get(id=document_id)
            result = DocumentService.query_document(document, message)

            return {
                'content': result['answer'],
                'agent_name': 'RAG Assistant',
                'tool_calls': [],
                'code_blocks': [],
                'visualizations': [],
                'metadata': {
                    'document': str(document_id),
                    'sources': result.get('sources', []),
                },
            }

        # Fallback to general with web search
        return self._handle_general_with_search(conversation, message, history)

    def _handle_general(self, conversation, message, history,
                        dataset_id=None, document_id=None):
        """Handle general queries."""
        system_prompt = """You are an expert AI Data Science and Analytics Assistant.
You can help with:
- Data analysis and exploration
- Statistical testing and hypothesis testing
- Machine learning model building
- Data visualization recommendations
- SQL queries and database analysis
- Document Q&A using RAG
- General data science questions

Provide thorough, well-structured answers with examples when appropriate."""

        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(history[-10:])
        messages.append({'role': 'user', 'content': message})

        response = self.llm.invoke(messages)

        return {
            'content': response.content,
            'agent_name': 'General Assistant',
            'tool_calls': [],
            'code_blocks': [],
            'visualizations': [],
            'metadata': {},
        }

    def _handle_general_with_search(self, conversation, message, history):
        """Handle general queries with web search fallback."""
        try:
            from langchain_community.utilities import SerpAPIWrapper
            from langchain.agents import load_tools, initialize_agent, AgentType

            tools = load_tools(['serpapi', 'llm-math'], llm=self.llm)
            agent = initialize_agent(
                tools, self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
            )
            response = agent.invoke(message)
            output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

            return {
                'content': output,
                'agent_name': 'Search Assistant',
                'tool_calls': [],
                'code_blocks': [],
                'visualizations': [],
                'metadata': {},
            }
        except Exception:
            return self._handle_general(conversation, message, history)

    def _format_history(self, history):
        """Format history for agent prefix."""
        if not history:
            return "No previous messages."
        lines = []
        for msg in history[-5:]:
            lines.append(f"{msg['role'].title()}: {msg['content'][:200]}")
        return "\n".join(lines)
