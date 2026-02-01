import time
import json
import logging
import traceback
import pandas as pd
import numpy as np

from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)


def _get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model='gpt-4o-mini',
        openai_api_key=settings.OPENAI_API_KEY,
        openai_organization=settings.OPENAI_ORG_ID or None,
        temperature=0,
    )


def _load_dataset_df(dataset):
    from datasets.services import DatasetService
    return DatasetService._read_dataframe(dataset)


@shared_task(bind=True, queue='analysis_tasks', max_retries=2)
def run_eda_analysis(self, session_id):
    """Run EDA analysis using LangChain agent."""
    from analysis.models import AnalysisSession
    from langchain_experimental.agents import create_pandas_dataframe_agent

    session = AnalysisSession.objects.get(id=session_id)
    session.status = 'running'
    session.save()
    start_time = time.time()

    try:
        df = _load_dataset_df(session.dataset)
        llm = _get_llm()

        agent = create_pandas_dataframe_agent(
            llm, df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type='openai-tools',
            prefix=f"""You have access to a pandas DataFrame called `df` that is ALREADY LOADED in memory.
NEVER try to read files from disk. NEVER use pd.read_csv(). The data is already in `df`.
Dataset has {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}
Always use the `df` variable directly.""",
        )

        response = agent.invoke(session.query)
        output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

        # Also gather basic EDA info
        eda_result = {
            'answer': output,
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns},
            'describe': json.loads(df.describe(include='all').to_json()),
        }

        session.result = eda_result
        session.status = 'completed'
        session.execution_time = time.time() - start_time
        session.save()

        return eda_result

    except Exception as e:
        logger.error(f"EDA analysis error: {traceback.format_exc()}")
        session.status = 'failed'
        session.error_message = str(e)
        session.execution_time = time.time() - start_time
        session.save()
        raise


@shared_task(bind=True, queue='analysis_tasks', max_retries=2)
def run_visualization_analysis(self, session_id):
    """Generate visualization using LangChain agent."""
    from analysis.models import AnalysisSession, Visualization
    from langchain_experimental.agents import create_pandas_dataframe_agent
    import plotly.express as px
    import plotly.graph_objects as go

    session = AnalysisSession.objects.get(id=session_id)
    session.status = 'running'
    session.save()
    start_time = time.time()

    try:
        df = _load_dataset_df(session.dataset)
        llm = _get_llm()

        chart_type = session.parameters.get('chart_type', 'auto')
        query = session.query

        # Use LLM to determine visualization parameters
        viz_prompt = f"""Given this DataFrame with columns {list(df.columns)} and dtypes {dict(df.dtypes.astype(str))},
        the user wants: "{query}"
        Chart type preference: {chart_type}
        First 3 rows sample: {df.head(3).to_dict()}

        Return a JSON object with:
        - chart_type: one of (scatter, bar, line, histogram, box, violin, heatmap, kde, pie, pair)
        - x: column name for x-axis or category names (REQUIRED for most chart types)
        - y: column name for y-axis or values (optional for pie/histogram)
        - color: column for color coding (optional)
        - title: chart title
        - description: brief description of what this shows
        - agg: for pie/bar charts on categorical data, set to "count" if the user wants to see the distribution/count of categories. Leave null if data already has numeric values to plot.

        IMPORTANT for pie charts: "x" should be the column containing category labels/names. If the user wants category distribution, set agg to "count".

        Only return the JSON, nothing else."""

        response = llm.invoke(viz_prompt)
        content = response.content.strip()
        # Extract JSON from response
        if '```' in content:
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
        viz_config = json.loads(content)

        # Generate Plotly figure
        ct = viz_config.get('chart_type', 'scatter')
        x_col = viz_config.get('x')
        y_col = viz_config.get('y')
        color_col = viz_config.get('color')
        title = viz_config.get('title', session.query)

        fig = None
        agg = viz_config.get('agg')

        if ct == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        elif ct == 'bar':
            if agg == 'count' and x_col:
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.bar(counts, x=x_col, y='count', color=color_col, title=title)
            else:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
        elif ct == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
        elif ct == 'histogram':
            fig = px.histogram(df, x=x_col, color=color_col, title=title)
        elif ct == 'box':
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
        elif ct == 'violin':
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title)
        elif ct == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, title=title, color_continuous_scale='RdBu_r')
        elif ct == 'kde':
            fig = px.histogram(df, x=x_col, color=color_col, marginal='rug',
                             histnorm='probability density', title=title)
        elif ct == 'pie':
            if x_col and (agg == 'count' or not y_col):
                # Aggregate by counting categories
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, 'count']
                fig = px.pie(counts, names=x_col, values='count', title=title)
            elif x_col and y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=title)
            else:
                # Fallback: find the first categorical column and count
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    col = cat_cols[0]
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'count']
                    fig = px.pie(counts, names=col, values='count', title=title)
                else:
                    fig = px.pie(df, names=df.columns[0], title=title)
        elif ct == 'pair':
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            fig = px.scatter_matrix(df[numeric_cols], title=title)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=title)

        plotly_json = json.loads(fig.to_json()) if fig else {}

        # Save visualization
        viz = Visualization.objects.create(
            name=title,
            chart_type=ct,
            description=viz_config.get('description', ''),
            analysis=session,
            dataset=session.dataset,
            config=viz_config,
            plotly_json=plotly_json,
        )

        session.result = {
            'visualization_id': str(viz.id),
            'chart_type': ct,
            'config': viz_config,
            'plotly_json': plotly_json,
        }
        session.status = 'completed'
        session.execution_time = time.time() - start_time
        session.save()

        return session.result

    except Exception as e:
        logger.error(f"Visualization error: {traceback.format_exc()}")
        session.status = 'failed'
        session.error_message = str(e)
        session.execution_time = time.time() - start_time
        session.save()
        raise


@shared_task(bind=True, queue='ml_tasks', max_retries=1)
def run_ml_model_training(self, session_id):
    """Train ML model using LangChain agent."""
    from analysis.models import AnalysisSession, MLModel
    from langchain_experimental.agents import create_pandas_dataframe_agent

    session = AnalysisSession.objects.get(id=session_id)
    session.status = 'running'
    session.save()
    start_time = time.time()

    try:
        df = _load_dataset_df(session.dataset)
        llm = _get_llm()

        model_type = session.parameters.get('model_type', 'auto')
        target_column = session.parameters.get('target_column', '')

        # Use agent to train model
        ml_prompt = f"""{session.query}

        Dataset has columns: {list(df.columns)}
        Target column: {target_column if target_column else 'determine the best target'}
        Model type preference: {model_type}

        Steps:
        1. Clean the data (handle missing values)
        2. Prepare features and target
        3. Split into train/test (80/20)
        4. Train the model
        5. Evaluate with accuracy, precision, recall, f1-score
        6. Return the results as a summary

        Please do all steps and give me the accuracy and a summary."""

        agent = create_pandas_dataframe_agent(
            llm, df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type='openai-tools',
            prefix=f"""You have access to a pandas DataFrame called `df` that is ALREADY LOADED in memory.
NEVER try to read files from disk. NEVER use pd.read_csv(). The data is already in `df`.
Dataset has {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}
Always use the `df` variable directly.""",
        )

        response = agent.invoke(ml_prompt)
        output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

        # Save ML model record
        ml_model = MLModel.objects.create(
            name=session.name,
            model_type=model_type if model_type != 'auto' else 'custom',
            task_type='classification',
            description=output,
            dataset=session.dataset,
            analysis=session,
            target_column=target_column,
            feature_columns=list(df.columns),
            metrics={'summary': output},
        )

        session.result = {
            'model_id': str(ml_model.id),
            'summary': output,
            'model_type': model_type,
        }
        session.status = 'completed'
        session.execution_time = time.time() - start_time
        session.save()

        return session.result

    except Exception as e:
        logger.error(f"ML training error: {traceback.format_exc()}")
        session.status = 'failed'
        session.error_message = str(e)
        session.execution_time = time.time() - start_time
        session.save()
        raise


@shared_task(bind=True, queue='analysis_tasks', max_retries=2)
def run_hypothesis_test(self, session_id):
    """Run hypothesis test using LangChain agent."""
    from analysis.models import AnalysisSession, HypothesisTest
    from langchain_experimental.agents import create_pandas_dataframe_agent

    session = AnalysisSession.objects.get(id=session_id)
    session.status = 'running'
    session.save()
    start_time = time.time()

    try:
        df = _load_dataset_df(session.dataset)
        llm = _get_llm()

        agent = create_pandas_dataframe_agent(
            llm, df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type='openai-tools',
            prefix=f"""You have access to a pandas DataFrame called `df` that is ALREADY LOADED in memory.
NEVER try to read files from disk. NEVER use pd.read_csv(). The data is already in `df`.
Dataset has {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}
Always use the `df` variable directly.""",
        )

        prompt = f"""Perform this hypothesis test: {session.query}

        Use scipy.stats for the statistical test.
        Provide:
        1. Null hypothesis
        2. Alternative hypothesis
        3. Test statistic
        4. P-value
        5. Whether to reject the null hypothesis at 0.05 significance level
        6. Conclusion in plain English"""

        response = agent.invoke(prompt)
        output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

        test_type = session.parameters.get('test_type', 'custom')

        ht = HypothesisTest.objects.create(
            name=session.name,
            test_type=test_type if test_type != 'auto' else 'custom',
            dataset=session.dataset,
            analysis=session,
            conclusion=output,
            details={'raw_output': output},
        )

        session.result = {
            'test_id': str(ht.id),
            'conclusion': output,
        }
        session.status = 'completed'
        session.execution_time = time.time() - start_time
        session.save()

        return session.result

    except Exception as e:
        logger.error(f"Hypothesis test error: {traceback.format_exc()}")
        session.status = 'failed'
        session.error_message = str(e)
        session.execution_time = time.time() - start_time
        session.save()
        raise


@shared_task(bind=True, queue='analysis_tasks', max_retries=2)
def run_sql_query(self, session_id):
    """Run natural language SQL query."""
    from analysis.models import AnalysisSession
    from datasets.models import Dataset
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain.agents import create_sql_agent

    session = AnalysisSession.objects.get(id=session_id)
    session.status = 'running'
    session.save()
    start_time = time.time()

    try:
        llm = _get_llm()

        # If a dataset is provided, create an in-memory SQLite DB from it
        if session.dataset:
            import sqlite3
            import tempfile

            df = _load_dataset_df(session.dataset)
            tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            conn = sqlite3.connect(tmp.name)
            table_name = session.dataset.name.replace(' ', '_').replace('-', '_').lower()
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()

            db = SQLDatabase.from_uri(f"sqlite:///{tmp.name}")
        else:
            # Use the connection from parameters
            connection_id = session.parameters.get('connection_id', '')
            if connection_id:
                from datasets.models import DatabaseConnection
                conn_obj = DatabaseConnection.objects.get(id=connection_id)
                engine_map = {
                    'postgresql': 'postgresql',
                    'mysql': 'mysql+pymysql',
                    'sqlite': 'sqlite',
                    'mssql': 'mssql+pyodbc',
                }
                db_url = f"{engine_map[conn_obj.engine]}://{conn_obj.username}:{conn_obj.password}@{conn_obj.host}:{conn_obj.port}/{conn_obj.database}"
                db = SQLDatabase.from_uri(db_url)
            else:
                raise ValueError("Either dataset_id or connection_id is required")

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type='openai-tools',
        )

        response = agent.invoke(session.query)
        output = response.get('output', str(response)) if isinstance(response, dict) else str(response)

        session.result = {
            'answer': output,
            'query_type': 'sql',
        }
        session.status = 'completed'
        session.execution_time = time.time() - start_time
        session.save()

        return session.result

    except Exception as e:
        logger.error(f"SQL query error: {traceback.format_exc()}")
        session.status = 'failed'
        session.error_message = str(e)
        session.execution_time = time.time() - start_time
        session.save()
        raise
