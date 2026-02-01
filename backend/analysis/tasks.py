import re
import time
import json
import logging
import traceback
import pandas as pd
import numpy as np

from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)


def _extract_metrics_from_text(text):
    """Extract numeric metrics from LLM output text into a structured dict."""
    metrics = {'summary': text}
    # Common metric patterns: "Accuracy: 0.81", "**Accuracy**: 81.01%", "F1 Score: 0.76"
    patterns = [
        (r'[*]*accuracy[*]*[:\s]+([0-9]+\.?[0-9]*)\s*%?', 'accuracy'),
        (r'[*]*precision[*]*[:\s]+([0-9]+\.?[0-9]*)\s*%?', 'precision'),
        (r'[*]*recall[*]*[:\s]+([0-9]+\.?[0-9]*)\s*%?', 'recall'),
        (r'[*]*f1[_ ]?score[*]*[:\s]+([0-9]+\.?[0-9]*)\s*%?', 'f1_score'),
        (r'[*]*mse[*]*[:\s]+([0-9]+\.?[0-9]*)', 'mse'),
        (r'[*]*mae[*]*[:\s]+([0-9]+\.?[0-9]*)', 'mae'),
        (r'[*]*rmse[*]*[:\s]+([0-9]+\.?[0-9]*)', 'rmse'),
        (r'[*]*r2[_ ]?score[*]*[:\s]+([0-9]+\.?[0-9]*)', 'r2_score'),
        (r'[*]*r[²2][*]*[:\s]+([0-9]+\.?[0-9]*)', 'r2_score'),
        (r'[*]*auc[*]*[:\s]+([0-9]+\.?[0-9]*)', 'auc'),
        (r'[*]*roc[_ ]?auc[*]*[:\s]+([0-9]+\.?[0-9]*)', 'roc_auc'),
        (r'[*]*loss[*]*[:\s]+([0-9]+\.?[0-9]*)', 'loss'),
        (r'[*]*test[_ ]?loss[*]*[:\s]+([0-9]+\.?[0-9]*)', 'test_loss'),
        (r'[*]*train[_ ]?loss[*]*[:\s]+([0-9]+\.?[0-9]*)', 'train_loss'),
    ]
    for pattern, key in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            # Normalize percentages to 0-1 range for accuracy/precision/recall/f1
            if key in ('accuracy', 'precision', 'recall', 'f1_score', 'auc', 'roc_auc') and val > 1:
                val = val / 100.0
            metrics[key] = round(val, 4)
    return metrics


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
            max_iterations=30,
            max_execution_time=300,
            handle_parsing_errors=True,
            prefix=f"""You have access to a pandas DataFrame called `df` that is ALREADY LOADED in memory.
NEVER try to read files from disk. NEVER use pd.read_csv(). The data is already in `df`.
Dataset has {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}
Dtypes: {dict(df.dtypes.astype(str))}
Always use the `df` variable directly.

You can import and use: sklearn, numpy, pandas. Do all work in a single code block when possible.
For the train/test split and model training, write the complete code in one execution.""",
        )

        try:
            response = agent.invoke(ml_prompt)
            output = response.get('output', str(response)) if isinstance(response, dict) else str(response)
        except Exception as agent_err:
            # Handle agent iteration limits or parse errors gracefully
            output = f"The ML agent encountered an issue: {str(agent_err)}"
            logger.warning(f"ML agent partial result: {agent_err}")

        # Save ML model record with extracted metrics
        extracted_metrics = _extract_metrics_from_text(output)
        ml_model = MLModel.objects.create(
            name=session.name,
            model_type=model_type if model_type != 'auto' else 'custom',
            task_type='classification',
            description=output,
            dataset=session.dataset,
            analysis=session,
            target_column=target_column,
            feature_columns=list(df.columns),
            metrics=extracted_metrics,
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


@shared_task(bind=True, queue='ml_tasks', max_retries=1)
def run_dl_model_training(self, session_id):
    """Train deep learning model using LangChain agent with PyTorch/TensorFlow."""
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
        framework = session.parameters.get('framework', 'pytorch')
        epochs = session.parameters.get('epochs', 50)
        batch_size = session.parameters.get('batch_size', 32)
        learning_rate = session.parameters.get('learning_rate', 0.001)
        task_type = session.parameters.get('task_type', 'auto')

        # Detect GPU availability
        gpu_available = False
        gpu_info = 'CPU only'
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_info = f'GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})'
                logger.info(f"DL training will use GPU: {gpu_info}")
        except Exception:
            pass

        # Build framework-specific instructions
        if framework == 'tensorflow':
            gpu_instruction = """
IMPORTANT: GPU is available. TensorFlow will automatically use it.
Print tf.config.list_physical_devices('GPU') to confirm GPU usage.""" if gpu_available else ""
            framework_instructions = f"""
Use TensorFlow/Keras for building the deep learning model.
Import: import tensorflow as tf, from tensorflow import keras, from tensorflow.keras import layers
Build the model using keras.Sequential or Functional API.
Compile with appropriate optimizer and loss function.
Use model.fit() for training with the specified epochs and batch_size.
After training, evaluate on the test set and print metrics.{gpu_instruction}"""
        else:
            gpu_instruction = f"""
IMPORTANT: GPU is available ({gpu_info}). You MUST use it.
Set device = torch.device('cuda') and move model and tensors to device with .to(device).
Print the device being used at the start.""" if gpu_available else """
Use device = torch.device('cpu')."""
            framework_instructions = f"""
Use PyTorch for building the deep learning model.
Import: import torch, import torch.nn as nn, import torch.optim as optim, from torch.utils.data import DataLoader, TensorDataset
Define the model as a class inheriting nn.Module.
{gpu_instruction}
Create a training loop with the specified epochs and batch_size.
After training, evaluate on the test set and print metrics."""

        # Model-specific architecture guidance
        model_guidance = {
            'cnn': 'Build a 1D CNN (Conv1d) for tabular data. Use Conv1d layers followed by MaxPool1d, flatten, and Dense layers.',
            'rnn': 'Build an RNN model. Reshape input to (batch, sequence_length, features). Use RNN/SimpleRNN layers.',
            'lstm': 'Build an LSTM model. Reshape input to (batch, sequence_length, features). Use LSTM layers with dropout.',
            'gru': 'Build a GRU model. Reshape input to (batch, sequence_length, features). Use GRU layers with dropout.',
            'transformer': 'Build a simple Transformer encoder for tabular data. Use multi-head attention and feedforward layers.',
            'autoencoder': 'Build an Autoencoder with encoder and decoder. Train to reconstruct input. Report reconstruction loss.',
            'gan': 'Build a simple GAN with generator and discriminator networks. Train adversarially.',
            'mlp': 'Build a Multi-Layer Perceptron with multiple hidden layers, ReLU activations, and dropout.',
            'resnet': 'Build a ResNet-style model with residual/skip connections for tabular data.',
        }.get(model_type, 'Choose the most appropriate deep learning architecture for this data and task.')

        dl_prompt = f"""{session.query}

        Dataset has columns: {list(df.columns)}
        Target column: {target_column if target_column else 'determine the best target'}
        Model type: {model_type}
        Task type: {task_type}
        Framework: {framework}
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {learning_rate}

        Architecture guidance: {model_guidance}

        {framework_instructions}

        Steps:
        1. Clean the data (handle missing values, encode categoricals with LabelEncoder or OneHotEncoder)
        2. Normalize/standardize numerical features
        3. Prepare features (X) and target (y)
        4. Split into train/test (80/20)
        5. Build the {model_type} deep learning model
        6. Train for {epochs} epochs with batch_size={batch_size} and learning_rate={learning_rate}
        7. Evaluate with appropriate metrics (accuracy for classification, MSE/MAE for regression)
        8. Print training history (loss per epoch) and final test metrics
        9. Return a detailed summary including model architecture, training loss curve info, and test metrics.

        IMPORTANT: Do all steps in as few code executions as possible.
        Print the final results clearly."""

        agent = create_pandas_dataframe_agent(
            llm, df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type='openai-tools',
            max_iterations=40,
            max_execution_time=600,
            handle_parsing_errors=True,
            prefix=f"""You have access to a pandas DataFrame called `df` that is ALREADY LOADED in memory.
NEVER try to read files from disk. NEVER use pd.read_csv(). The data is already in `df`.
Dataset has {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}
Dtypes: {dict(df.dtypes.astype(str))}
Always use the `df` variable directly.

You can import and use: torch, tensorflow, keras, sklearn, numpy, pandas.
Do all work in a single code block when possible.
For deep learning training, write the complete code in one execution.""",
        )

        try:
            response = agent.invoke(dl_prompt)
            output = response.get('output', str(response)) if isinstance(response, dict) else str(response)
        except Exception as agent_err:
            output = f"The DL agent encountered an issue: {str(agent_err)}"
            logger.warning(f"DL agent partial result: {agent_err}")

        # Determine actual framework used
        actual_framework = framework if framework != 'auto' else 'pytorch'

        # Save ML model record with extracted metrics
        extracted_metrics = _extract_metrics_from_text(output)
        ml_model = MLModel.objects.create(
            name=session.name,
            model_type=model_type if model_type != 'auto' else 'mlp',
            task_type=task_type if task_type != 'auto' else 'classification',
            framework=actual_framework,
            description=output,
            dataset=session.dataset,
            analysis=session,
            target_column=target_column,
            feature_columns=list(df.columns),
            metrics=extracted_metrics,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        session.result = {
            'model_id': str(ml_model.id),
            'summary': output,
            'model_type': model_type,
            'framework': actual_framework,
            'gpu_used': gpu_available,
            'gpu_info': gpu_info,
        }
        session.status = 'completed'
        session.execution_time = time.time() - start_time
        session.save()

        return session.result

    except Exception as e:
        logger.error(f"DL training error: {traceback.format_exc()}")
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
