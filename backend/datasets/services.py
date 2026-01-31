import io
import os
import json
import logging
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)


class DatasetService:
    """Service layer for dataset operations."""

    @staticmethod
    def _read_dataframe(dataset) -> pd.DataFrame:
        """Read a dataset into a pandas DataFrame."""
        from .models import Dataset

        if dataset.file:
            path = dataset.file.path
            fmt = dataset.file_format
        elif dataset.url:
            path = dataset.url
            fmt = dataset.file_format
        else:
            raise ValueError("No file or URL associated with this dataset")

        if fmt == 'csv':
            return pd.read_csv(path)
        elif fmt == 'xlsx':
            return pd.read_excel(path)
        elif fmt == 'json':
            return pd.read_json(path)
        elif fmt == 'parquet':
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path)

    @staticmethod
    def _compute_metadata(df: pd.DataFrame) -> dict:
        """Compute comprehensive metadata for a DataFrame."""
        columns_info = {}
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
            }
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                col_info.update({
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None,
                    'median': float(df[col].median()) if not df[col].isnull().all() else None,
                })
            columns_info[col] = col_info

        missing = {col: int(df[col].isnull().sum()) for col in df.columns}
        dtypes_info = {col: str(df[col].dtype) for col in df.columns}

        # Preview (first 10 rows)
        preview = json.loads(df.head(10).to_json(orient='records', date_format='iso'))

        # Statistics
        try:
            stats = json.loads(df.describe(include='all').to_json())
        except Exception:
            stats = {}

        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns_info': columns_info,
            'missing_values': missing,
            'dtypes': dtypes_info,
            'preview_data': preview,
            'statistics': stats,
        }

    @classmethod
    def create_from_upload(cls, file, name='', description='', tags=None, owner=None):
        """Create dataset from uploaded file."""
        from .models import Dataset

        filename = file.name
        ext = Path(filename).suffix.lower().lstrip('.')
        format_map = {'csv': 'csv', 'xlsx': 'xlsx', 'xls': 'xlsx',
                     'json': 'json', 'parquet': 'parquet'}
        file_format = format_map.get(ext, 'csv')

        dataset = Dataset(
            name=name or filename,
            description=description,
            source='upload',
            file_format=file_format,
            file=file,
            file_size=file.size,
            tags=tags or [],
            owner=owner,
        )
        dataset.save()

        # Compute metadata
        try:
            df = cls._read_dataframe(dataset)
            metadata = cls._compute_metadata(df)
            for key, value in metadata.items():
                setattr(dataset, key, value)
            dataset.save()
        except Exception as e:
            logger.error(f"Metadata computation error: {e}")

        return dataset

    @classmethod
    def import_from_kaggle(cls, dataset_ref, name='', description='', owner=None):
        """Import dataset from Kaggle."""
        from .models import Dataset

        os.environ['KAGGLE_USERNAME'] = settings.KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = settings.KAGGLE_KEY

        import kaggle
        with tempfile.TemporaryDirectory() as tmpdir:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset_ref, path=tmpdir, unzip=True)

            # Find first CSV or relevant file
            files = list(Path(tmpdir).glob('**/*'))
            data_files = [f for f in files if f.suffix.lower() in
                         ['.csv', '.xlsx', '.json', '.parquet']]

            if not data_files:
                raise ValueError("No supported data files found in Kaggle dataset")

            file_path = data_files[0]
            ext = file_path.suffix.lower().lstrip('.')
            format_map = {'csv': 'csv', 'xlsx': 'xlsx', 'json': 'json', 'parquet': 'parquet'}

            with open(file_path, 'rb') as f:
                content = f.read()

            dataset = Dataset(
                name=name or f"Kaggle: {dataset_ref}",
                description=description or f"Imported from Kaggle: {dataset_ref}",
                source='kaggle',
                file_format=format_map.get(ext, 'csv'),
                kaggle_ref=dataset_ref,
                file_size=len(content),
                owner=owner,
            )
            dataset.file.save(file_path.name, ContentFile(content))
            dataset.save()

            # Compute metadata
            try:
                df = cls._read_dataframe(dataset)
                metadata = cls._compute_metadata(df)
                for key, value in metadata.items():
                    setattr(dataset, key, value)
                dataset.save()
            except Exception as e:
                logger.error(f"Kaggle metadata error: {e}")

            return dataset

    @classmethod
    def import_from_url(cls, url, name='', description='', owner=None):
        """Import dataset from URL."""
        from .models import Dataset
        import httpx

        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()

        filename = url.split('/')[-1].split('?')[0] or 'data.csv'
        ext = Path(filename).suffix.lower().lstrip('.')
        format_map = {'csv': 'csv', 'xlsx': 'xlsx', 'json': 'json', 'parquet': 'parquet'}

        dataset = Dataset(
            name=name or filename,
            description=description or f"Imported from {url}",
            source='url',
            file_format=format_map.get(ext, 'csv'),
            url=url,
            file_size=len(response.content),
            owner=owner,
        )
        dataset.file.save(filename, ContentFile(response.content))
        dataset.save()

        try:
            df = cls._read_dataframe(dataset)
            metadata = cls._compute_metadata(df)
            for key, value in metadata.items():
                setattr(dataset, key, value)
            dataset.save()
        except Exception as e:
            logger.error(f"URL import metadata error: {e}")

        return dataset

    @classmethod
    def import_from_database(cls, connection_id, query, name='', owner=None):
        """Import data from external database."""
        from .models import Dataset, DatabaseConnection
        from sqlalchemy import create_engine

        conn = DatabaseConnection.objects.get(id=connection_id)

        engine_map = {
            'postgresql': 'postgresql',
            'mysql': 'mysql+pymysql',
            'sqlite': 'sqlite',
            'mssql': 'mssql+pyodbc',
        }

        db_url = f"{engine_map[conn.engine]}://{conn.username}:{conn.password}@{conn.host}:{conn.port}/{conn.database}"
        engine = create_engine(db_url)

        df = pd.read_sql(query, engine)
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        dataset = Dataset(
            name=name or f"DB Import: {query[:50]}",
            description=f"Imported from {conn.name}: {query}",
            source='database',
            file_format='csv',
            file_size=csv_buffer.getbuffer().nbytes,
            owner=owner,
        )
        dataset.file.save(f"db_import_{uuid4().hex[:8]}.csv", ContentFile(csv_buffer.read()))
        dataset.save()

        metadata = cls._compute_metadata(df)
        for key, value in metadata.items():
            setattr(dataset, key, value)
        dataset.save()

        return dataset

    @classmethod
    def get_preview(cls, dataset, limit=100):
        """Get preview data for a dataset."""
        df = cls._read_dataframe(dataset)
        df_preview = df.head(limit)
        return {
            'columns': list(df.columns),
            'data': json.loads(df_preview.to_json(orient='records', date_format='iso')),
            'total_rows': len(df),
            'preview_rows': len(df_preview),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
        }

    @classmethod
    def get_statistics(cls, dataset):
        """Get descriptive statistics."""
        df = cls._read_dataframe(dataset)
        stats = {}

        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['numeric'] = json.loads(numeric_df.describe().to_json())

        # Categorical statistics
        cat_df = df.select_dtypes(include=['object', 'category'])
        if not cat_df.empty:
            stats['categorical'] = json.loads(cat_df.describe().to_json())

        # Missing values
        stats['missing'] = {col: int(df[col].isnull().sum()) for col in df.columns}
        stats['missing_pct'] = {col: round(df[col].isnull().sum() / len(df) * 100, 2)
                                for col in df.columns}

        # Correlation matrix (numeric only)
        if not numeric_df.empty:
            stats['correlation'] = json.loads(numeric_df.corr().to_json())

        return stats

    @classmethod
    def get_column_info(cls, dataset):
        """Get detailed column information."""
        df = cls._read_dataframe(dataset)
        info = {}
        for col in df.columns:
            info[col] = {
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'sample_values': df[col].dropna().head(5).tolist(),
            }
            if df[col].dtype in ['float64', 'int64']:
                info[col].update({
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q50': float(df[col].quantile(0.50)),
                    'q75': float(df[col].quantile(0.75)),
                })
        return info

    @classmethod
    def export_dataset(cls, dataset, fmt='csv'):
        """Export dataset to specified format."""
        df = cls._read_dataframe(dataset)
        export_dir = Path(settings.MEDIA_ROOT) / 'exports'
        export_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{dataset.name}_{uuid4().hex[:8]}.{fmt}"
        filepath = export_dir / filename

        if fmt == 'csv':
            df.to_csv(filepath, index=False)
        elif fmt == 'xlsx':
            df.to_excel(filepath, index=False)
        elif fmt == 'json':
            df.to_json(filepath, orient='records')
        elif fmt == 'parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return {
            'filename': filename,
            'url': f"/media/exports/{filename}",
            'format': fmt,
            'rows': len(df),
        }

    @staticmethod
    def test_db_connection(conn):
        """Test a database connection."""
        from sqlalchemy import create_engine, text
        engine_map = {
            'postgresql': 'postgresql',
            'mysql': 'mysql+pymysql',
            'sqlite': 'sqlite',
            'mssql': 'mssql+pyodbc',
        }
        db_url = f"{engine_map[conn.engine]}://{conn.username}:{conn.password}@{conn.host}:{conn.port}/{conn.database}"
        engine = create_engine(db_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    @staticmethod
    def list_tables(conn):
        """List tables in a connected database."""
        from sqlalchemy import create_engine, inspect
        engine_map = {
            'postgresql': 'postgresql',
            'mysql': 'mysql+pymysql',
            'sqlite': 'sqlite',
            'mssql': 'mssql+pyodbc',
        }
        db_url = f"{engine_map[conn.engine]}://{conn.username}:{conn.password}@{conn.host}:{conn.port}/{conn.database}"
        engine = create_engine(db_url)
        inspector = inspect(engine)
        return inspector.get_table_names()


class DocumentService:
    """Service layer for document/RAG operations."""

    @classmethod
    def upload_and_index(cls, file, name='', description='', owner=None):
        """Upload a document and index it for RAG."""
        from .models import Document

        filename = file.name
        ext = Path(filename).suffix.lower().lstrip('.')
        type_map = {'pdf': 'pdf', 'docx': 'docx', 'txt': 'txt',
                    'pptx': 'pptx', 'md': 'md'}
        doc_type = type_map.get(ext, 'txt')

        document = Document(
            name=name or filename,
            description=description,
            doc_type=doc_type,
            file=file,
            file_size=file.size,
            owner=owner,
        )
        document.save()

        # Index the document
        try:
            cls.index_document(document)
        except Exception as e:
            logger.error(f"Document indexing error: {e}")

        return document

    @classmethod
    def index_document(cls, document):
        """Index a document in ChromaDB for RAG."""
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        import chromadb

        # Load document
        path = document.file.path
        if document.doc_type == 'pdf':
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)

        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

        # Create embeddings and store in ChromaDB
        collection_name = f"doc_{document.id.hex}"

        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )

        collection = client.get_or_create_collection(name=collection_name)

        embeddings_model = OpenAIEmbeddings(
            model=document.embedding_model,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(chunks))]

        # Batch embed
        embeddings = embeddings_model.embed_documents(texts)

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        document.is_indexed = True
        document.chunk_count = len(chunks)
        document.collection_name = collection_name
        document.save()

    @classmethod
    def query_document(cls, document, question):
        """Query a document using RAG."""
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import chromadb

        if not document.is_indexed:
            raise ValueError("Document is not indexed. Please index it first.")

        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )

        collection = client.get_collection(name=document.collection_name)

        # Embed the question
        embeddings_model = OpenAIEmbeddings(
            model=document.embedding_model,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        query_embedding = embeddings_model.embed_query(question)

        # Search for relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
        )

        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""

        # Generate answer
        llm = ChatOpenAI(
            model='gpt-4o-mini',
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

        prompt = f"""Based on the following context from a document, answer the question.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)

        return {
            'question': question,
            'answer': response.content,
            'sources': results['documents'][0][:3] if results['documents'] else [],
            'document': document.name,
        }
