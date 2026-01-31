import { logger } from '../utils/logger';

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, any>;
  handler: (args: Record<string, any>) => Promise<any>;
}

export class ToolRegistry {
  private tools: Map<string, ToolDefinition> = new Map();

  constructor() {
    this.registerBuiltinTools();
  }

  register(tool: ToolDefinition): void {
    this.tools.set(tool.name, tool);
    logger.info(`Tool registered: ${tool.name}`);
  }

  get(name: string): ToolDefinition | undefined {
    return this.tools.get(name);
  }

  getToolList(): Array<{ name: string; description: string; inputSchema: Record<string, any> }> {
    return Array.from(this.tools.values()).map(t => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    }));
  }

  async callTool(name: string, args: Record<string, any>): Promise<any> {
    const tool = this.tools.get(name);
    if (!tool) {
      throw new Error(`Tool not found: ${name}`);
    }
    return tool.handler(args);
  }

  private registerBuiltinTools(): void {
    // EDA Tool
    this.register({
      name: 'eda_analysis',
      description: 'Perform exploratory data analysis on a dataset using natural language',
      inputSchema: {
        type: 'object',
        properties: {
          dataset_id: { type: 'string', description: 'Dataset UUID' },
          query: { type: 'string', description: 'Natural language EDA query' },
        },
        required: ['dataset_id', 'query'],
      },
      handler: async (args) => {
        const axios = (await import('axios')).default;
        const res = await axios.post(`${process.env.BACKEND_URL}/api/analysis/sessions/run_eda/`, {
          dataset_id: args.dataset_id,
          query: args.query,
        });
        return res.data;
      },
    });

    // Visualization Tool
    this.register({
      name: 'create_visualization',
      description: 'Create data visualizations (charts, plots, heatmaps) from natural language',
      inputSchema: {
        type: 'object',
        properties: {
          dataset_id: { type: 'string', description: 'Dataset UUID' },
          query: { type: 'string', description: 'What visualization to create' },
          chart_type: { type: 'string', description: 'Chart type (auto, scatter, bar, line, etc.)' },
        },
        required: ['dataset_id', 'query'],
      },
      handler: async (args) => {
        const axios = (await import('axios')).default;
        const res = await axios.post(`${process.env.BACKEND_URL}/api/analysis/sessions/run_visualization/`, {
          dataset_id: args.dataset_id,
          query: args.query,
          chart_type: args.chart_type || 'auto',
        });
        return res.data;
      },
    });

    // ML Model Training Tool
    this.register({
      name: 'train_ml_model',
      description: 'Train a machine learning model using natural language description',
      inputSchema: {
        type: 'object',
        properties: {
          dataset_id: { type: 'string', description: 'Dataset UUID' },
          query: { type: 'string', description: 'ML task description' },
          model_type: { type: 'string', description: 'Model type (auto, logistic_regression, random_forest, etc.)' },
          target_column: { type: 'string', description: 'Target variable column' },
        },
        required: ['dataset_id', 'query'],
      },
      handler: async (args) => {
        const axios = (await import('axios')).default;
        const res = await axios.post(`${process.env.BACKEND_URL}/api/analysis/sessions/run_ml/`, {
          dataset_id: args.dataset_id,
          query: args.query,
          model_type: args.model_type || 'auto',
          target_column: args.target_column || '',
        });
        return res.data;
      },
    });

    // Hypothesis Testing Tool
    this.register({
      name: 'hypothesis_test',
      description: 'Perform statistical hypothesis testing',
      inputSchema: {
        type: 'object',
        properties: {
          dataset_id: { type: 'string', description: 'Dataset UUID' },
          query: { type: 'string', description: 'Hypothesis test description' },
          test_type: { type: 'string', description: 'Test type (auto, t_test, chi_square, etc.)' },
        },
        required: ['dataset_id', 'query'],
      },
      handler: async (args) => {
        const axios = (await import('axios')).default;
        const res = await axios.post(`${process.env.BACKEND_URL}/api/analysis/sessions/run_hypothesis_test/`, {
          dataset_id: args.dataset_id,
          query: args.query,
          test_type: args.test_type || 'auto',
        });
        return res.data;
      },
    });

    // SQL Query Tool
    this.register({
      name: 'sql_query',
      description: 'Query data using natural language SQL',
      inputSchema: {
        type: 'object',
        properties: {
          dataset_id: { type: 'string', description: 'Dataset UUID (optional)' },
          connection_id: { type: 'string', description: 'Database connection UUID (optional)' },
          query: { type: 'string', description: 'Natural language query' },
        },
        required: ['query'],
      },
      handler: async (args) => {
        const axios = (await import('axios')).default;
        const res = await axios.post(`${process.env.BACKEND_URL}/api/analysis/sessions/run_sql/`, {
          dataset_id: args.dataset_id,
          connection_id: args.connection_id,
          query: args.query,
        });
        return res.data;
      },
    });

    // RAG Document Query Tool
    this.register({
      name: 'query_document',
      description: 'Query a document using RAG (Retrieval-Augmented Generation)',
      inputSchema: {
        type: 'object',
        properties: {
          document_id: { type: 'string', description: 'Document UUID' },
          question: { type: 'string', description: 'Question about the document' },
        },
        required: ['document_id', 'question'],
      },
      handler: async (args) => {
        const axios = (await import('axios')).default;
        const res = await axios.post(
          `${process.env.BACKEND_URL}/api/datasets/documents/${args.document_id}/query/`,
          { question: args.question }
        );
        return res.data;
      },
    });

    // Dataset Info Tool
    this.register({
      name: 'get_dataset_info',
      description: 'Get information about a dataset (columns, stats, preview)',
      inputSchema: {
        type: 'object',
        properties: {
          dataset_id: { type: 'string', description: 'Dataset UUID' },
          info_type: { type: 'string', description: 'Type: preview, statistics, column_info' },
        },
        required: ['dataset_id'],
      },
      handler: async (args) => {
        const axios = (await import('axios')).default;
        const infoType = args.info_type || 'preview';
        const res = await axios.get(
          `${process.env.BACKEND_URL}/api/datasets/datasets/${args.dataset_id}/${infoType}/`
        );
        return res.data;
      },
    });

    // Web Search Tool
    this.register({
      name: 'web_search',
      description: 'Search the web for current information',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query' },
        },
        required: ['query'],
      },
      handler: async (args) => {
        if (!process.env.SERPAPI_API_KEY) {
          return { error: 'SerpAPI key not configured' };
        }
        const axios = (await import('axios')).default;
        const res = await axios.get('https://serpapi.com/search', {
          params: {
            q: args.query,
            api_key: process.env.SERPAPI_API_KEY,
            engine: 'google',
          },
        });
        const results = res.data.organic_results?.slice(0, 5) || [];
        return {
          results: results.map((r: any) => ({
            title: r.title,
            link: r.link,
            snippet: r.snippet,
          })),
        };
      },
    });

    logger.info(`Registered ${this.tools.size} built-in tools`);
  }
}
