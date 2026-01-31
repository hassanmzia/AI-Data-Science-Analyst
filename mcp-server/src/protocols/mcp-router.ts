import { Request, Response } from 'express';
import { ToolRegistry } from '../tools/registry';
import { AgentManager } from '../agents/agent-manager';
import { logger } from '../utils/logger';

export class MCPRouter {
  private toolRegistry: ToolRegistry;
  private agentManager: AgentManager;

  constructor(toolRegistry: ToolRegistry, agentManager: AgentManager) {
    this.toolRegistry = toolRegistry;
    this.agentManager = agentManager;
  }

  async listTools(req: Request, res: Response): Promise<void> {
    try {
      const tools = this.toolRegistry.getToolList();
      res.json({
        tools: tools.map(t => ({
          name: t.name,
          description: t.description,
          inputSchema: t.inputSchema,
        })),
      });
    } catch (error: any) {
      logger.error('MCP listTools error:', error);
      res.status(500).json({ error: error.message });
    }
  }

  async callTool(req: Request, res: Response): Promise<void> {
    try {
      const { name, arguments: args } = req.body;
      if (!name) {
        res.status(400).json({ error: 'Tool name required' });
        return;
      }

      const result = await this.toolRegistry.callTool(name, args || {});
      res.json({
        content: [{
          type: 'text',
          text: typeof result === 'string' ? result : JSON.stringify(result, null, 2),
        }],
      });
    } catch (error: any) {
      logger.error('MCP callTool error:', error);
      res.status(500).json({
        content: [{
          type: 'text',
          text: `Error: ${error.message}`,
        }],
        isError: true,
      });
    }
  }

  async listResources(req: Request, res: Response): Promise<void> {
    res.json({
      resources: [
        {
          uri: 'ds://datasets',
          name: 'Datasets',
          description: 'Available datasets for analysis',
          mimeType: 'application/json',
        },
        {
          uri: 'ds://documents',
          name: 'Documents',
          description: 'Uploaded documents for RAG',
          mimeType: 'application/json',
        },
        {
          uri: 'ds://models',
          name: 'ML Models',
          description: 'Trained machine learning models',
          mimeType: 'application/json',
        },
        {
          uri: 'ds://archive',
          name: 'Project Archive',
          description: 'Archived projects for reuse',
          mimeType: 'application/json',
        },
      ],
    });
  }

  async readResource(req: Request, res: Response): Promise<void> {
    try {
      const { uri } = req.body;
      const axios = (await import('axios')).default;
      const backendUrl = process.env.BACKEND_URL || 'http://backend:8050';

      let data: any;
      switch (uri) {
        case 'ds://datasets':
          data = (await axios.get(`${backendUrl}/api/datasets/datasets/`)).data;
          break;
        case 'ds://documents':
          data = (await axios.get(`${backendUrl}/api/datasets/documents/`)).data;
          break;
        case 'ds://models':
          data = (await axios.get(`${backendUrl}/api/analysis/models/`)).data;
          break;
        case 'ds://archive':
          data = (await axios.get(`${backendUrl}/api/archive/projects/`)).data;
          break;
        default:
          res.status(404).json({ error: `Resource not found: ${uri}` });
          return;
      }

      res.json({
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(data, null, 2),
        }],
      });
    } catch (error: any) {
      logger.error('MCP readResource error:', error);
      res.status(500).json({ error: error.message });
    }
  }

  async listPrompts(req: Request, res: Response): Promise<void> {
    res.json({
      prompts: [
        {
          name: 'eda_analysis',
          description: 'Perform comprehensive EDA on a dataset',
          arguments: [
            { name: 'dataset_name', description: 'Name of the dataset', required: true },
          ],
        },
        {
          name: 'build_ml_model',
          description: 'Build and evaluate an ML model',
          arguments: [
            { name: 'dataset_name', description: 'Name of the dataset', required: true },
            { name: 'target', description: 'Target variable', required: true },
            { name: 'task_type', description: 'classification or regression', required: true },
          ],
        },
        {
          name: 'hypothesis_test',
          description: 'Formulate and test a statistical hypothesis',
          arguments: [
            { name: 'hypothesis', description: 'The hypothesis to test', required: true },
          ],
        },
      ],
    });
  }

  async getPrompt(req: Request, res: Response): Promise<void> {
    const { name, arguments: args } = req.body;

    const prompts: Record<string, (args: any) => any> = {
      eda_analysis: (a: any) => ({
        messages: [{
          role: 'user',
          content: {
            type: 'text',
            text: `Perform a comprehensive exploratory data analysis on the "${a.dataset_name}" dataset.
Include: shape, column types, missing values, descriptive statistics, correlations, and key insights.`,
          },
        }],
      }),
      build_ml_model: (a: any) => ({
        messages: [{
          role: 'user',
          content: {
            type: 'text',
            text: `Build a ${a.task_type} model to predict "${a.target}" using the "${a.dataset_name}" dataset.
Steps: clean data, feature engineering, train-test split, model training, evaluation with appropriate metrics.`,
          },
        }],
      }),
      hypothesis_test: (a: any) => ({
        messages: [{
          role: 'user',
          content: {
            type: 'text',
            text: `Test the following hypothesis: ${a.hypothesis}
Include: null/alternative hypotheses, test selection, test statistic, p-value, and conclusion.`,
          },
        }],
      }),
    };

    const promptFn = prompts[name];
    if (!promptFn) {
      res.status(404).json({ error: `Prompt not found: ${name}` });
      return;
    }

    res.json(promptFn(args || {}));
  }
}
