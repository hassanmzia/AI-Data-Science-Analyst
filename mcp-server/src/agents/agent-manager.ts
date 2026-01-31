import OpenAI from 'openai';
import { v4 as uuidv4 } from 'uuid';
import { ToolRegistry } from '../tools/registry';
import { logger } from '../utils/logger';

interface AgentDefinition {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  tools: string[];
  systemPrompt: string;
}

interface TaskMessage {
  message: string;
  agentType: string;
  datasetId?: string;
  documentId?: string;
  conversationId?: string;
  context?: Record<string, any>;
}

interface A2ATask {
  id: string;
  agentId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  input: any;
  output?: any;
  error?: string;
  createdAt: Date;
  updatedAt: Date;
}

export class AgentManager {
  private agents: Map<string, AgentDefinition> = new Map();
  private tasks: Map<string, A2ATask> = new Map();
  private openai: OpenAI;
  private toolRegistry: ToolRegistry;

  constructor(toolRegistry: ToolRegistry) {
    this.toolRegistry = toolRegistry;
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
    this.registerAgents();
  }

  private registerAgents(): void {
    const agentDefs: AgentDefinition[] = [
      {
        id: 'data-analyst',
        name: 'Data Analyst Agent',
        description: 'Expert at exploratory data analysis, statistical summaries, and data insights',
        capabilities: ['eda', 'statistics', 'data_profiling', 'data_cleaning'],
        tools: ['eda_analysis', 'get_dataset_info', 'create_visualization'],
        systemPrompt: `You are an expert Data Analyst AI agent. You specialize in:
- Exploratory Data Analysis (EDA)
- Statistical summaries and profiling
- Data quality assessment
- Pattern identification
- Generating actionable insights from data

Always provide thorough, well-structured analysis with clear explanations.`,
      },
      {
        id: 'data-scientist',
        name: 'Data Scientist Agent',
        description: 'Expert at machine learning, hypothesis testing, and advanced analytics',
        capabilities: ['ml_modeling', 'hypothesis_testing', 'feature_engineering', 'prediction'],
        tools: ['train_ml_model', 'hypothesis_test', 'eda_analysis', 'create_visualization'],
        systemPrompt: `You are an expert Data Scientist AI agent. You specialize in:
- Machine Learning model building and evaluation
- Hypothesis testing and statistical inference
- Feature engineering and selection
- Predictive modeling
- Deep learning and neural networks

Provide rigorous, methodological approaches with proper evaluation metrics.`,
      },
      {
        id: 'sql-expert',
        name: 'SQL Expert Agent',
        description: 'Expert at natural language to SQL translation and database querying',
        capabilities: ['sql_querying', 'database_analysis', 'data_extraction'],
        tools: ['sql_query', 'get_dataset_info'],
        systemPrompt: `You are an expert SQL and Database analyst AI agent. You specialize in:
- Natural language to SQL translation
- Complex query optimization
- Database schema analysis
- Data extraction and transformation

Write efficient, well-structured SQL queries and explain results clearly.`,
      },
      {
        id: 'visualization-expert',
        name: 'Visualization Expert Agent',
        description: 'Expert at creating insightful data visualizations',
        capabilities: ['charting', 'dashboard_design', 'visual_analytics'],
        tools: ['create_visualization', 'get_dataset_info', 'eda_analysis'],
        systemPrompt: `You are an expert Data Visualization AI agent. You specialize in:
- Creating meaningful visualizations
- Choosing appropriate chart types
- KDE plots, heatmaps, scatter plots, etc.
- Dashboard design principles
- Visual storytelling with data

Create clear, informative visualizations that highlight key patterns and insights.`,
      },
      {
        id: 'rag-assistant',
        name: 'RAG Assistant Agent',
        description: 'Expert at document Q&A using retrieval-augmented generation',
        capabilities: ['document_qa', 'information_retrieval', 'summarization'],
        tools: ['query_document', 'web_search'],
        systemPrompt: `You are an AI Research Assistant with RAG capabilities. You specialize in:
- Answering questions from uploaded documents
- Information retrieval and synthesis
- Document summarization
- Cross-referencing information
- Web search for current information

Provide accurate answers with source citations.`,
      },
      {
        id: 'general-assistant',
        name: 'General DS/DA Assistant',
        description: 'A versatile assistant for all data science and analytics tasks',
        capabilities: ['eda', 'ml_modeling', 'visualization', 'sql', 'rag', 'general'],
        tools: ['eda_analysis', 'create_visualization', 'train_ml_model',
                'hypothesis_test', 'sql_query', 'query_document', 'get_dataset_info', 'web_search'],
        systemPrompt: `You are a versatile AI Data Science and Analytics Assistant. You can help with:
- Exploratory Data Analysis
- Data Visualization
- Machine Learning Model Building
- Statistical Hypothesis Testing
- SQL Database Querying
- Document Q&A (RAG)
- General data science questions

Route tasks to the most appropriate approach and provide comprehensive help.`,
      },
    ];

    for (const agent of agentDefs) {
      this.agents.set(agent.id, agent);
    }

    logger.info(`Registered ${this.agents.size} agents`);
  }

  getAgentList(): Array<{ id: string; name: string; description: string; capabilities: string[] }> {
    return Array.from(this.agents.values()).map(a => ({
      id: a.id,
      name: a.name,
      description: a.description,
      capabilities: a.capabilities,
    }));
  }

  getAgent(agentId: string): AgentDefinition | undefined {
    return this.agents.get(agentId);
  }

  async processMessage(msg: TaskMessage): Promise<any> {
    const agentTypeMap: Record<string, string> = {
      data_analyst: 'data-analyst',
      data_scientist: 'data-scientist',
      sql_expert: 'sql-expert',
      ml_engineer: 'data-scientist',
      rag_assistant: 'rag-assistant',
      general: 'general-assistant',
    };

    const agentId = agentTypeMap[msg.agentType] || 'general-assistant';
    const agent = this.agents.get(agentId);

    if (!agent) {
      throw new Error(`Agent not found: ${agentId}`);
    }

    // Build tool definitions for OpenAI
    const tools: OpenAI.ChatCompletionTool[] = agent.tools.map(toolName => {
      const tool = this.toolRegistry.get(toolName);
      if (!tool) return null;
      return {
        type: 'function' as const,
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
        },
      };
    }).filter(Boolean) as OpenAI.ChatCompletionTool[];

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: 'system', content: agent.systemPrompt },
    ];

    // Add context
    if (msg.context?.history) {
      for (const h of msg.context.history.slice(-10)) {
        messages.push({ role: h.role, content: h.content });
      }
    }

    messages.push({ role: 'user', content: msg.message });

    try {
      // First call - may include tool calls
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages,
        tools: tools.length > 0 ? tools : undefined,
        tool_choice: tools.length > 0 ? 'auto' : undefined,
        temperature: 0,
        max_tokens: 4096,
      });

      const assistantMessage = response.choices[0].message;

      // Handle tool calls
      if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
        messages.push(assistantMessage);

        const toolResults: any[] = [];
        for (const toolCall of assistantMessage.tool_calls) {
          try {
            const args = JSON.parse(toolCall.function.arguments);
            // Inject dataset/document IDs if not provided
            if (msg.datasetId && !args.dataset_id) {
              args.dataset_id = msg.datasetId;
            }
            if (msg.documentId && !args.document_id) {
              args.document_id = msg.documentId;
            }

            const result = await this.toolRegistry.callTool(toolCall.function.name, args);
            toolResults.push({
              tool: toolCall.function.name,
              result,
            });

            messages.push({
              role: 'tool',
              tool_call_id: toolCall.id,
              content: JSON.stringify(result),
            });
          } catch (error: any) {
            messages.push({
              role: 'tool',
              tool_call_id: toolCall.id,
              content: JSON.stringify({ error: error.message }),
            });
          }
        }

        // Get final response after tool calls
        const finalResponse = await this.openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages,
          temperature: 0,
          max_tokens: 4096,
        });

        return {
          content: finalResponse.choices[0].message.content,
          agentName: agent.name,
          agentId: agent.id,
          toolCalls: toolResults,
          metadata: {
            model: 'gpt-4o-mini',
            totalTokens: (response.usage?.total_tokens || 0) +
                         (finalResponse.usage?.total_tokens || 0),
          },
        };
      }

      return {
        content: assistantMessage.content,
        agentName: agent.name,
        agentId: agent.id,
        toolCalls: [],
        metadata: {
          model: 'gpt-4o-mini',
          totalTokens: response.usage?.total_tokens || 0,
        },
      };

    } catch (error: any) {
      logger.error(`Agent ${agentId} error:`, error);
      throw error;
    }
  }

  // A2A Task Management
  createTask(agentId: string, input: any): A2ATask {
    const task: A2ATask = {
      id: uuidv4(),
      agentId,
      status: 'pending',
      input,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.tasks.set(task.id, task);

    // Process async
    this.executeTask(task);

    return task;
  }

  private async executeTask(task: A2ATask): Promise<void> {
    task.status = 'running';
    task.updatedAt = new Date();

    try {
      const result = await this.processMessage({
        message: task.input.message || task.input.query || JSON.stringify(task.input),
        agentType: task.agentId,
        datasetId: task.input.datasetId,
        documentId: task.input.documentId,
        context: task.input.context || {},
      });

      task.output = result;
      task.status = 'completed';
    } catch (error: any) {
      task.error = error.message;
      task.status = 'failed';
    }

    task.updatedAt = new Date();
  }

  getTask(taskId: string): A2ATask | undefined {
    return this.tasks.get(taskId);
  }

  cancelTask(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (task && task.status === 'pending') {
      task.status = 'cancelled';
      task.updatedAt = new Date();
      return true;
    }
    return false;
  }
}
