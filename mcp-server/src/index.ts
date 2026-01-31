import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import http from 'http';
import { MCPRouter } from './protocols/mcp-router';
import { A2ARouter } from './protocols/a2a-router';
import { ToolRegistry } from './tools/registry';
import { AgentManager } from './agents/agent-manager';
import { logger } from './utils/logger';

const app = express();
const PORT = parseInt(process.env.PORT || '4050');

// Middleware
app.use(cors({
  origin: [
    'http://172.168.1.95:3050',
    'http://localhost:3050',
    'http://backend:8050',
  ],
  credentials: true,
}));
app.use(express.json({ limit: '50mb' }));

// Initialize services
const toolRegistry = new ToolRegistry();
const agentManager = new AgentManager(toolRegistry);
const mcpRouter = new MCPRouter(toolRegistry, agentManager);
const a2aRouter = new A2ARouter(agentManager);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'MCP/A2A Server',
    version: '1.0.0',
    agents: agentManager.getAgentList(),
    tools: toolRegistry.getToolList(),
  });
});

// MCP Protocol endpoints
app.post('/mcp/tools/list', (req, res) => mcpRouter.listTools(req, res));
app.post('/mcp/tools/call', (req, res) => mcpRouter.callTool(req, res));
app.post('/mcp/resources/list', (req, res) => mcpRouter.listResources(req, res));
app.post('/mcp/resources/read', (req, res) => mcpRouter.readResource(req, res));
app.post('/mcp/prompts/list', (req, res) => mcpRouter.listPrompts(req, res));
app.post('/mcp/prompts/get', (req, res) => mcpRouter.getPrompt(req, res));

// A2A Protocol endpoints
app.get('/a2a/agents', (req, res) => a2aRouter.listAgents(req, res));
app.post('/a2a/agents/:agentId/tasks', (req, res) => a2aRouter.createTask(req, res));
app.get('/a2a/tasks/:taskId', (req, res) => a2aRouter.getTask(req, res));
app.post('/a2a/tasks/:taskId/cancel', (req, res) => a2aRouter.cancelTask(req, res));
app.get('/a2a/agents/:agentId/card', (req, res) => a2aRouter.getAgentCard(req, res));

// Agent interaction endpoint
app.post('/api/agent/chat', async (req, res) => {
  try {
    const { message, agentType, datasetId, documentId, conversationId, context } = req.body;
    const result = await agentManager.processMessage({
      message,
      agentType: agentType || 'general',
      datasetId,
      documentId,
      conversationId,
      context: context || {},
    });
    res.json(result);
  } catch (error: any) {
    logger.error('Agent chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Create HTTP server
const server = http.createServer(app);

// WebSocket server for real-time communication
const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws) => {
  logger.info('WebSocket client connected');

  ws.on('message', async (data) => {
    try {
      const msg = JSON.parse(data.toString());
      const result = await agentManager.processMessage(msg);
      ws.send(JSON.stringify(result));
    } catch (error: any) {
      ws.send(JSON.stringify({ error: error.message }));
    }
  });

  ws.on('close', () => {
    logger.info('WebSocket client disconnected');
  });
});

server.listen(PORT, '0.0.0.0', () => {
  logger.info(`MCP/A2A Server running on port ${PORT}`);
  logger.info(`Tools registered: ${toolRegistry.getToolList().length}`);
  logger.info(`Agents available: ${agentManager.getAgentList().length}`);
});
