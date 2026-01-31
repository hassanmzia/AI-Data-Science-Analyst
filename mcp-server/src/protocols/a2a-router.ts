import { Request, Response } from 'express';
import { AgentManager } from '../agents/agent-manager';
import { logger } from '../utils/logger';

export class A2ARouter {
  private agentManager: AgentManager;

  constructor(agentManager: AgentManager) {
    this.agentManager = agentManager;
  }

  listAgents(_req: Request, res: Response): void {
    const agents = this.agentManager.getAgentList();
    res.json({ agents });
  }

  getAgentCard(req: Request, res: Response): void {
    const agentId = String(req.params.agentId);
    const agent = this.agentManager.getAgent(agentId);

    if (!agent) {
      res.status(404).json({ error: `Agent not found: ${agentId}` });
      return;
    }

    // A2A Agent Card format
    res.json({
      name: agent.name,
      description: agent.description,
      url: `http://172.168.1.95:4050/a2a/agents/${agentId}`,
      version: '1.0.0',
      capabilities: {
        streaming: false,
        pushNotifications: false,
        stateTransitionHistory: true,
      },
      skills: agent.capabilities.map(cap => ({
        id: cap,
        name: cap.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()),
        description: `${agent.name} capability: ${cap}`,
        tags: [cap],
      })),
      defaultInputModes: ['text/plain'],
      defaultOutputModes: ['text/plain', 'application/json'],
    });
  }

  createTask(req: Request, res: Response): void {
    const agentId = String(req.params.agentId);
    const agent = this.agentManager.getAgent(agentId);

    if (!agent) {
      res.status(404).json({ error: `Agent not found: ${agentId}` });
      return;
    }

    try {
      const task = this.agentManager.createTask(agentId, req.body);
      res.status(201).json({
        id: task.id,
        agentId: task.agentId,
        status: task.status,
        createdAt: task.createdAt,
      });
    } catch (error: any) {
      logger.error('A2A createTask error:', error);
      res.status(500).json({ error: error.message });
    }
  }

  getTask(req: Request, res: Response): void {
    const taskId = String(req.params.taskId);
    const task = this.agentManager.getTask(taskId);

    if (!task) {
      res.status(404).json({ error: `Task not found: ${taskId}` });
      return;
    }

    res.json({
      id: task.id,
      agentId: task.agentId,
      status: task.status,
      output: task.output,
      error: task.error,
      createdAt: task.createdAt,
      updatedAt: task.updatedAt,
    });
  }

  cancelTask(req: Request, res: Response): void {
    const taskId = String(req.params.taskId);
    const success = this.agentManager.cancelTask(taskId);

    if (success) {
      res.json({ message: 'Task cancelled' });
    } else {
      res.status(400).json({ error: 'Task cannot be cancelled' });
    }
  }
}
