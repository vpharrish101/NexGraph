### NexGraph: Graph-Driven Workflow Engine
A lightweight, modular engine that executes AI workflows defined as directed graphs. Nodes call tools (ML/DL/RL modules), edges define flow logic, and the entire pipeline streams live updates

### Core Features: -
  -Graph-Defined workflow Pipelines
  -Each node links to a function in ToolKit.py (e.g., summarizer, RL refinement, etc.).
  -GraphFlow_Engine walks the graph, executes tools, evaluates edge conditions, and streams outputs.
  -Real-time updates on workflow progress: summaries, actions taken, Q-values, loop counts, etc.
  -Upload graphs and receive a unique workflow UUID.

### Added capabilities: -
  - Added websoockets and async wait fn's for continuous connection
  - Stateless file designs
  - Simple Deep-QN based action selection mechanism for summarization
