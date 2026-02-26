### NexGraph: Graph-Driven Workflow Engine
A lightweight, modular engine that executes AI workflows defined as directed graphs. Nodes call tools (ML/DL/RL modules), edges define flow logic, and the entire pipeline streams live updates

### Core Features: -
  -Graph-Defined workflow Pipelines.\
  -Each node links to a function in ToolKit.py (e.g., summarizer, RL refinement, etc.).\
  -GraphFlow_Engine walks the graph, executes tools, evaluates edge conditions, and streams outputs.\
  -Real-time updates on workflow progress: summaries, actions taken, Q-values, loop counts, etc.\
  -Upload graphs and receive a unique workflow UUID.

### Added capabilities: -
  - Added websoockets and async wait fn's for continuous connection
  - Stateless file designs
  - Simple Deep-QN based action selection mechanism for summarization


### Steps to run: -
  1. Open the root folder, and execute
         ```uvicorn app.main_orch:app```
  2. Post the example graph provided in graph.json in the FastAPI's post option, and copy the UUID generated at output.
  3. Paste the UUID in the url at ws.py (it is denoted where to paste) and run the file.

### Output: -

You should see the output like: -
<img width="1357" height="723" alt="image" src="https://github.com/user-attachments/assets/cefeba8e-1883-4c49-8ec6-ae60415142ec" />

