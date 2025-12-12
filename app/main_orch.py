import uuid

from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from app import PyD_models
from app import engine

app=FastAPI(Title="NexGraph")

graphs={}

@app.get("/health")
def health_chk():
    return {"status":"OK","message":"Backend up and running"}


@app.post("/graph/create")
def create_graph(graph:PyD_models.GraphDef):
    g_id=str(uuid.uuid4())
    graphs[g_id]=graph
    return {"graph_id":g_id,"message":"Graph stored successfully"}

@app.websocket("/ws/run/{graph_id}")
async def run_workflow_socket(websocket: WebSocket,graph_id: str):
    await websocket.accept()


    if graph_id not in graphs:
        await websocket.send_json({"error":"Graph not found"})
        await websocket.close()
        return

    data=await websocket.receive_json()
    initial_state=data.get("state",{})

    graph_def=graphs[graph_id]

    async def stream_log(node_id,current_state):
        preview=current_state.get("current_summary","")
        preview=(preview[:50]+"...") if len(preview)>50 else preview

        await websocket.send_json({
            "step": node_id,
            "summary_preview": preview,
            "action": current_state.get("action_taken", "thinking")
        })

    engine_s=engine.GraphFLow_Engine(graph_def,callback=stream_log)
    final_state await engine_s.run(initial_state)

    await websocket.send_json({"status":"complete","final_state":final_state})
    await websocket.close()

