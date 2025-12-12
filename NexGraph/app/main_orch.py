import PyD_models
import engine
import uuid

from fastapi import FastAPI,WebSocket,WebSocketDisconnect

app=FastAPI()

graphs={}


@app.post("/graph/create")
def create_graph(graph:PyD_models.GraphDef):
    g_id=str(uuid.uuid4())
    graphs[g_id]=graph
    return {"graph_id":g_id,"message":"Graph stored successfully"}


@app.websocket("/ws/run/{graph_id}")
async def run_workflow_socket(websocket:WebSocket,
                              graph_id:str):
    await websocket.accept()
    
    if graph_id not in graphs:
        await websocket.send_json({"error":"Graph not found"})
        await websocket.close()
        return

    try:
        data=await websocket.receive_json()
        initial_state=data.get("state",{})
        
        async def stream_log(node_id,current_state):
            await websocket.send_json({
                "step":node_id,
                "summary_preview":current_state.get("current_summary", "")[:50] + "...",
                "action":current_state.get("action_taken", "thinking...")
            })
            
        engine_s=engine.GraphFLow_Engine(graphs[graph_id],callback=stream_log)
        final_state=await engine_s.run(initial_state)
        await websocket.send_json({"status":"complete","final_state":final_state})
        await websocket.close()
        
    except WebSocketDisconnect:
        print(f"Client disconnected from graph {graph_id}")


