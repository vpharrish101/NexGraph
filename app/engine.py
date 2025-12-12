from app import PyD_models
from typing import Dict,Callable,Optional
from app.utils import cfg_logging

class GraphFLow_Engine:
    def __init__(self,
                 graph_def:PyD_models.GraphDef,
                 callback:Optional[Callable]=None,
                 loops=50):  
        
        self.nodes={n.id: n for n in graph_def.nodes}
        self.edges=graph_def.edges
        self.start_node=graph_def.start_node
        self.callback=callback
        self.loops=loops
    
    async def run(self,initial_state:Dict)->Dict: #type:ignore
        from app import ToolKit
        state=initial_state.copy()
        cid=self.start_node   #cid => current id of node
        loop_ctr=0
        while cid:
            if loop_ctr>self.loops:
                cfg_logging.logger.warning("<engine> Loop limit exceeded.")
                break

            node=self.nodes.get(cid)
            if not node:
                cfg_logging.logger.error(f"<engine> Node '{cid}' not found")
                break

            tool_fn=ToolKit.TOOL_REGISTRY.get(node.toolkit_fn)
            if tool_fn:
                updates=await tool_fn(state)
                if isinstance(updates,dict):
                    state.update(updates)
                if isinstance(updates, dict) and updates.get("done"):
                    cfg_logging.logger.info("<engine> Workflow requested stop (done=True).")
                    break

                cfg_logging.logger.info(f"<engine> [Node: {cid}] Ran {node.toolkit_fn} | Updates: {updates}")

            if self.callback:
                await self.callback(cid,state)

            next_id=None
            for edge in self.edges:
                if edge.from_node!=cid:
                    continue

                if edge.condition:
                    try:
                        safe_env={"state":state,"len":len}
                        if eval(edge.condition,{},safe_env):
                            next_id=edge.to_node
                            break
                    except Exception as e:
                        cfg_logging.logger.error(f"<engine> Condition error in edge {edge}: {e}")
                else:
                    next_id=edge.to_node
                    break

            if next_id is None:
                cfg_logging.logger.info(f"<engine> No outgoing edge from '{cid}'. Ending workflow.")
                break

            if next_id==cid:
                loop_ctr+=1
                cfg_logging.logger.info(f"<engine> [Loop] Node '{cid}' repeated. Count: {loop_ctr}")

            else: loop_ctr=0
            cid=next_id

        return state


