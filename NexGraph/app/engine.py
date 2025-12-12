import PyD_models
import ToolKit

from typing import Dict,Callable,Optional

class GraphFLow_Engine:
    def __init__(self,
                 graph_def:PyD_models.GraphDef,
                 callback:Optional[Callable]=None):
        
        self.nodes={n.id: n for n in graph_def.nodes}
        self.edges=graph_def.edges
        self.start_node=graph_def.start_node
        self.callback=callback
    
    async def run(self,initial_state:Dict)->Dict: #type:ignore
        state=initial_state.copy()
        cid=self.start_node   #cid -> current id of node
        while cid:
            node=self.nodes.get(cid)
            if not node:break

            tool_fn=ToolKit.TOOL_REGISTRY.get(node.toolkit_fn)
            if tool_fn is not None:
                updates=await tool_fn(state)
                state.update(updates)
            if self.callback:
                await self.callback(cid, state)
            next_id=None
            for edge in self.edges:
                if edge.from_node==cid:
                    if edge.condition:
                        try:
                            allowed_names={"state":state,"len":len}
                            if eval(edge.condition,{},allowed_names):
                                next_id=edge.to_node
                                break
                        except Exception as e:
                            print(f"Condition Error: {e}")
                    else:
                        next_id=edge.to_node
                        break
            cid=next_id
            
        return state
    
def a():
    return 0

