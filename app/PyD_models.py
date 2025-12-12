from pydantic import BaseModel
from typing import List,Optional,Dict,Any


class Node(BaseModel):
    id:str
    toolkit_fn:str 

class Edge(BaseModel):
    '''
    We make the edge with the assumption that there's only one node connecting to and fro.
    '''
    from_node:str
    to_node:str
    condition:Optional[str]=None 

class GraphDef(BaseModel):
    nodes:List[Node]
    edges:List[Edge]
    start_node:str

