import torch
import torch.nn as nn
import asyncio

from app.utils import hf_summarizer

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
i=0


async def summy(state:dict)->dict: #type:ignore
    text=state.get("text","")
    summary=hf_summarizer.summarize(text)
    return {"current_summary":summary}


class DeepRL_Agent(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=2):
        super().__init__()
        self.Layer1=nn.Linear(input_dim,16)
        self.ReLU=nn.ReLU()
        self.Layer2=nn.Linear(16,output_dim)

    def forward(self,x):
        x=self.Layer1(x)
        x=self.ReLU(x)
        return self.Layer2(x)
    
DQN_Agent=DeepRL_Agent().to(device)
Action_map={
    0:"Light_trim",
    1:"Heavy_trim"
}

async def drl_refine(state:dict)->dict:  #type:ignore
    summary=state.get("current_summary","")
    lim=state.get("lim",50)

    curr_len=len(summary)
    if lim>0:ratio=curr_len/max(lim,1)
    else: ratio=1
    diff=max(0,curr_len-lim)/100.0

    Tensor_IN=torch.FloatTensor([[ratio,diff]]).to(device)

    with torch.no_grad():
        q_val=DQN_Agent(Tensor_IN)
        action_idx=int(torch.argmax(q_val).item())

    action=Action_map[action_idx]
    global i
    i+=1
    if i>20: return {
        "current_summary":summary,
        "done":True
    }
    if action=="Heavy_trim":
        cut_idx=max(60,int(len(summary)*0.7))
        new_summary=summary[:cut_idx]
    else:
        tokens=summary.split()
        if len(tokens)>5:
            new_summary=" ".join(tokens[:-3])
        else:
            new_summary=summary
            
    return {
        "current_summary":new_summary, #type:ignore
        "action_taken":action,
        "drl_q_values":q_val.tolist()
    }


TOOL_REGISTRY={
    "dl_summarize":summy,
    "drl_refine":drl_refine
}
