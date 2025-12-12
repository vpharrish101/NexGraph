import torch
import torch.nn as nn
import asyncio

from transformers import pipeline

device=torch.device("cuda")

summy_pipeline=pipeline("summarization",model="sshleifer/distilbart-cnn-12-6",device=device)
async def summy(state:dict)->dict: #type:ignore
    text=state.get("text","")
    summary_list=await asyncio.to_thread(
        summy_pipeline,text,max_length=60,min_length=10) 
    
    return {"Current_summary":summary_list[0]['summary_text']}


class DeepRL_Agent(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=2):
        self.Layer1=nn.Linear(input_dim,16)
        self.ReLU=nn.ReLU()
        self.Layer2=nn.Linear(16,output_dim)

    def forward(self,x):
        x=self.Layer1(x)
        x=self.ReLU
        return self.Layer2(x)
    
DQN_Agent=DeepRL_Agent().to(device)
Action_map={
    0:"Light_trim",
    1:"Heavy_trim"
}


async def drl_refine(state:dict)->dict:  #type:ignore
    summary=state.get("Current_summary","")
    lim=state.get("lim",50)

    curr_len=len(summary)
    if lim>0:ratio=curr_len/summary
    else: ratio=1
    diff=max(0,curr_len-lim)/100.0

    Tensor_IN=torch.FloatTensor([ratio,diff]).to(device)

    with torch.no_grad():
        q_val=DQN_Agent(Tensor_IN)
        action_idx=int(torch.argmax(q_val).item())

    action=Action_map[action_idx]

    if action is "Heavy_trim":
        cut_idx=int(len(summary)*0.7)
        new_summary=summary[:cut_idx]
    elif action is "Light_trim":
        new_summary=" ".join(summary.split()[:-1])

    return {
        "current_summary": new_summary, #type:ignore
        "action_taken": action,
        "drl_q_values": q_val.tolist()
    }


TOOL_REGISTRY={
    "dl_summarize":summy,
    "drl_refine":drl_refine
}
