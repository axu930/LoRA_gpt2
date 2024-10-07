import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
import numpy as np

import math
import os


# ----------
# Implement GPT2 modules 

class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    lora_r: int = 4 # bottleneck dimension for LoRA



class LoRA(nn.Module):

    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 r: int,
                 ):
        super().__init__()

        self.A = nn.Linear(in_features, r, bias = False)
        self.B = nn.Linear(r, out_features, bias = False)

        nn.init.kaiming_uniform_(self.A.weight, a = math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)

        return x
    


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output embedding
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.qlora = LoRA(config.n_embd, config.n_embd, config.lora_r)
        self.klora = LoRA(config.n_embd, config.n_embd, config.lora_r)
        self.vlora = LoRA(config.n_embd, config.n_embd, config.lora_r)
        self.olora = LoRA(config.n_embd, config.n_embd, config.lora_r)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # LoRA adaption
        k = k + self.klora(k)
        q = q + self.qlora(q)
        v = v + self.vlora(v)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # With LoRA adaption
        y = self.c_proj(y) + self.olora(y)
        return y


class MLP(nn.Module):

    def __init__(self, config,):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

        # self.lora = LoRA(config.n_embd, config.n_embd, config.lora_r)

    def forward(self, x):
        # LoRA bypass
        # y = self.lora(x)

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x) # + y
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        # config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig()
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        print("Finished loading")

        return model


# ----------
# dataloader + methods for loading tokens and text generation

with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding('gpt2')
data = enc.encode(text) # 593306 tokens


def generate_text(model, in_tokens: torch.tensor, num_tokens = 30, num_samples = 5):
    with torch.no_grad():
        tokens = in_tokens.unsqueeze(0).repeat(num_samples,1)
        for _ in range(num_tokens):
            logits = model(tokens)[0]
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim = -1)
            top50probs, top50indices = torch.topk(probs, 50, dim = -1)
            ix = torch.multinomial(top50probs,1)
            xcol = torch.gather(top50indices, -1, ix)
            tokens = torch.cat((tokens, xcol), dim = 1) 

    #Print the generated text
    for i in range(num_samples):
        toks = tokens[i,:].tolist()
        print(">", enc.decode(toks))

    return tokens



# ---------
# optimizer and training methods

class DataLoader:

    def __init__(self, B, T, data, split):
        self.B = B
        self.T = T

        # train/val split
        n = int(len(data) * 0.9)
        if split == 'val':
            self.data = torch.tensor(data[n:])
        else:
            self.data = torch.tensor(data[:n])

        self.current_position = 0

    def next_batch(self):
        B,T = self.B, self.T
        
        if self.current_position + B*T + 1 >= len(self.data):
            overflow = self.current_position + B*T + 1 - len(self.data)
            buf = torch.cat((self.data[self.current_position:], self.data[:overflow]))
            self.current_position = overflow
        else:
            buf = self.data[self.current_position:self.current_position + B*T + 1]
            self.current_position = self.current_position + B*T + 1
        
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        return x,y

B = 4 #Batch size
T = 1024 #Sequence length
# 4096 tokens per batch -> ~130 steps per epoch
# 16,384 tokens per batch -> ~33 steps per epoch
lr = 1e-3 #Learning rate
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(B, T, data, split = 'train')
    val_loader = DataLoader(B, T, data, split='val')
    
    # Logging
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass
    text_file = os.path.join(log_dir, f"generated_text.txt")

    for step in range(steps):
        # periodically check loss, generate text, and save state at the end
        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                x,y = val_loader.next_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device, dtype = torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss.detach()
                print(f"step: {step}, validation loss: {loss.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"\nstep: {step}, val: {loss.item():.4f}")

                # Once every 5 eval cycles, generate some text
                if step % 50 == 0: 
                    with open(text_file, "a") as f:
                        toks = generate_text(model, torch.tensor([198]), num_tokens=50, num_samples=8)
                        f.write("\n -------------------- \n")
                        f.write(f"Text generated at step {step}")
                        for i in range(8):
                            x = toks[i,:].tolist()
                            text = "\n >" + enc.decode(x)
                            f.write(text)
                    
                #save at the very end
                if step == steps-1:
                    # write model checkpoints at the end
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': model.state_dict(),
                        'config': model.config,
                        'step': step,
                        'val_loss': loss.item()
                    }
                    torch.save(checkpoint, checkpoint_path)
        
        #Training stuff
        model.train()
        optimizer.zero_grad()
        x,y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x,y)
        loss.backward()
        optimizer.step()


# ----------
# make sure that weights from hf load properly
# model = GPT(GPTConfig)

model = GPT.from_pretrained('gpt2')

#Freeze parameters
for name, param in model.named_parameters():
    param.requires_grad = False
    if name.__contains__('lora'):
        param.requires_grad = True

trainable_params = 0  #294912
total_params = 0  #124734720
for param in model.parameters():
    if param.requires_grad == True:
        trainable_params += param.numel()
    total_params += param.numel()
print("Total parameters: ", total_params)
print("Trainable parameters: ", trainable_params)



#model = model.to(device)
# model training loop
# train(model, 500)



