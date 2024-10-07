# LoRA_gpt2
Implementing LoRA fine tuning on the GPT2 124M huggingface checkpoint. Implementation of the GPT2 model follows Andrej Karpathy's implementation in his youtube video https://youtu.be/l8pRSuU81PU?si=p49BtdB-or_ox5f9. Implementation of the LoRA modules follows the original LoRA paper https://arxiv.org/pdf/2106.09685 and uses LoRA on the query, key, value, and out matrices on the causal self attention modules with rank 4. The fine tuning data is from the complete works of HP Lovecraft. Final training was done on google colab because my poor computer (CPU) literally cannot train more than ~100k parameters without overheating. 

Generated text samples can be found in the log folder. Interestingly, the loss doesn't improve much, but the style of the generated text gets noticeably closer in style to Lovecraft. 

Trainable params = 294,912
Total params = 124,734,720

