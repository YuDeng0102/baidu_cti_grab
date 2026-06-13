import torch
import torch.nn as nn
import time

vocab_size = 10000
emb_dim = 512
num_bags = 2000
bag_size = 50

emb = nn.Embedding(vocab_size, emb_dim).cuda()
emb_bag = nn.EmbeddingBag(vocab_size, emb_dim, mode='sum').cuda()
emb_bag.weight.data.copy_(emb.weight.data)

values = torch.randint(0, vocab_size, (num_bags * bag_size,)).cuda()
lengths = torch.full((num_bags,), bag_size, dtype=torch.long).cuda()
offsets = torch.cat([torch.tensor([0]).cuda(), lengths.cumsum(0)[:-1]])

# Check correctness
res1 = torch.segment_reduce(emb(values), reduce='sum', lengths=lengths)
res2 = emb_bag(values, offsets)
print("Max diff:", (res1 - res2).abs().max().item())

# Benchmark
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    res1 = torch.segment_reduce(emb(values), reduce='sum', lengths=lengths)
torch.cuda.synchronize()
t1 = time.time()
for _ in range(100):
    res2 = emb_bag(values, offsets)
torch.cuda.synchronize()
t2 = time.time()

print(f"Emb + reduce: {(t1 - t0)*1000:.2f} ms")
print(f"EmbeddingBag: {(t2 - t1)*1000:.2f} ms")
