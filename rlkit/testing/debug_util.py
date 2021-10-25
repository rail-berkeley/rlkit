"""For tracing programs and comparing outputs"""

import torch

i = 0

def save(x):
    torch.save(x, "../tmp.pt")
    return x

def load():
    return torch.load("../tmp.pt")

def savei(x):
    global i
    torch.save(x, "../tmp/%d.pt" % i)
    i = i + 1
    return x

def loadi():
    global i
    x = torch.load("../tmp/%d.pt" % i)
    i = i + 1
    return x
