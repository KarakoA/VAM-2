import torch.nn as nn

import torch

from utils.graph_viz import make_dot

class A:
    def f(self):
        (self.b,self.c) = (2,3)
def main2():
    a = A()
    a.f()
    print(a.b)

def main():
    input = torch.tensor((1.0,2.0,3.0,4.0,5.0)).reshape(1,1,5)
    batch,seq, data= input.shape
    stacked_lstm = nn.LSTM(input_size=data,
                                hidden_size=5,
                                num_layers=2,
                                batch_first=True)

    #output, (hn, cn) = rnn(input, (h0, c0))
    output, (h1,c1) = stacked_lstm.forward(input)
    print(f"Output:  {output}\n")
    print(f"Hidden1: {h1[1]}\n")
    print(f"Cell1:   {c1}\n")

    dot = make_dot(h1[1], None)
    dot.render('./tmp/h1.gv', view = True)

    #dot = make_dot(output, None)
    #dot.render('./tmp/output.gv', view = True)
    dot = make_dot(h1, None)
    dot.render('./tmp/h.gv', view = True)
if __name__ == '__main__':
    main2()
