import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(input_size=1,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor, hidden: torch.Tensor = None) ->[torch.Tensor, torch.Tensor]:
        output, hidden = self.rnn(input, hidden)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output, hidden

    def sample(self,
               input: torch.Tensor,
               hidden: torch.Tensor = None,
               n_steps: int = 1) -> [torch.Tensor, torch.Tensor]:
        """Generates a pixel sequence given an input pixel sequence one at a time.
        The output pixel at step (t) is used as the input at (t+1).
        """
        with torch.no_grad():
            # consume input sequence
            output, hidden = self.forward(input, hidden)

            # output at last time step is the first input for generation
            input = output[:, -1, :].unsqueeze(1)

            outputs = torch.empty(input.size(0), n_steps)

            for i in range(n_steps):
                input, hidden = self.forward(input, hidden)
                outputs[:, i] = input.squeeze() # Nx1
            return outputs, hidden