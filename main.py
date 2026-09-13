import torch
import torch.nn as nn
import torch.optim as optim

# these are the training paramaters...
X = torch.tensor([1,2,4,6,10,16,7])
Y = torch.tensor([1,4,8,12,20,32,14])

class LinearRegression(nn.Module):
    def __int__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.lin(x)
        return out




#THE MODEL TO WORK ON...
model = LinearRegression(1, 1)
loss_fn = nn.MSELoss()

#defining loss and learning rate to train the model
learning_rate = 0.01
epochs = 100

#defining an optimizer...

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    Y_pred = model(X)
    loss = loss_fn(Y, Y_pred)

    #calculating the grad..
    loss.backward()
    



         


    
        
