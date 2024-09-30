import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set a random seed for reproducibility
torch.manual_seed(666233)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The train materials
story = """Once upon a time, there was a little puppy named Max. Max loved to play in the garden. One day, he chased a butterfly and got lost. He looked around but couldn't find his way home. Max felt scared and started to bark.
A kind girl named Lily heard him and followed the sound. When she saw Max, she picked him up and said, "Don't worry, little puppy. I'll take you home." 
Lily asked her neighbors if they knew Max. Finally, they found his house. Max's owner was so happy! Max learned to always stay close to home after that."""

news = """This week, a new playground opened in our town. Many kids were excited to play there. The playground has slides, swings, and a sandbox. There is also a climbing wall.
The mayor cut the ribbon, and all the children rushed to try the new equipment. "This playground is so much fun!" said a girl named Emma.
The playground will be open every day from morning until evening. Everyone is welcome to come and play!"""

tech_write = """Plants need sunlight, water, and soil to grow. The sunlight helps the plant make food. Water helps the plant stay healthy. The soil gives the plant nutrients.
In this experiment, we will grow two plants. One will get sunlight, and the other will stay in the dark. We will give both plants the same amount of water.
After one week, we will see which plant grows better. We think the plant in the sunlight will grow taller and greener."""

poetry = """Whispers of the breeze, soft and slow,
Through the trees, they gently flow.
Sunset paints the sky with gold,
A peaceful world, quiet and bold.

Stars appear as night unfolds,
Casting dreams and stories told.
In this calm, the heart does see,
Life’s a dance of mystery."""

data = [story, news, tech_write, poetry]

# Define the special "end of text" token
EOT_TOKEN = "<EOT>"

# Combine the text data into a single sequence with EOT tokens at the end of each article
combined_text = f" {EOT_TOKEN} ".join(data) + f" {EOT_TOKEN}"

# Character-level encoding (with EOT included)
chars = sorted(set(combined_text))  # Get unique characters including <EOT>
char_to_ix = {ch: i for i, ch in enumerate(chars)}  # Character to index
ix_to_char = {i: ch for i, ch in enumerate(chars)}  # Index to character

# N-gram size
N = 8

# Function to generate a dataset from the combined text
def generate_dataset(text, n_gram):
    inputs, outputs = [], []
    for i in range(len(text) - n_gram):
        sequence = text[i:i+n_gram]
        target = text[i+n_gram]
        inputs.append(torch.tensor([char_to_ix[c] for c in sequence]))
        outputs.append(F.one_hot(torch.tensor(char_to_ix[target]), len(chars)))
    return torch.stack(inputs), torch.stack(outputs)

inputs_list, outputs_list = [], []

# Generate the dataset for each article
for article in data:
    print("Article:", article)
    this_inputs, this_outputs = generate_dataset(article, N)
    inputs_list.extend(this_inputs)
    outputs_list.extend(this_outputs)

inputs = torch.stack(inputs_list).to(device)
outputs = torch.stack(outputs_list).to(device)

# Define the N-gram MLP model class
class NgramMLP(nn.Module):
    def __init__(self, n_gram, vocab_size, embedding_dim=64, hidden_dim=256):
        super().__init__()
        self.n_gram = n_gram
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * n_gram, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = x.view(-1, self.token_embedding.embedding_dim * self.n_gram)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Custom cross-entropy loss function
def cross_entropy_loss(outputs, targets):
    log_probs = torch.log_softmax(outputs, dim=1)
    return -(log_probs * targets).sum(dim=1).mean()

# Training function for the model
def train(model, inputs, outputs, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = cross_entropy_loss(predictions, outputs.float())
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Train the model
vocab_size = len(chars)
model = NgramMLP(N, vocab_size).to(device)
train(model, inputs, outputs, epochs=2000, learning_rate=0.001)

def generate(model, start_text, max_new_chars=1000):
    model.eval()
    with torch.no_grad():
        current_sequence = start_text[-model.n_gram:]
        generated_text = start_text
        
        for _ in range(max_new_chars):
            input_tensor = torch.tensor([char_to_ix[c] for c in current_sequence]).to(device)
            output = model(input_tensor.unsqueeze(0))
            probs = F.softmax(output, dim=1)
            next_char_ix = torch.multinomial(probs, 1).item()
            next_char = ix_to_char[next_char_ix]
            
            generated_text += next_char
            current_sequence = current_sequence[1:] + next_char
            
            if next_char == EOT_TOKEN:
                break
        
        return generated_text

# Test the generation
start_sequence = "Once upon a time, there was a little puppy named Max."
print(generate(model, start_sequence, max_new_chars=1000))
