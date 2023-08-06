# Week 6: Long short-term memory (LSTM) and gated recurrent units (GRU)


# Long Short-Term Memory (LSTM)
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) introduced by Hochreiter and Schmidhuber in 1997. They're specifically designed to avoid the long-term dependency problem, a challenge with standard RNNs, which have trouble learning to connect information that is spaced far apart in time sequences.

LSTM networks include a 'memory cell' that can maintain information in memory for long periods. As the network processes sequences of data, it can selectively read from, write to, or forget its memory cell using structures called gates. These gates are essentially neural networks themselves, which learn to control the memory cell's behavior.

Each LSTM unit has three such gates:

Forget Gate: Decides what information should be thrown away or kept.
Input Gate: Updates the cell state with new information.
Output Gate: Determines what the next hidden state should be.
The critical advantage of LSTMs is their ability to remember from long-term sequences (window sizes), which is extremely useful in many NLP tasks.

# Gated Recurrent Units (GRU)
Gated Recurrent Units (GRU) are another variant of recurrent neural networks, introduced by Cho, et al. in 2014. They're similar to LSTMs with a forget and an input gate, but they lack an output gate. Also, the memory cell and hidden state are merged into a single context unit, resulting in a simpler and more streamlined architecture.

In particular, GRUs have two gates:

Reset Gate: Determines how much of the previous state to forget.
Update Gate: Controls how much of the state information to update with new data.
Due to their reduced complexity, GRUs are often faster to train than LSTMs. However, they may not perform as well on tasks requiring sophisticated memory usage as they lack an output gate.

LSTM and GRU in Practice
LSTM and GRU units are especially useful for sequence prediction problems because they can use their memory to process sequences of arbitrary length, instead of assuming a pre-specified window as input, like in traditional neural networks. They're widely used in tasks such as language modeling, machine translation, speech recognition, and much more.

For example, in the task of text generation, an LSTM or GRU network can learn the probability of a word given its preceding words, thus generating a sequence that follows the learned language patterns. The memory cells in these networks allow them to maintain a form of 'context' by remembering the past words, which influences the upcoming word predictions.

# Readings

[Long Sort-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

[Tutorial on GRU](https://d2l.ai/chapter_recurrent-modern/gru.html)

[Using LSTM and GRU neural network methods for traffic flow prediction](https://www.researchgate.net/profile/Li-Li-86/publication/312402649_Using_LSTM_and_GRU_neural_network_methods_for_traffic_flow_prediction/links/5c20d38d299bf12be3971696/Using-LSTM-and-GRU-neural-network-methods-for-traffic-flow-prediction.pdf)

# Code examples

