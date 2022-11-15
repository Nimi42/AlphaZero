# AlphaZero
Chess Engine with AlphaZero flavour

# Monte Carlo Tree Search
The goal of this repository is to implement an AlphaZero like algortihm to play chess starting with Monte Carlo Tree Search (MCTS).
The original Monte Carlo Tree Search algorithm uses the UCB1 balance exploitation (nodes showing promise) with exploration
(potentially good nodes). A good explanation on MCTS can be found on [here][1]. The UCB1 score is defined as follows:

$$ 
\begin{align}
    UCB1      & = \frac{w_i}{s_i} &+ & c \sqrt{\frac{\ln s_p}{s_i}}
\end{align}
$$

AlphaZero uses a very similar scoring function called PUCT that can accomodate probabilities given by a neural network and some
info on it can be found [here][2].
It is given as follows:

$$ 
\begin{align}
    PUCT(s,a) & = Q(s,a)          &+ & U(s,a) \\
              & = \frac{w_i}{s_i} &+ & c_{puct} P(s,a) \frac{\sqrt{\sum_b N(s,b)}} {1 + N(s,a)} \\
              & = \frac{w_i}{s_i} &+ & c_{puct} P(s,a) \frac{\sqrt{s_p}} {1 + s_i}
\end{align}
$$

where $Q(s,a)$ is basically just the average winrate/value (AlphaZero does not do playouts) for all simulations just like in UCB1.
Since the output of the neural network is basically a number between 1 (first player wins) and -1 (second player wins) the only real
difference between the original MCTS and MCTS, apart from the neural network, is the way the reward is updated during the backpropagation.
The calculation of the exploration part might be different, but uses the same values (number of times a node has been visited).

$$
\begin{align*}
    PUCT \Rightarrow&& w_i &= \sum^{s_i}_j x_j & \bigg|& \quad x_j =   
        \begin{cases}
            1, & \text{If player has won} \\
            \text{-}1, & \text{If player has lost}
        \end{cases} \\
    UCB1 \Rightarrow&& w_i &= \sum^{s_i}_j y_j & \bigg|& \quad y_j =   
        \begin{cases}
            1, & \text{If player has won} \\
            0, & \text{If player has lost}
        \end{cases}
\end{align*}
$$

It is easy to see that the reward for $x_j$ and $y_j$ can easily be scaled to represent each other.

$$
y_j = \frac{x_j +1}{2}
$$

Using some simple math we can substitute for $y_j$ and simplify the term to show that we can adjust the values during
the scoring instead of directly during the backpropagation.

$$
\begin{align*}
     UCB1 \Rightarrow&& w_i &= \sum^{s_i}_j \frac{x_j +1}{2} & \bigg|& \quad y_j = \frac{x_j +1}{2} \\
           &&&= \frac{1}{2} \sum^{s_i}_j x_j + 1 \\
           &&&= \frac{1}{2} (s_i + \sum^{s_i}_j x_j) \\
     \Rightarrow&& \frac{w_i}{s_i} &= \frac{1}{2} \frac{(s_i + \sum\limits^{s_i}_j x_j)}{s_i} \\
           &&&= \frac{1}{2} (\frac{s_i}{s_i} + \frac{\sum\limits^{s_i}_j x_j}{s_i}) \\
           &&&= \frac{(1 + \frac{ {w_i}^{puct} } {s_i})}{2}
\end{align*}
$$

This allows us to use the same mcts algorithm without many changes for both the original MCTS and AlphaZero and should explain,
why the scoring function used in this repository is different from UCB1.

[1]: https://medium.com/@quasimik/monte-carlo-tree-search-applied-to-letterpress-34f41c86e238
[2]: https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
