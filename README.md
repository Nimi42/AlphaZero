# AlphaZero
Chess Engine with AlphaZero flavour

$$ 
\begin{align}
    UCB1      & = \frac{w_i}{s_i} &+ & c \sqrt{\frac{\ln s_p}{s_i}}      \\
    PUCT(s,a) & = Q(s,a)          &+ & U(s,a) \\
              & = \frac{w_i}{s_i} &+ & c_{puct} P(s,a) \frac{\sqrt{\sum_b N(s,b)}} {1 + N(s,a)} \\
              & = \frac{w_i}{s_i} &+ & c_{puct} P(s,a) \frac{\sqrt{s_p}} {1 + s_i}
\end{align}
$$

asd

$$
\begin{align*}
    UCB1 \Rightarrow&& w_i &= \sum^{s_i}_j y_j & \bigg|& \quad y_j =   
        \begin{cases}
            1, & \text{If player has won} \\
            0, & \text{If player has lost}
        \end{cases} \\
    PUCT \Rightarrow&& w_i &= \sum^{s_i}_j x_j & \bigg|& \quad x_j =   
        \begin{cases}
            1, & \text{If player has won} \\
            \text{-}1, & \text{If player has lost}
        \end{cases} \\
     UCB1 \Rightarrow&& w_i &= \sum^{s_i}_j \frac{x_j +1}{2} & \bigg|& \quad y_j = \frac{x_j +1}{2} \\
           &&&= \frac{1}{2} \sum^{s_i}_j x_j + 1 \\
           &&&= \frac{1}{2} (s_i + \sum^{s_i}_j x_j) \\
     \Rightarrow&& \frac{w_i}{s_i} &= \frac{1}{2} \frac{(s_i + \sum\limits^{s_i}_j x_j)}{s_i} \\
           &&&= \frac{1}{2} (\frac{s_i}{s_i} + \frac{\sum\limits^{s_i}_j x_j}{s_i}) \\
           &&&= \frac{1}{2} (1 + \frac{ {w_i}^{puct} } {s_i})
\end{align*}
$$
