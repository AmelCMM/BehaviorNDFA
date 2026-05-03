# BehaviorHMM

A Hidden Markov Model that learns from sequences of raw behavioral observations and infers hidden emotional states. Feed it labeled examples, and it figures out transition patterns between states — then uses those patterns to decode new observations and predict what comes next.

Built in C++17, **no external dependencies**.

---

## What it does

You give it observations like "she sent a long, positive message at 2pm after replying within 5 minutes" and a label like `happy`. Over time it learns what kinds of messages tend to come from which states, and how states tend to flow into each other. Then you give it a new sequence of unlabeled observations and ask: *what state is she in, and what happens next?*

Three core operations:

- **`learn`** — train on labeled `(hidden_state, observation)` pairs
- **`decode`** — run Viterbi to infer the most likely hidden state sequence for new observations  
- **`predict_next_observation`** — given recent observations, predict the most probable next observation (with probabilities)

---

## How observations work

Each `RawObservation` has six fields:

```cpp
struct RawObservation {
    std::string type;       // "positive_message", "negative_message", "neutral_message", "no_reply"
    int hour;               // 0-23
    int reply_delay_min;    // minutes since last message
    int msg_length;         // character count
    double sentiment;       // -1.0 (very negative) to +1.0 (very positive)
    bool she_initiated;     // true if she sent first
};