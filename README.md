# BehaviorHMM

A feature-enhanced Hidden Markov Model written in C++17 with no external dependencies. It takes raw behavioral observations, buckets them into discrete feature strings, and runs Viterbi decoding to infer hidden states. Next-state and next-observation prediction are included.

Built as a clean, self-contained reference implementation — no Eigen, no Boost, no ML libraries.

---

## What it does

Most HMM implementations work on a fixed, pre-defined observation alphabet. This one does not. Observations are structured objects that get hashed into feature bucket strings at runtime, so the alphabet grows dynamically as new combinations appear during training. Laplace smoothing keeps unseen combinations from zeroing out probabilities.

The core pipeline:

1. Raw observations come in as structs with fields like message type, reply delay, sentiment, and time of day
2. `to_bucket()` maps each struct to a composite string like `pos_afternoon_instant_medium_pos_init`
3. The HMM trains on labeled `(hidden_state, observation)` pairs using count accumulation
4. At inference time, Viterbi runs in log-space to decode the most likely hidden state sequence
5. Next-state distribution and next-observation predictions are derived from the final decoded state

---

## File format

Single file, no build config required:

```
behavior_hmm.cpp
```

Compile with any C++17-compliant compiler:

```bash
g++ -std=c++17 -O2 -o behavior_hmm behavior_hmm.cpp
./behavior_hmm
```

On MSVC:

```bash
cl /std:c++17 /O2 behavior_hmm.cpp
```

---

## Architecture

### `RawObservation`

The input struct. Each field gets discretized into a string token:

| Field | Type | Buckets |
|---|---|---|
| `type` | string | `pos`, `neg`, `neu`, `none` |
| `hour` | int | `night`, `morning`, `afternoon`, `evening` |
| `reply_delay_min` | int | `instant`, `fast`, `slow`, `very_slow` |
| `msg_length` | int | `very_short`, `short`, `medium`, `long` |
| `sentiment` | double | `neg`, `neu`, `pos` |
| `she_initiated` | bool | `init`, `reply` |

`to_bucket()` concatenates these tokens with underscores. The resulting string is the observation symbol fed into the HMM.

### `BehaviorHMM`

The model class. Hidden states are passed in at construction. Observation symbols are registered dynamically as training data arrives.

**Key methods:**

```cpp
// Register and train on labeled sequences
void learn(const std::vector<std::pair<std::string, RawObservation>>& seq);

// Viterbi decoding — returns most likely hidden state sequence
std::vector<std::string> decode(const std::vector<RawObservation>& obs);

// Returns transition distribution from the last decoded state
std::vector<double> predict_next_state(const std::vector<RawObservation>& obs);

// Returns ranked observation predictions for the next timestep
std::vector<std::pair<std::string, double>> predict_next_observation(const std::vector<RawObservation>& obs);
```

---

## Internals

### Training

Count accumulation over labeled sequences, then normalized in `rebuild()`:

- Start counts: incremented for the state at `t=0`
- Transition counts: incremented for each `(state[t-1], state[t])` pair
- Emission counts: incremented for each `(state, observation_bucket)` pair

### Smoothing

Laplace (+1) smoothing is applied during normalization. Every unseen `(state, observation)` pair gets a small nonzero probability, which prevents zero-probability paths during Viterbi.

### Viterbi

Runs entirely in log-space to avoid floating-point underflow on long sequences. A small epsilon (`1e-12`) is added before taking logarithms to handle zero-probability edges without producing `-inf` from genuinely missing data.

```
dp[t][j] = max over i { dp[t-1][i] + log(trans[i][j]) } + log(emit[j][obs[t]])
```

Backpointers are stored for full sequence reconstruction.

### Prediction

After decoding, the transition row of the final state gives the distribution over next hidden states. Marginalizing over that distribution with the emission matrix gives a ranked list of likely next observations.

---

## Usage

### Define hidden states

```cpp
BehaviorHMM hmm({"happy", "neutral", "frustrated"});
```

### Train on labeled data

```cpp
std::vector<std::pair<std::string, RawObservation>> data = {
    {"happy",      {"positive_message", 14, 2, 120,  0.9, true}},
    {"happy",      {"positive_message", 15, 5,  90,  0.8, false}},
    {"neutral",    {"neutral_message",  10, 40, 20,  0.1, false}},
    {"frustrated", {"negative_message", 22, 180, 5, -0.7, false}},
    {"frustrated", {"no_reply",         23, 700, 0,  0.0, false}},
    {"neutral",    {"positive_message",  9, 60, 30,  0.3, true}}
};

hmm.learn(data);
```

### Decode a new sequence

```cpp
std::vector<RawObservation> test = {
    {"positive_message", 14, 3, 110, 0.85, true},
    {"positive_message", 15, 4, 100, 0.80, false}
};

auto states = hmm.decode(test);
// → ["happy", "happy"]
```

### Predict what comes next

```cpp
auto next = hmm.predict_next_observation(test);
// Returns ranked list of (bucket_string, probability) pairs
for (size_t i = 0; i < 3 && i < next.size(); i++)
    std::cout << next[i].second << " : " << next[i].first << "\n";
```

---

## Expected output

```
=== Feature-Enhanced HMM (C++17, no external libs) ===

Training on 6 labeled steps...
Hidden states: 3
Observation bucket types: 6
Example buckets:
  pos_afternoon_instant_medium_pos_init
  pos_afternoon_instant_medium_pos_reply
  neu_morning_fast_short_neu_reply
  neg_evening_slow_very_short_neg_reply
  none_night_very_slow_very_short_neu_reply

Decoded hidden states: happy happy

Predicted next observation (top 3):
  0.214 : pos_afternoon_instant_medium_pos_init
  0.189 : pos_afternoon_instant_medium_pos_reply
  0.143 : neu_morning_fast_short_neu_reply

EXECUTION COMPLETE, LOL :-)
```

---

## Extending the model

### Add observation fields

Extend `RawObservation` with new fields and update `to_bucket()`. The rest of the model picks up the new observation space automatically — no other code changes needed.

### Add hidden states

Pass more state names to the constructor:

```cpp
BehaviorHMM hmm({"happy", "neutral", "frustrated", "excited", "distant"});
```

### Swap out the bucket strategy

`to_bucket()` is a plain member function. Replace the discretization thresholds or concatenation logic to match your domain without touching the HMM internals.

### Baum-Welch (unsupervised training)

The current implementation requires labeled training sequences. If labels are unavailable, the Baum-Welch forward-backward algorithm can estimate parameters from unlabeled observations. This is the natural next extension.

---

## Limitations

- Observation vocabulary is built only from training data. An observation bucket seen at inference time but never during training will cause `decode()` to return an empty sequence.
- The model assumes first-order Markov dependencies. Higher-order transitions require storing longer state history.
- Training is online in the sense that `learn()` accumulates counts across calls, but there is no forgetting mechanism. Old data weighs equally with new data indefinitely.
- Laplace smoothing is uniform. A better prior (Dirichlet, domain-specific) would improve performance on sparse data.

---

## License

MIT
