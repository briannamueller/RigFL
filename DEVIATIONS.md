# Deviations from the original algorithms

RigFL aims to follow each algorithm's published specification. Departures that
affect interpretation or comparability are listed here.

| Algorithm | Departure | Why / consequence |
|---|---|---|
| **FedAvg / FedProx** | Every client participates in every round. | The current shared round loop has no client-sampling policy. The update is faithful to full-participation FedAvg/FedProx, but experiments do not reproduce partial-participation settings from the papers. |
| **FedAvg / FedProx** | Non-floating model state is copied from the client with the largest local sample count (first client wins ties). | Integer buffers such as BatchNorm's `num_batches_tracked` cannot be sample-weight averaged without inventing a non-client integer value. Floating parameters and buffers remain sample-count weighted. |
| **FedProx** | Clients perform a fixed configured number of local epochs. | This implements the proximal objective but not the paper's systems-heterogeneity experiments with variable work or its abstract gamma-inexact local solver. |
| **FedAMP** | Uses the paper's squared L2 proximal penalty. | PFLlib's current implementation applies an unsquared L2 norm, while FedAMP Eq. 6 specifies squared L2. RigFL follows the published objective. |
| **FedAMP / APPLE** | Parameter mixtures leave non-parameter module buffers at the target client's value. | The papers define arithmetic over parameter vectors. State such as BatchNorm counters is not part of those vectors and cannot be meaningfully combined with arbitrary learned weights. |
| **APPLE** | Uses the paper's factor of `1/2` in the directed-relationship proximal term. | The released APPLE code omits the factor shown in Eq. 7. RigFL follows the published objective, so the configured `mu` has the paper's scale. |
| **FedPAC** | Small QP weights are zeroed and then renormalized. | The released code zeros weights below `1e-3` without restoring their sum to one, although Eq. 11 requires nonnegative weights summing to one. RigFL preserves the published constraint. |
| **FedPAC** | The classifier QP is solved by deterministic projected gradient rather than CVXPY. | Both solve the same nonnegative simplex-constrained quadratic program. This avoids adding a heavyweight solver dependency; numerical solutions can differ within the configured tolerance. |
| **FedProto / FedTGP** | RigFL adds predictive probabilities that the papers do not specify. | See [Prototype loss interpretation](#prototype-loss-interpretation) for how these probabilities affect evaluation. |
| **FedProto** | Prototypes are computed in a clean pass after local training. | Some implementations accumulate them during training, which averages features from a model that was still moving. |
| **LG-FedAvg** | Trains from scratch. | The original's released scripts warm-start from an 800–1800-round FedAvg checkpoint and then run 500 LG rounds. From-scratch is a weaker configuration, so this number is not comparable to the paper's. |
| **LG-FedAvg** | Shares exactly one `nn.Linear` (the head). | The original keeps two layers local. RigFL's split point is fixed by `ClientModel`'s structure rather than configurable. |
| **FedKD** | Averages mentee *parameters*, not gradients. | The paper transmits gradients and applies `Θ_s -= η_s · ḡ`. The two coincide at `local_epochs = 1` and diverge above it. |
| **FedKD** | Distils only the final shared representation. | The original pairs 4 student to 12 teacher layers on a fixed stride and also distils self-attention maps. Neither has an analogue in a CNN pool. |
| **FedKD** | 4-D convolution tensors are not SVD-compressed. | Only 2-D weights are factorized. **Accuracy is faithful; communication cost is not — do not quote a communication-saving number from this implementation.** |
| **FedKD** | SGD rather than Adam. | The paper's learning rates (2e-6 / 5e-6) are tuned for a Transformer on NLP tasks and do not transfer to a CNN pool. |
| **FML / FedKD** | Applied to a heterogeneous architecture pool. | Both originals assume an identical shared component across clients (a meme model, a mentee). Using them across genuinely different backbones is a generalization beyond their original setting. |
| **LG-FedAvg** | Applied to a heterogeneous architecture pool. | The original's main experiments use one architecture for every client. |

## Prototype loss interpretation

FedProto and FedTGP predict the nearest prototype. RigFL converts those distances
into `softmax(-d)` probabilities, whose argmax preserves the nearest-prototype
decision.

Distance scale is learned and can change across rounds. Prototype loss is useful
for monitoring that prediction rule within one run, but it is not a calibrated
quantity for comparing unrelated algorithms or representation spaces. RigFL does not
add normalization or fitted temperature calibration, because either would change
the evaluated algorithm and introduce another experimental choice.
