# Deviations from the original algorithms

RigFL aims to follow each algorithm's published specification. Departures that
affect interpretation or comparability are listed here.

| Algorithm | Departure | Why / consequence |
|---|---|---|
| **FedAMP** | Uses the constant self-weight and normalized peer attention from the authors' experiments instead of deriving both from the step size in published Eq. 5. | The experimental parameterization always produces a convex cloud model and exposes the local proximal denominator directly as `beta`. Defaults are informed by the reported experiments: `self_weight=0.5`, `sigma=10`, and constant `beta=10000`; the authors decayed beta during training. |
| **FedAMP** | All clients start from one shared model, perform the first local phase without a cloud or proximal penalty, and start later finite-step local solves from their cloud models. | The published pseudocode leaves initialization and numerical solver initialization unspecified. The [v1 preprint](https://arxiv.org/pdf/2007.03797v1) locally pretrains the initial models, while the [authors' linked notebook](https://cnnorth4-modelarts-sdk.obs.cn-north-4.myhuaweicloud.com/snapshot/pytorch_fedamp_emnist_classification.ipynb) uses a shared base, aggregates after the first local phase, and reloads each cloud before later training. RigFL combines that executable protocol with Eq. 6's squared proximal objective. |
| **APPLE** | Uses the paper's factor of `1/2` in the directed-relationship proximal term. | The released APPLE code omits the factor shown in Eq. 7. RigFL follows the published objective, so the configured `mu` has the paper's scale. |
| **FedProto / FedTGP** | RigFL adds predictive probabilities that the papers do not specify. | See [Prototype loss interpretation](#prototype-loss-interpretation) for how these probabilities affect evaluation. |
| **FedProto** | Global class prototypes use the released implementation's equal-client average rather than the class-sample weighting suggested by the paper's objective. | The paper weights client-class terms by their share of class samples, while Eq. 6 also contains a client-count normalization. The released code instead averages one prototype per contributing client equally; RigFL follows that executable reference. |
| **FedProto** | Client prototypes are recomputed in a clean pass after local training. | The released implementation accumulates representations from batches while the model is still being updated. RigFL instead evaluates the final local model over its training data once, so every representation in the prototype comes from the same final parameter state. |
| **FedTGP** | Client guidance uses per-sample mean-squared error to the global prototype. | Eq. 11 specifies Euclidean distance between the client's class prototype and the corresponding global prototype. The released code instead applies per-sample MSE, and RigFL follows that executable reference; `lamda` therefore has the released implementation's scale. |
| **FedTGP** | The adaptive margin is the largest classwise nearest-other-class distance, not the literal all-pairs maximum printed in Eq. 9. | The paper's prose defines a class margin through its nearest other class, and the released code implements the maximum of those classwise margins. RigFL follows that intended and executable definition. |
| **pFedMoE** | The private gate uses LayerNorm after its first linear transformation instead of the paper's SwitchNorm-before-linear plus BatchNorm layers. | LayerNorm stays defined for a final batch of one sample. The two-layer sigmoid/softmax gate and sample-specific convex mixture are unchanged, but exact gate optimization can differ from the paper. |

## Prototype loss interpretation

FedProto and FedTGP predict the nearest prototype. RigFL converts those distances
into `softmax(-d)` probabilities, whose argmax preserves the nearest-prototype
decision.

Distance scale is learned and can change across rounds. Prototype loss is useful
for monitoring that prediction rule within one run, but it is not a calibrated
quantity for comparing unrelated algorithms or representation spaces. RigFL does not
calibrate these probabilities.
