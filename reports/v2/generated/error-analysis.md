# Visual error analysis

The table lists the measured cases used in the orthophoto/reference/prediction/error/uncertainty panels.

| Patch | Map sheet | mIoU | Boundary F1 | Entropy | Confidence |
|---|---|---:|---:|---:|---:|
| `test-L3243F-01024-08960` | L3243F | 0.0000 | 0.0000 | 0.1771 | 0.9636 |
| `test-L3243H-08448-02560` | L3243H | 0.0605 | 0.0000 | 0.6437 | 0.7186 |
| `test-L3243F-09728-03584` | L3243F | 0.0700 | 0.0000 | 0.4888 | 0.8199 |
| `test-L3243F-05632-06144` | L3243F | 0.2000 | 1.0000 | 0.4170 | 0.8980 |
| `test-L3243F-02560-08192` | L3243F | 0.4436 | 0.2726 | 0.3520 | 0.9036 |
| `test-L3243F-01280-06656` | L3243F | 0.4773 | 0.4677 | 0.2830 | 0.9277 |
| `test-L3243F-03840-02048` | L3243F | 0.5031 | 0.2859 | 0.2368 | 0.9328 |

Boundary-pixel accuracy: 0.5222; interior-pixel accuracy: 0.8024.
The boundary band is 1.5 m wide on each side of class transitions and is excluded from primary IoU/Dice metrics.
Higher boundary error is consistent with positional uncertainty only if that measured difference is present; inspect the generated panels before assigning a cause.

## Visual findings

The boundary band is materially harder: its 0.5222 accuracy is 0.2802 below the 0.8024 evaluated
interior accuracy. In the best displayed case, the cultivated-land edge and small building footprint
are localized closely, and uncertainty rises along those transitions. The highest-entropy case mixes
open-natural, residual land, cultivation, and a small structure; its uncertainty map follows both
mapped edges and broad ambiguous interiors, so boundary displacement alone does not explain the
errors.

The worst displayed patch is labelled open-natural but predicted as cultivated land with high
confidence. Its recently cleared or field-like texture is direct evidence of semantic and temporal
ambiguity in vector-supervised labels. This failure is consistent with the aggregate open-natural
recall of 0.4262. Built structures are also difficult at 0.5592 recall, reflecting their small spatial
support. These observations describe the selected examples and measured class results; they do not
establish a causal source of each disagreement.
