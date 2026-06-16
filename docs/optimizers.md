# Optimizers

Optimizers update the network parameters after gradients are computed.

## Implemented Optimizers

- `SGD`
- `Momentum`
- `RMSProp`
- `Adam`

## Shared Interface

Each optimizer exposes an `update(...)` method that receives:

- a parameter name
- the current parameter values
- the gradient values
- the learning rate

That shared interface lets you swap optimizers without changing the model code.

## Behavior Overview

### SGD

The simplest update rule. It applies the gradient directly scaled by the learning rate.

### Momentum

Tracks velocity across steps to smooth updates and accelerate learning in consistent directions.

### RMSProp

Adapts the step size using a running average of squared gradients, which helps stabilize training.

### Adam

Combines momentum and RMSProp-style scaling. It is the most flexible default choice in the current codebase.

## Example

```python
from optimizers.adam import Adam

optimizer = Adam()
```

## Practical Use

The trainer passes the selected optimizer into each layer's backward step, so the same optimizer state is reused across the full network.
