import copy
import time

from deployment.services.numpy_net_backend import build_model
from losses import binary_cross_entropy
from optimizers.adam import Adam
from optimizers.momentum import Momentum
from optimizers.rmsprop import RMSProp
from optimizers.sgd import SGD

from datasets.moons import Moons
from sklearn.metrics import trains_test_split

X, y = Moons().generate(n_samples=1000, noise=0.1)
class OptimizerComparison:

    def __init__(
        self,
        model_builder,
        optimizers,
        loss_fn,
        epochs=1000,
        learning_rate=0.01,):
        self.model_builder = model_builder
        self.optimizers = optimizers
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.learning_rate = learning_rate
    def run(self, X, y):

        results = {}

        for optimizer in self.optimizers:

            # Create a fresh optimizer copy
            optimizer = copy.deepcopy(optimizer)

            model = self.model_builder()

            history = {
                "loss": [],
                "accuracy": [],
            }

            start_time = time.perf_counter()

            for epoch in range(self.epochs):

                # Forward pass
                pred = model.forward(X)

                # Loss
                loss = self.loss_fn.forward(
                    y,
                    pred,
                )

                # Backward pass
                grad = self.loss_fn.backward(
                    y,
                    pred,
                )

                model.backward(
                    grad,
                    self.learning_rate,
                    optimizer,
                )

                # Binary classification accuracy
                accuracy = (
                    (pred > 0.5).astype(int) == y
                ).mean()

                history["loss"].append(
                    float(loss)
                )

                history["accuracy"].append(
                    float(accuracy)
                )

                # Optional progress logging
                if epoch % 100 == 0:
                    print(
                        f"{optimizer.__class__.__name__}"
                        f" | Epoch {epoch}"
                        f" | Loss: {loss:.6f}"
                        f" | Accuracy: {accuracy:.4f}"
                    )

            elapsed_time = (
                time.perf_counter() - start_time
            )

            history["final_loss"] = float(loss)
            history["final_accuracy"] = float(accuracy)
            history["training_time"] = float(elapsed_time)

            results[
                optimizer.__class__.__name__
            ] = history

        return results
    
comparison = OptimizerComparison(
    model_builder=build_model,
    optimizers=[
        SGD(),
        Momentum(),
        RMSProp(),
        Adam(),
    ],
    loss_fn=binary_cross_entropy(),
    epochs=1000,
    learning_rate=0.01,
)

results = comparison.run(X, y)

print(results["Adam"]["final_accuracy"])
print(results["Adam"]["training_time"])