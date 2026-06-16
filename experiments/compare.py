from losses.mse import MSE


def compare(
    model_builder,
    optimizers,
    X,
    y,
    epochs=5000,
    learning_rate=0.01
):

    results = []

    for optimizer in optimizers:

        model = model_builder()

        loss_fn = MSE()

        for _ in range(epochs):

            pred = model.forward(X)

            grad = loss_fn.backward(
                y,
                pred
            )

            model.backward(
                grad,
                learning_rate,
                optimizer
            )

        pred = model.forward(X)

        pred = (
            pred > 0.5
        ).astype(int)

        accuracy = (
            pred == y
        ).mean()

        results.append({
            "optimizer":
                optimizer.__class__.__name__,
            "accuracy":
                accuracy
        })

    return results