class SGD:
    """Gradient Descent optimizer."""

    def update(
        self,
        param_name,
        params,
        grads,
        learning_rate
    ):
        return params - learning_rate * grads