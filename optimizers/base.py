class Optimizer:

    def update(
        self,
        param_name,
        params,
        grads,
        learning_rate
    ):
        raise NotImplementedError(
            "Optimizer must implement update()"
        )