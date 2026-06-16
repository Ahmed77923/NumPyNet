class Metric:

    def __call__(self, y_true, y_pred):
        return self.compute(
            y_true,
            y_pred
        )

    def compute(
        self,
        y_true,
        y_pred
    ):
        raise NotImplementedError(
            "Subclasses must implement compute()"
        )