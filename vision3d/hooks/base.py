from abc import ABC

class Hook(ABC):
    """
    Abstract base class for training hooks.
    """

    def before_train_epoch(self, epoch: int, trainer) -> None:
        """Called before each training epoch."""
        pass

    def after_train_epoch(self, epoch: int, trainer) -> None:
        """Called after each training epoch."""
        pass

    def before_train_step(self, step: int, trainer) -> None:
        """Called before each training step."""
        pass

    def after_train_step(self, step: int, trainer) -> None:
        """Called after each training step."""
        pass

    def before_val_epoch(self, epoch: int, trainer) -> None:
        """Called before each val epoch."""
        pass

    def after_val_epoch(self, epoch: int, trainer) -> None:
        """Called after each val epoch."""
        pass

    def before_val_step(self, step: int, trainer) -> None:
        """Called before each val step."""
        pass

    def after_val_step(self, step: int, trainer) -> None:
        """Called after each val step."""
        pass
