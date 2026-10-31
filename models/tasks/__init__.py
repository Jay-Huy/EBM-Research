"""Task models (Phase C.4 Task 4): encoder and decoder of each dataset around the core latent."""
import models.tasks.matrix  # noqa: F401  (registers inverse, lowrank, addition)
import models.tasks.sudoku  # noqa: F401  (registers sudoku_rrn)
import models.tasks.cifar  # noqa: F401  (registers cifar10, cifar100)
