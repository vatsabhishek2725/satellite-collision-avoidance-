"""OrbitZoo — orbital-dynamics + MARL environment."""

__all__ = ["OrbitZoo"]
__version__ = "0.1.2"


def __getattr__(name):
    if name == "OrbitZoo":
        from orbitzoo.env import OrbitZoo

        return OrbitZoo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
