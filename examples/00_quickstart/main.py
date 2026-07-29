"""Offline quickstart: generate with mock:echo."""

from aire import AI, __version__
from aire.models.base import run_sync


def main() -> None:
    model = AI.models.use_sync("mock:echo")
    reply = run_sync(model.ask("Hello from aire quickstart"))
    print(reply)
    print("aire", __version__)
    print("providers:", AI.models.providers())


if __name__ == "__main__":
    main()
