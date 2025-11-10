from pathlib import Path

import click

# Defer heavy imports into the command functions so the module can be imported
# for `--help` or when invoked without installing all optional dependencies.

@click.group()
def cli():
    pass


@cli.command()
@click.argument("config", type=click.Path(exists=True), required=True)
def train(config):
    """
    Run training job via the CLI from a YAML config file.
    """
    print("Load config")
    # Import lazily to avoid importing heavy dependencies at module import time
    from model_training.trainer import Trainer

    trainer = Trainer(config)
    try:
        print("Start training")
        trainer.train()
        trainer.validate()
        print("Ran training")
    finally:
        print("Finished training job")



if __name__ == "__main__":
    # When this module is executed with `python -m model_training.cli ...`
    # call the click group so CLI subcommands (like `train`) are executed.
    cli()