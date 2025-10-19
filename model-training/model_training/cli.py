from pathlib import Path

import click

from core.schemas import QuantizationAwareTrainingConfig
from qat import QuantizationAwareTrainingPipeline
from trainer import Trainer


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config", type=click.Path(exists=True), required=True)
def train(config):
    """
    Run training job via the CLI from a YAML config file.
    """
    trainer = Trainer(config)
    try:
        trainer.train()
        trainer.validate()
    finally:
        print("No finish run")
        #trainer.finish_run()


@cli.command()
@click.argument("config", type=click.Path(exists=True), required=True)
def qat(config: Path):
    """
    Run QAT job via CLI from a YAML config file
    """
    train_config = QuantizationAwareTrainingConfig.from_yaml(config)
    pipeline = QuantizationAwareTrainingPipeline(train_config)
    try:
        pipeline.run()
    finally:
        print("The END")
        #pipeline.finish_wandb()
