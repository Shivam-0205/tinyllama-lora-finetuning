import wandb

def initialize_wandb(project_name, run_name):
    wandb.init(
        project=project_name,
        name=run_name
    )

def log_metrics(metrics):
    wandb.log(metrics)

def finish_wandb():
    wandb.finish()
