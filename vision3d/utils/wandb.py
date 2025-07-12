import wandb
from typing import Dict, Any, Optional, Union
import logging
import os

class WandbLogger:
    """Optional Weights & Biases logger for training metrics and losses."""
    
    def __init__(self, 
                 enabled: bool = False,
                 project_name: str = "vision3d",
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[list] = None,
                 save_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize WandB logger.
        
        Args:
            enabled: Whether to enable wandb logging
            project_name: WandB project name
            entity: WandB entity (username or team)
            run_name: Name for this run
            config: Configuration dictionary to log
            tags: List of tags for the run
            save_dir: Directory to save wandb logs
            **kwargs: Additional wandb.init arguments
        """
        self.enabled = enabled
        self.run = None
        
        if not self.enabled:
            return
            
        try:
            # Set up wandb directory if specified
            if save_dir:
                os.environ["WANDB_DIR"] = save_dir
                
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                tags=tags,
                **kwargs
            )
            logging.info(f"WandB logging initialized: {self.run.url}")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB: {e}")
            self.enabled = False
    
    def log_metrics(self, 
                   metrics: Dict[str, Any], 
                   step: Optional[int] = None, 
                   prefix: str = "",
                   commit: bool = True):
        """Log metrics to wandb."""
        if not self.enabled or not self.run:
            return
            
        formatted_metrics = {}
        for key, value in metrics.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            
            # Handle nested dictionaries
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_metric_name = f"{metric_name}_{sub_key}"
                    formatted_metrics[sub_metric_name] = sub_value
            else:
                formatted_metrics[metric_name] = value
        
        try:
            wandb.log(formatted_metrics, step=step, commit=commit)
        except Exception as e:
            logging.warning(f"Failed to log metrics to WandB: {e}")
    
    def log_loss(self, 
                loss_dict: Union[Dict[str, float], float], 
                step: Optional[int] = None, 
                prefix: str = "",
                commit: bool = True):
        """Log loss values to wandb."""
        if not self.enabled or not self.run:
            return
            
        if isinstance(loss_dict, dict):
            self.log_metrics(loss_dict, step=step, prefix=prefix, commit=commit)
        else:
            metric_name = f"{prefix}/loss" if prefix else "loss"
            try:
                wandb.log({metric_name: loss_dict}, step=step, commit=commit)
            except Exception as e:
                logging.warning(f"Failed to log loss to WandB: {e}")
    
    def watch_model(self, model, log_freq: int = 1000):
        """Watch model gradients and parameters."""
        if self.enabled and self.run:
            try:
                wandb.watch(model, log_freq=log_freq)
            except Exception as e:
                logging.warning(f"Failed to watch model: {e}")
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                self.enabled = False
            except Exception as e:
                logging.warning(f"Failed to finish WandB run: {e}")
    
    def is_enabled(self) -> bool:
        """Check if wandb logging is enabled."""
        return self.enabled and self.run is not None