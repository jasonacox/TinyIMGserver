
"""
Configuration management for TinyIMGserver.

This module handles loading and accessing configuration settings from config.yaml.
It provides global access to settings like model definitions, timeouts, and
other server parameters.
"""

import yaml

def load_config(path="/Users/jason/Code/TinyIMGserver/app/config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Global configuration object
CONFIG = load_config()