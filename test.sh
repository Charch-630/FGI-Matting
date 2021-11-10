#!/usr/bin/env bash
echo Which PYTHON: `which python`
python main.py \
--config=config/FGI_config.toml \
--phase=test
