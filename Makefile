data/features.json: \
		data/train.csv \
		playground/cli/feature_engineering.py \
		playground/feature_engineering
	poetry run python \
		-m playground.cli.feature_engineering \
		--train-file=data/train.csv \
		--config-file=data/features.json \
		--target-column=stroke

data/train.transformed.csv: \
		data/train.csv \
		data/features.json \
		playground/cli/transform.py \
		playground/feature_engineering
	poetry run python \
		-m playground.cli.transform \
		--config-file=data/features.json \
		--target-column=stroke \
		--input-file=data/train.csv \
		--output-file=data/train.transformed.csv

data/test.transformed.csv: \
		data/test.csv \
		data/features.json \
		playground/cli/transform.py \
		playground/feature_engineering
	poetry run python \
		-m playground.cli.transform \
		--config-file=data/features.json \
		--input-file=data/test.csv \
		--output-file=data/test.transformed.csv

split: \
		data/train.transformed.csv \
		playground/cli/split_data.py
	poetry run python \
		-m playground.cli.split_data \
		--input-file=data/train.transformed.csv \
		--train-output-file=split/train.transformed.csv \
		--validation-output-file=split/valid.transformed.csv \
		--evaluation-output-file=split/eval.transformed.csv

saved_model: \
		split \
		playground/model/train.py
	poetry run python \
		-m playground.cli.train \
		--train-file=split/train.transformed.csv \
		--validation-file=split/valid.transformed.csv \
		--evaluation-file=split/eval.transformed.csv \
		--output-dir=saved_model
