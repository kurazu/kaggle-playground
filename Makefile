# S03E02

s03e02/features.json: \
		s03e02/train.csv \
		playground/pipelines/s03e02.py
	poetry run python \
		-m playground.cli.fit \
		--train-file=s03e02/train.csv \
		--config-file=s03e02/features.json \
		--customization=playground.pipelines.s03e02.model_customization

s03e02/train.transformed.csv: \
		s03e02/train.csv \
		s03e02/features.json \
		playground/pipelines/s03e02.py
	poetry run python \
		-m playground.cli.transform \
		--config-file=s03e02/features.json \
		--customization=playground.pipelines.s03e02.model_customization \
		--input-file=s03e02/train.csv \
		--output-file=s03e02/train.transformed.csv

s03e02/test.transformed.csv: \
		s03e02/test.csv \
		s03e02/features.json \
		playground/pipelines/s03e02.py
	poetry run python \
		-m playground.cli.transform \
		--config-file=s03e02/features.json \
		--customization=playground.pipelines.s03e02.model_customization \
		--input-file=s03e02/test.csv \
		--output-file=s03e02/test.transformed.csv

s03e02/split.train.csv s03e02/split.valid.csv s03e02/split.eval.csv: \
		s03e02/train.transformed.csv \
		playground/cli/split_data.py \
		playground/pipelines/s03e02.py
	poetry run python \
		-m playground.cli.split_data \
		--input-file=s03e02/train.transformed.csv \
		--train-output-file=s03e02/split.train.csv \
		--validation-output-file=s03e02/split.valid.csv \
		--evaluation-output-file=s03e02/split.eval.csv \
		--customization=playground.pipelines.s03e02.model_customization

s03e02/saved_model: \
		s03e02/split.train.csv \
		s03e02/split.valid.csv \
		s03e02/split.eval.csv \
		playground/pipelines/s03e02.py
	poetry run python \
		-m playground.cli.train \
		--customization=playground.pipelines.s03e02.model_customization \
		--train-file=s03e02/split.train.csv \
		--validation-file=s03e02/split.valid.csv \
		--evaluation-file=s03e02/split.eval.csv \
		--output-dir=s03e02/saved_model

s03e02/submission.csv: \
		s03e02/test.transformed.csv \
		s03e02/saved_model \
		playground/pipelines/s03e02.py 
	poetry run python \
		-m playground.cli.predict \
		--customization=playground.pipelines.s03e02.model_customization \
		--input-file=s03e02/test.transformed.csv \
		--output-file=s03e02/submission.csv \
		--model-dir=s03e02/saved_model

# S03E03

s03e03/features.json: \
		s03e03/train_cleaned.csv \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.fit \
		--train-file=s03e03/train_cleaned.csv \
		--config-file=s03e03/features.json \
		--customization=playground.pipelines.s03e03.model_customization

s03e03/train.transformed.csv: \
		s03e03/train_cleaned.csv \
		s03e03/features.json \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.transform \
		--config-file=s03e03/features.json \
		--customization=playground.pipelines.s03e03.model_customization \
		--input-file=s03e03/train_cleaned.csv \
		--output-file=s03e03/train.transformed.csv

s03e03/test.transformed.csv: \
		s03e03/test.csv \
		s03e03/features.json \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.transform \
		--config-file=s03e03/features.json \
		--customization=playground.pipelines.s03e03.model_customization \
		--input-file=s03e03/test.csv \
		--output-file=s03e03/test.transformed.csv

s03e03/split.train.csv s03e03/split.valid.csv s03e03/split.eval.csv: \
		s03e03/train.transformed.csv \
		playground/cli/split_data.py \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.split_data \
		--input-file=s03e03/train.transformed.csv \
		--train-output-file=s03e03/split.train.csv \
		--validation-output-file=s03e03/split.valid.csv \
		--evaluation-output-file=s03e03/split.eval.csv \
		--customization=playground.pipelines.s03e03.model_customization

s03e03/saved_model: \
		s03e03/split.train.csv \
		s03e03/split.valid.csv \
		s03e03/split.eval.csv \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.train \
		--customization=playground.pipelines.s03e03.model_customization \
		--train-file=s03e03/split.train.csv \
		--validation-file=s03e03/split.valid.csv \
		--evaluation-file=s03e03/split.eval.csv \
		--output-dir=s03e03/saved_model

s03e03/submission.csv: \
		s03e03/test.transformed.csv \
		s03e03/saved_model \
		playground/pipelines/s03e03.py 
	poetry run python \
		-m playground.cli.predict \
		--customization=playground.pipelines.s03e03.model_customization \
		--input-file=s03e03/test.transformed.csv \
		--output-file=s03e03/submission.csv \
		--model-dir=s03e03/saved_model