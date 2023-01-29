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

s03e03/joined.csv: \
		s03e03/old_cleaned.csv \
		s03e03/train_cleaned.csv \
		playground/cli/join.py
	poetry run python \
		-m playground.cli.join \
		--old-file=s03e03/old_cleaned.csv \
		--new-file=s03e03/train_cleaned.csv \
		--output-file=s03e03/joined.csv

s03e03/features.json: \
		s03e03/joined.csv \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.fit \
		--train-file=s03e03/joined.csv \
		--config-file=s03e03/features.json \
		--customization=playground.pipelines.s03e03.model_customization

s03e03/train.transformed.csv: \
		s03e03/joined.csv \
		s03e03/features.json \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.transform \
		--config-file=s03e03/features.json \
		--customization=playground.pipelines.s03e03.model_customization \
		--input-file=s03e03/joined.csv \
		--output-file=s03e03/train.transformed.csv

s03e03/test.augmented.csv: \
		s03e03/test.csv \
		playground/cli/augment.py
	poetry run python \
		-m playground.cli.augment \
		--input-file=s03e03/test.csv \
		--output-file=s03e03/test.augmented.csv \
		--column-name=source \
		--column-value=new

s03e03/test.transformed.csv: \
		s03e03/test.augmented.csv \
		s03e03/features.json \
		playground/pipelines/s03e03.py
	poetry run python \
		-m playground.cli.transform \
		--config-file=s03e03/features.json \
		--customization=playground.pipelines.s03e03.model_customization \
		--input-file=s03e03/test.augmented.csv \
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

# S03E04

# s03e04/train.augmented.csv: \
# 		s03e04/train.csv \
# 		playground/cli/augment.py
# 	poetry run python \
# 		-m playground.cli.augment \
# 		--input-file=s03e04/train.csv \
# 		--output-file=s03e04/train.augmented.csv \
# 		--column-name=dataset \
# 		--column-value=old

# s03e04/old.augmented.csv: \
# 		s03e04/old.adjusted.csv \
# 		playground/cli/augment.py
# 	poetry run python \
# 		-m playground.cli.augment \
# 		--input-file=s03e04/old.adjusted.csv \
# 		--output-file=s03e04/old.augmented.csv \
# 		--column-name=dataset \
# 		--column-value=old

# s03e04/joined.csv: \
# 		s03e04/train.augmented.csv \
# 		s03e04/old.augmented.csv \
# 		playground/cli/join.py
# 	poetry run python \
# 		-m playground.cli.join \
# 		--old-file=s03e04/old.augmented.csv \
# 		--new-file=s03e04/train.augmented.csv \
# 		--output-file=s03e04/joined.csv

# s03e04/features.json: \
# 		s03e04/joined.csv \
# 		playground/pipelines/s03e04.py
# 	poetry run python \
# 		-m playground.cli.fit \
# 		--train-file=s03e04/joined.csv \
# 		--config-file=s03e04/features.json \
# 		--customization=playground.pipelines.s03e04.model_customization

# s03e04/train.augmented_transformed.csv: \
# 		s03e04/joined.csv \
# 		s03e04/features.json \
# 		playground/pipelines/s03e04.py
# 	poetry run python \
# 		-m playground.cli.transform \
# 		--config-file=s03e04/features.json \
# 		--customization=playground.pipelines.s03e04.model_customization \
# 		--input-file=s03e04/joined.csv \
# 		--output-file=s03e04/train.augmented_transformed.csv

# s03e04/train.transformed.csv: \
# 		s03e04/train.augmented_transformed.csv
# 	poetry run python \
# 		-m playground.cli.drop_column \
# 		--input-file=s03e04/train.augmented_transformed.csv \
# 		--output-file=s03e04/train.transformed.csv \
# 		--column=dataset__passthrough

# s03e04/split.train.csv s03e04/split.valid.csv s03e04/split.eval.csv: \
# 		s03e04/train.transformed.csv \
# 		playground/cli/split_data.py \
# 		playground/pipelines/s03e04.py
# 	poetry run python \
# 		-m playground.cli.split_data \
# 		--input-file=s03e04/train.transformed.csv \
# 		--train-output-file=s03e04/split.train.csv \
# 		--validation-output-file=s03e04/split.valid.csv \
# 		--evaluation-output-file=s03e04/split.eval.csv \
# 		--customization=playground.pipelines.s03e04.model_customization

s03e04/train.transformed.csv s03e04/test.transformed.csv: \
		s03e04/train.csv \
		s03e04/old.csv \
		s03e04/test.csv \
		playground/cli/s03e04_prepare_datasets.py
	poetry run python \
		-m playground.cli.s03e04_prepare_datasets \
		--train-file=s03e04/train.csv \
		--test-file=s03e04/test.csv \
		--old-file=s03e04/old.csv \
		--train-output-file=s03e04/train.transformed.csv \
		--test-output-file=s03e04/test.transformed.csv

s03e04/saved_model: \
		s03e04/split.train.csv \
		s03e04/split.valid.csv \
		s03e04/split.eval.csv \
		playground/pipelines/s03e04.py
	poetry run python \
		-m playground.cli.train \
		--customization=playground.pipelines.s03e04.model_customization \
		--train-file=s03e04/split.train.csv \
		--validation-file=s03e04/split.valid.csv \
		--evaluation-file=s03e04/split.eval.csv \
		--output-dir=s03e04/saved_model

# s03e04/train_test.joined.csv: \
# 		s03e04/train.csv \
# 		s03e04/test.csv \
# 		playground/cli/join.py
# 	poetry run python \
# 		-m playground.cli.join \
# 		--old-file=s03e04/train.csv \
# 		--new-file=s03e04/test.csv \
# 		--output-file=s03e04/train_test.joined.csv

# s03e04/train_test.transformed.csv: \
# 		s03e04/train_test.joined.csv \
# 		s03e04/features.json \
# 		playground/pipelines/s03e04.py
# 	poetry run python \
# 		-m playground.cli.transform \
# 		--config-file=s03e04/features.json \
# 		--customization=playground.pipelines.s03e04.model_customization \
# 		--input-file=s03e04/train_test.joined.csv \
# 		--output-file=s03e04/train_test.transformed.csv


# s03e04/test.augmented_transformed.csv: \
# 		s03e04/train_test.transformed.csv \
# 		playground/cli/filter.py
# 	poetry run python \
# 		-m playground.cli.filter \
# 		--input-file=s03e04/train_test.transformed.csv \
# 		--output-file=s03e04/test.augmented_transformed.csv \
# 		--column=source__passthrough \
# 		--value=new

# s03e04/test.transformed.csv: \
# 		s03e04/test.augmented_transformed.csv
# 	poetry run python \
# 		-m playground.cli.drop_column \
# 		--input-file=s03e04/test.augmented_transformed.csv \
# 		--output-file=s03e04/test.transformed.csv \
# 		--column=source__passthrough

# s03e04/submission.csv: \
# 		s03e04/test.transformed.csv \
# 		s03e04/saved_model \
# 		playground/pipelines/s03e04.py
# 	poetry run python \
# 		-m playground.cli.predict \
# 		--customization=playground.pipelines.s03e04.model_customization \
# 		--input-file=s03e04/test.transformed.csv \
# 		--output-file=s03e04/submission.csv \
# 		--model-dir=s03e04/saved_model