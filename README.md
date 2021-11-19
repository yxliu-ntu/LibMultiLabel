# README

## ENV Installation

```(shell)
make clean all
```

## ENV Activation

```(shell)
source ./venv/bin/activate
```

## Download Sample Data

```(shell)
cd data/
./generate_data.sh
```

## Grid Search

```(shell)
./grid.sh ./example_config/ml-1m/ffm_2tower.yml Naive-LRLR 0
```
