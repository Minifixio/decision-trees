# Decision Trees Project

This Python project builds decision trees from CSV files following a specific format. The data files should be in CSV format, where each line represents a data point. The first column is the label of the point, and the following columns are features. The first line indicates the types of the data, where 'l' is for "label", 'b' is for "binary", 'c' is for "categorical", and 'r' is for "real".

## Usage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Minifixio/decision-trees.git
   ```
2. Navigate to the project directory:
   ```bash
   cd decision-trees
   ```

### Running the Algorithm

Use the following commands to run the algorithm:

```bash
python main.py file_path [-hgt HEIGHT] [-msp MIN_SPLIT_POINTS] [-tsp TREE_SIZE_PROPORTION] [-fudyadt]
```

- `file_path`: Path to the CSV file.

Optional arguments:
- `-hgt`, `--height`: Height of the decision tree (optional).
- `-msp`, `--min_split_points`: Minimum split points (optional).
- `-tsp`, `--tree_size_proportion`: Tree size proportion (optional).
- `-fudyadt`: Use the FuDyADT method (optional).

### File Format

Ensure that your CSV file follows the specified format, where the first line denotes the types of the data, and subsequent lines represent data points.

Example:
```csv
b,r,r,c,c,c,c,c,c,c,r,r,r,c,c,l,b,b,b,b,b,b
1,5000,1,0,0,0,5,1,2,2,100,5,5,0,0,1,1,1,1,0,1,0
...
```

I have included some example files in the examples folder.

## Project Structure

- `main.py`: Main script for running the algorithm.
- `PointSet.py`: Module for the PointSet object, which handles labels, features, and partitions.
- `Tree.py`: Module implementing decision trees.
- `evaluation.py`: Module for computing F1 scores.

## Credits

This project is based on the Data Mining class at Télécom Paris by Professor Mauro Sozio.

## References

- [FuDyADT Algorithm Paper](https://arxiv.org/pdf/2212.00778.pdf) by Gabriel Damay, Marco Bressan, and Mauro Sozio.
