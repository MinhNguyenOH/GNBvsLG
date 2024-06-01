# Gaussian Naive Bayes vs Logistic Regression for differential privacy

## Overview

This project focuses on the accuracy of Gaussian Naive Bayes and Logistic Regression models when implemented with differential privacy. The project evaluates the impact of privacy constraints on these models across various datasets.

## Project Structure

- **analysis.md**: Documentation and analysis of the project and datasets.
- **final_project_report.pdf**: Detailed project report.
- **dataset.py**: Script for handling datasets.
- **main.py**: Main script to run the machine learning models.
- **run.sh**: Shell script to execute the main script with different models and datasets.
- **breast.csv**: Breast cancer dataset.
- **letter.csv**: Letter recognition dataset.
- **magic.csv**: MAGIC gamma telescope dataset.
- **rice.csv**: Rice dataset.
- **wine.csv**: Wine quality dataset.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- NumPy

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/MinhNguyenOH/GNBvsLG.git
   cd GNBvsLG
   ```

2. Install the required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Project

1. Ensure you are in the project directory.
2. Use the provided shell script to run the main script with different models and datasets:

   ```sh
   sh run.sh <model> <dataset>
   ```

   - `<model>`: `gnb` for Gaussian Naive Bayes or `logit` for Logistic Regression.
   - `<dataset>`: One of `breast`, `wine`, `rice`, `letter`, `magic`.

### Running Individual Components

- **Main Script**:

  ```sh
  python3 main.py <model> <dataset> <start_column> <end_column>
  ```

## Data

### Breast Cancer Data

- **File**: `breast.csv`
- **Description**: This dataset contains features computed from a digitized image of a fine needle aspirate of a breast mass.

### Wine Quality Data

- **File**: `wine.csv`
- **Description**: This dataset contains physicochemical and sensory variables for Portuguese "Vinho Verde" wine.

### Rice Data

- **File**: `rice.csv`
- **Description**: This dataset contains morphological features of rice grains.

### Letter Recognition Data

- **File**: `letter.csv`
- **Description**: This dataset contains numerical attributes for recognizing English alphabet letters.

### MAGIC Gamma Telescope Data

- **File**: `magic.csv`
- **Description**: This dataset contains information for statistical classification of gamma particles and cosmic rays.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to open an issue or contact the project maintainers:
- Minh Dang Truong: truong_m1@denison.edu
- Andrew Pham: pham_l2@denison.edu
- Minh Nguyen: nguyen_v2@denison.edu
```
