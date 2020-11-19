# (Gaussian) Naive Bayes Classifier
by **S Rohit** (2013107, MTech-Y1) for IIT-GOA CS-561 project, 2020.

The project involved implementing a (Gaussian) Naïve Bayes Classifier — a fast and simple classifier that uses Bayes’ theorem under the assumption that all features are mutually independent for a given class. Then, this classifier was tested against an EMNIST [dataset](https://drive.google.com/file/d/18ZY7I1ym0E9s2ecqjfvmPSGwlXzsNi2n) of A–Z handwritten character alphabets.

**Result:** Overall accuracy: \~70%.

## Report
The project report is attached as a PDF along with its LaTeX source files.  

##### Contents of the report
1. Elementary concepts needed to understand the Gaussian Naive Bayes' classifier.  
2. Derivation of a working formula for the classiier that is suitable for implementation.
3. Results of testing the classifier the above mentioned dataset: overall and character-wise accuracy, visualization of the mean and variance of each character, and a confusion matrix.
4. Conclusion along with references to some key texts used for this project.

## Code
Codes for the above classifier are available in Python and C++.

1. **NaiveBayes class**: `NaiveBayes.hpp` / `NaiveBayes.py`  
	Gaussian Naive Bayes classifier that can work with almost any dataset.  
	Highly scalable and optimized for parallel processing.

2. **Driver program:** `main.cpp` / `main.py`  
	A driver program to test the NaiveBayes class against the above dataset.  
	Path to the dataset must be provided as command-line argument.


#### Usage
1. C++ without makefile  
```bash
g++ -O3 main.cpp -o main.exe         # use of O3 optimization flags highly recommended
./main.exe path/to/dataset.csv       # prints the class-wise and overall accuracy
```
2. C++ using makefile  
```bash
make                                 # compiles the C++ code as in step 1
./main.exe path/to/dataset.csv       # prints the class-wise and overall accuracy
make clean                           # (optional) delete the executable
```
3. Python
```bash
python3 main.py path/to/dataset.csv  # displays accuracy, visualizations, & confusion matrix
```
<br/>

## Also includes,
1. Screenshot of the execution of these codes on a Linux platform.
2. Outputs (mean, variance, & confusion matrix) of one such experiment as PNG.
