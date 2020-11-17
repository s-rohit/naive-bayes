/**
 * @date  : 2020-11-17
 * @author: Rohit S <rohit2013107@iitgoa.ac.in>
 * @title : Gaussian Naive Bayes Implementation
 * Driver program to test the NaiveBayes class
 * against an EMNIST dataset of 370,000+ grayscale 
 * 28x28 images of A-Z handwritten alphabets
 */

#include <stdio.h>         // for basic io
#include <stdlib.h>        // for random number generator
#include <time.h>          // to seed the random generator
#include <forward_list>    // to hold testing data
#include "NaiveBayes.hpp"  // my NaiveBayes class
#include "MemoryFile.hpp"  // read files easily

// experiment config
#define CLASSES  (26)      // number of classes
#define FEATURES (28*28)   // number of pixels 
#define SMOOTHEN (2048)    // smoothing factor

// data type to hold the information about a sample
typedef struct{uint label; float value[FEATURES];} Sample;

// repeatedly fetch samples from the file
// returns true iff a sample was successfully read
bool fetch(MemoryFile *file, Sample *sample);


// main driver program
int main(int argc, char const *argv[]){

	try{
		
		// setup & init
		// ------------
		srand(time(NULL));                                         // seed random number generator
		auto datafile   = MemoryFile(argc > 1 ? argv[1] : 0);      // dataset path (cmd line input)
		auto classifier = NaiveBayes<CLASSES,FEATURES,SMOOTHEN>(); // my NaiveBayes classifier
		auto testQ      = std::forward_list<Sample*>();            // stack of samples to be tested
		uint testclass[CLASSES] = {0};                             // tests performed per class
		uint confusion[CLASSES][CLASSES] = {0};                    // confusion matrix

		// read samples from file
		// ----------------------
		for(auto sample = new Sample; fetch(&datafile, sample); ){ 
			// 90% samples (randomly selected) used for training data
			if(rand()%10) classifier.train(sample->label, sample->value);
			// remaining 10% added to testing queue
			else sample = (testQ.push_front(sample), new Sample);
		}

		// perform tests
		// -------------
		for(auto test: testQ){                                  // for each test
			char prediction = classifier.predict(test->value);  // predicted label
			confusion[test->label][prediction]++;               // add to confusion matrix
			testclass[test->label]++;                           // add to row-wise sum
			delete test;                                        // free memory
		}

		// accuracy
		// --------
		float accuracy; 
		uint successes = 0, total = 0;
		printf("Accuracy per class:\n"); 
		for(int i=0; i<CLASSES; ++i){                            // for each class
			successes += confusion[i][i];                        // -- total successes
			total     += testclass[i];                           // -- total tests
			accuracy   = confusion[i][i]/(float)testclass[i];    // -- accuracy = successes/tests
			printf("%c %.2f%%\n", 'A'+i, accuracy*100);          // @ print as percent
		}
		accuracy = successes/(float)total;                       // total accuracy
		printf("Overall accuracy: %.2f%%\n", accuracy*100);      // @ print as percent

	// show fatal errors, if any
	} catch(const char *e){fprintf(stderr, "%s\n", e);}

	// exit
	return 0;
}


// repeatedly fetch samples from the file
// returns true iff a sample was successfully read
inline bool fetch(MemoryFile *file, Sample *sample){
	// if end of file, return false
	if(file->curr >= file->end) return false;
	// flag to differentiate label value and pixel values
	bool pxls = false;
	// numeric value read so far and current pixel number
	uint num = 0, attr = 0;
	// read from current file position till end of file
	for(char *ch = file->curr; ch < file->end; ++ch){
		// case1: digit
		if(*ch >= '0' && *ch <= '9'){
			// read a number digit-by-digit
			num = (num*10) + (*ch-'0');
		}
		// case2: comma
		else if(*ch==','){
			// label value if it's the first number read, else pixel value
			pxls ? (sample->value[attr++] = num) : (sample->label = num);
			// pixel values till end of line
			pxls = true;
			// reset to read a new number
			num  = 0;
		}
		// case3: newline
		else if(*ch=='\n'){
			// assign last read number to last pixel value
			sample->value[attr] = num;
			// move file's current pointer to after the newline
			file->curr = ch+1;
			// break (and return)
			break;
		}
		// all other cases ignored
	}
	// return true, for sample read
	return true;
}