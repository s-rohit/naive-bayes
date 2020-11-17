/**
 * @date  : 2020-11-17
 * @author: Rohit S <rohit2013107@iitgoa.ac.in>
 * @title : Gaussian Naive Bayes Implementation
 * Gaussian Naive Bayes classifier that can work with almost any dataset
 * Highly scalable and optimized for parallel processing
 */

// NOTE: For this simple demo, the class definition is included in the header itself.
// Remember to split this into appropriate header and cpp files for larger projects.

#include <valarray>   // for valarray (see below)
#include <algorithm>  // for argmin function

// valarray allows fast element-wise operation between vectors
#define Array std::valarray<float> 

// largest value that float can take
#define INFTY std::numeric_limits<float>::max()

// NaiveBayes class: must be intialized with 
// number of data classes, features, smoothing factor
template <uint classes, uint features, uint smoothen>
class NaiveBayes{
	
	bool trained = false; // lazy-evaluation: compute terms just before testing begins
	// --- data learned during training ---
	float freq[classes];  // class frequency, init to zero
	Array sum1[classes];  // sum of feature values per class (for mean)
	Array sum2[classes];  // sum of square of feature values per class (for sdev)
	// --- data needed during testing ---
	float term[classes];  // constant term per class
	Array mean[classes];  // expected feature values per class 
	Array varn[classes];  // augmented variance of feature value per class 

public:

	// initialize
	NaiveBayes(){
		for(int k=0; k<classes; ++k){
			// class freq init to 0
			freq[k] = 0;
			// each array has |features| elements
			sum1[k].resize(features,0);
			sum2[k].resize(features,0);
			mean[k].resize(features,0);
			varn[k].resize(features,1);
		}
	}

	// highly scalable training operation:
	// 1. does not copy the training sample's data -- only adds them to current values
	// 2. does not compute means and variances immediately -- lazy evaluation (see below)
	void train(char label, float *raw_values){
		// convert values to valarray datatype for parallel processing
		auto values  = Array(raw_values, features);
		// add to training data
		trained      = false;               // mark self as "untrained"
		freq[label] += 1;                   // increment class frequency
		sum1[label] += values;              // add feature values to sum1
		sum2[label] += std::pow(values,2);  // add squares of value to sum2
	}

	// lazy evaluation: compute means and variances just before testing begins.
	// speeds up train() calls by postponing expensive operations like div, log, etc
	void finalize(){
		// for each class
		for(int k = 0; k < classes; ++k){
			// if at least one sample was seen for this class
			if(freq[k]){
				// mean value of feature, μ = Σx / N
				mean[k] = sum1[k] / freq[k];
				// variance of feature, σ^2 = Σ(x^2) / N - μ^2
				// augmented variance, z = 2*σ^2 + smoothing_factor
				varn[k] = 2*( sum2[k] / freq[k] - std::pow(mean[k],2) ) + smoothen; 
				// class constant  = -log(N) + Σ(z)/2
				term[k] = -std::log(freq[k]) + std::log(varn[k]).sum()/2;
			}else{
				// if no samples were seen, set class_constant to infinity
				term[k] = INFTY;
			}
		}
		// mark self as trained
		trained = true;
	}

	// returns the class label predicted for the given input values
	uint predict(float *raw_values){
		// lazy evaluation: train once before testing begins
		if(!trained) finalize();
		// convert values to valarray datatype for parallel processing
		auto values = Array(raw_values, features);
		// find the probability multiplier for each class (see report)
		float prob[classes];
		// prob[k] = class_const[k] + Σ (x-μ)^2/z
		for(int k = 0; k < classes; ++k) 
			prob[k] = term[k] + (std::pow(values - mean[k], 2)/varn[k]).sum();
		// predicted class is the class with the smallest multiplier
		// argmin index = ptr to min - ptr to start
		return std::min_element(prob, prob+classes) - prob; 
	}

};