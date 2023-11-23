#include <vector>
#include <cmath>
#include <random>
#include <time.h>

using namespace std;

#pragma once
class CNN
{
private:

//***************************************** Parameters *****************************

	int nOfClasses = 10;

	int nOfConvLayers = 2;

	int nOfFullLayers = 2;

	int sizeOfPulling = 2;

	vector<int> nOfFilters = { 6, 16 };

	vector<int> sizeOfKernels = { 5, 5 };

	vector<int> sizeOfLayers = { 100, 80 };






//***************************************** States *********************************

	vector<vector<vector<vector<vector<double>>>>> convWeights;

	vector<vector<double>> convBiases;

	vector<vector<vector<double>>> fullWeights;

	vector<vector<double>> fullBiases;

	vector<vector<double>> outputWeights;

	vector<double> outputBiases;

	





//***************************************** Activation functions *******************
	double convActivation(double x);

	double convActivationDiff(double x);



	double layerActivation(double x);

	double layerActivationDiff(double x);



	vector<double> softmaxOutputs(vector<double> softmaxInputs, int nOfClass);
};

