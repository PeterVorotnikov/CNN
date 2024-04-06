#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std;

#pragma once
class CNN
{
public:

//***************************************** HyperParameters *****************************

	int nOfClasses = 10;

	int nOfConvLayers = 2;

	int nOfFullLayers = 2;

	int sizeOfPooling = 2;

	vector<int> nOfFilters = { 6, 16 };

	vector<int> sizeOfKernels = { 5, 5 };

	vector<int> sizeOfLayers = { 80, 60 };


	int imageWidth = 28;

	int imageHeight = 28;

	int nOfImageChannels = 1;


	int targetClass = 0;

	double lossValue = 0;


	double learningRate = 0.01;

	double epsilon = pow(10, -8);

	double alpha = 3 * pow(10, -2);

	double beta1 = 0.9;

	double beta2 = 0.99;

	





//***************************************** Parameters *********************************

	// Conv layers

	vector<vector<vector<vector<vector<double>>>>> convWeights;

	vector<vector<vector<vector<vector<double>>>>> convWeightsDiff;

	vector<vector<vector<vector<vector<double>>>>> convWeightsV;

	vector<vector<vector<vector<vector<double>>>>> convWeightsG;

	vector<vector<double>> convBiases;

	vector<vector<double>> convBiasesDiff;

	vector<vector<double>> convBiasesV;

	vector<vector<double>> convBiasesG;



	// Full layers

	vector<vector<vector<double>>> fullWeights;

	vector<vector<vector<double>>> fullWeightsDiff;

	vector<vector<vector<double>>> fullWeightsV;

	vector<vector<vector<double>>> fullWeightsG;

	vector<vector<double>> fullBiases;

	vector<vector<double>> fullBiasesDiff;

	vector<vector<double>> fullBiasesV;

	vector<vector<double>> fullBiasesG;




	// Output layers

	vector<vector<double>> outputWeights;

	vector<double> outputBiases;

	vector<vector<double>> outputWeightsDiff;

	vector<double> outputBiasesDiff;

	vector<vector<double>> outputWeightsV;

	vector<double> outputBiasesV;

	vector<vector<double>> outputWeightsG;

	vector<double> outputBiasesG;






//************************************ States ********************************

	// Input image

	vector<vector<vector<double>>> initialImage;



	// Conv layers

	vector<vector<vector<vector<double>>>> convLayersInputs;

	vector<vector<vector<vector<double>>>> convLayersOutputs;

	vector<vector<vector<vector<double>>>> convLayersDiff;



	// Pooling layers

	vector<vector<vector<vector<double>>>> poolingLayersOutputs;

	vector<vector<vector<vector<double>>>> poolingLayersDiff;

	vector<vector<vector<vector<int>>>> poolingMemory;



	// Full layers

	vector<vector<double>> fullLayersInputs;

	vector<vector<double>> fullLayersOutputs;

	vector<vector<double>> fullLayersDiff;



	// Output layer

	vector<double> outputInputs;

	vector<double> outputOutputs;

	vector<double> outputDiff;

	

	




//***************************************** Activation functions *******************


	double convActivation(double x);

	double convActivationDiff(double x);



	double layerActivation(double x);

	double layerActivationDiff(double x);



	vector<double> softmaxOutputs(vector<double> softmaxInputs);



//***************************************** Tensor transpose *******************

	vector<vector<double>> tensorTranspose(vector<vector<double>> tensor);










public:

//*********************************** Constructors *******************************

	void parametersInit();

	void statesInit();

	void init();

	CNN();





//*********************************** Predict *******************************

	void forwardPropagation();

//*********************************** Train *********************************

	void backPropagation();

	void updateWeights();

public:

	vector<double> predict(vector<vector<vector<double>>> image);

	double fit(vector<vector<vector<double>>> image, int target);

	double getLoss(vector<vector<vector<double>>> image, int target);

//*********************************** Save/load *********************************

	bool save(string fileName);

	bool load(string fileName);

};

