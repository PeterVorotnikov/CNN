#include <vector>
#include <cmath>
#include <random>
#include <time.h>

using namespace std;

#pragma once
class CNN
{
private:

//***************************************** HyperParameters *****************************

	int nOfClasses = 10;

	int nOfConvLayers = 2;

	int nOfFullLayers = 2;

	int sizeOfPooling = 2;

	vector<int> nOfFilters = { 6, 16 };

	vector<int> sizeOfKernels = { 5, 5 };

	vector<int> sizeOfLayers = { 100, 80 };


	int imageWidth = 28;

	int imageHeight = 28;


	





//***************************************** Parameters *********************************

	// Conv layers

	vector<vector<vector<vector<vector<double>>>>> convWeights;

	vector<vector<vector<vector<vector<double>>>>> convWeightsDiff;

	vector<vector<double>> convBiases;

	vector<vector<double>> convBiasesDiff;



	// Full layers

	vector<vector<vector<double>>> fullWeights;

	vector<vector<vector<double>>> fullWeightsDiff;

	vector<vector<double>> fullBiases;

	vector<vector<double>> fullBiasesDiff;




	// Output layers

	vector<vector<double>> outputWeights;

	vector<double> outputBiases;

	vector<vector<double>> outputWeightsDiff;

	vector<double> outputBiasesDiff;






//************************************ States ********************************

	// Input image

	vector<vector<double>> initialImage;



	// Conv layers

	vector<vector<vector<vector<double>>>> convLayersInputs;

	vector<vector<vector<vector<double>>>> convLayersOutputs;

	vector<vector<vector<vector<double>>>> convNeuronsDiff;



	// Pooling layers

	vector<vector<vector<vector<double>>>> poolingLayersInputs;

	vector<vector<vector<vector<double>>>> poolingLayersOutputs;

	vector<vector<vector<vector<int>>>> poolingMemory;



	// Full layers

	vector<vector<double>> fullLayersInputs;

	vector<vector<double>> fullLayersOutputs;

	vector<vector<double>> fullNeuronsDiff;



	// Output layer

	vector<double> outputInputs;

	vector<double> outputOutputs;

	vector<double> outputDiff;

	

	




//***************************************** Activation functions *******************


	double convActivation(double x);

	double convActivationDiff(double x);



	double layerActivation(double x);

	double layerActivationDiff(double x);



	vector<double> softmaxOutputs(vector<double> softmaxInputs, int nOfClass);










public:

//*********************************** Constructors *******************************

	void parametersInit();

	void poolingMemoryInit();

	void statesInit();

	void init();

	CNN();


};

