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


	int imageWidth = 28;

	int imageHeight = 28;


	





//***************************************** HyperParameters *********************************

	vector<vector<vector<vector<vector<double>>>>> convWeights;

	vector<vector<double>> convBiases;

	vector<vector<vector<double>>> fullWeights;

	vector<vector<double>> fullBiases;

	vector<vector<double>> outputWeights;

	vector<double> outputBiases;





//************************************ Direct propagation states ********************************


	vector<vector<double>> initialImage;

	vector<vector<vector<vector<double>>>> convLayersOutputs;

	vector<vector<vector<vector<double>>>> poolingLayersOutputs;

	vector<vector<double>> fullLayersOutputs;

	vector<double> outputs;
	

	vector<vector<vector<vector<int>>>> poolingMemory;

	



//************************************ Back propagation states **********************


	vector<vector<vector<vector<double>>>> convNeuronsDiff;

	vector<vector<vector<vector<vector<double>>>>> convWeightsDiff;

	vector<vector<double>> convBiasesDiff;

	vector<vector<double>> fullNeuronsDiff;

	vector<vector<vector<double>>> fullWeightsDiff;

	vector<vector<double>> fullBiasesDiff;

	vector<double> outputNeuronsDiff;

	vector<vector<double>> outputWeightsDiff;

	vector<double> outputBiasesDiff;




//***************************************** Activation functions *******************


	double convActivation(double x);

	double convActivationDiff(double x);



	double layerActivation(double x);

	double layerActivationDiff(double x);



	vector<double> softmaxOutputs(vector<double> softmaxInputs, int nOfClass);










public:

//*********************************** Constructors *******************************

	void weightsInit();

	void poolingMemoryInit();

	void directStatesInit();

	void backStatesInit();

	void init();

	CNN();


};

