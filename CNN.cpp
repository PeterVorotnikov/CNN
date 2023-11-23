#include "CNN.h"




//***************************************** Activation functions *******************


double CNN::convActivation(double x) {

	if (x > 0) {

		return x;

	}

	else {

		return 0;

	}
}

double CNN::convActivationDiff(double x) {

	if (x > 0) {

		return 1;

	}

	else {

		return 0;

	}
}











double CNN::layerActivation(double x) {

	return 1.0 / (1.0 + exp(-x));

}
double CNN::layerActivationDiff(double x) {

	return layerActivation(x) * (1 - layerActivation(x));

}











vector<double> CNN::softmaxOutputs(vector<double> softmaxInputs, int nOfClass) {

	double denominator = 0;

	for (int i = 0; i < nOfClasses; i++) {

		denominator += exp(softmaxInputs[i]);

	}

	vector<double> answer(nOfClasses);

	for (int i = 0; i < nOfClasses; i++) {

		answer[i] = exp(softmaxInputs[i] / denominator);

	}

	return answer;
}


double outputDiffTrueClass(vector<double> softMaxInputs, int nOfClass, int nOfTrueClass) {

}