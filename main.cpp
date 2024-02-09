#include <iostream>
#include "CNN.h"

using namespace std;

int main() {
	srand(29);
	CNN cnn;
	for (int i = 0; i < 100; i++) {
		cnn.forwardPropagation();
		cnn.backPropagation();

		cout << cnn.convWeightsDiff[1][3][1][4][2] << " ";
		//cout << cnn.fullWeightsDiff[0][1][0] << " ";

		double loss1 = cnn.lossValue;

		cnn.convWeights[1][3][1][4][2] += 0.000001;
		//cnn.fullWeights[0][1][0] += 0.000001;
		cnn.forwardPropagation();

		double loss2 = cnn.lossValue;
		cnn.convWeights[1][3][1][4][2] -= 0.000001;
		//cnn.fullWeights[0][1][0] -= 0.000001;
		cout << (loss2 - loss1) / 0.000001 << endl;
		cnn.forwardPropagation();
		cnn.updateWeights();
	}
	cout << "!\n";
}