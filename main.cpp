#include <iostream>
#include "CNN.h"

using namespace std;

int main() {
	CNN cnn;
	cnn.forwardPropagation();
	cnn.backPropagation();

	vector<vector<double>> v = {
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9}
	};
	v = cnn.tensorTranspose(v);
}