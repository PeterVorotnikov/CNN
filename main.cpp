#include <iostream>
#include "CNN.h"

using namespace std;

int main() {
	CNN cnn;
	cnn.load("model.txt");
	cout << cnn.convBiases[0][0];
}