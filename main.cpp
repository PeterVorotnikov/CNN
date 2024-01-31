#include <iostream>
#include "CNN.h"

using namespace std;

int main() {
	CNN cnn;
	cnn.forwardPropagation();
	cnn.backPropagation();
}