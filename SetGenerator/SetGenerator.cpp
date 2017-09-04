#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;


template <typename T>
std::string tostring(T value){
	std::ostringstream os;
	os << value;
	return os.str();
}

void RenderInterface(){
	
	cout << "+----------Depth DataSet Generator----------+" << endl; 
	cout << "|   Copyright Murilo Kinape Rivabem 2017    |" << endl;
	cout << "|                                           |" << endl;
	cout << "|	 Press Esc to exit the program      |" << endl;
	cout << "+-------------------------------------------+" << endl;

}

int main(int argc, char* argv[]){
	
	if(argc<2){
		cerr << "Usage: " << argv[0] << " | <handconfiguration>" << endl;
		return -1;
	}

	VideoCapture webcam(0); //Pega a câmera "0" pq so tem 1 conectada
	if(!webcam.isOpened()){
		cerr << "Interface de Video nao detectada";
		return -1;
	}

	RenderInterface();

	char c;
	int n=0;
	std::string arg(argv[1]);
	Mat frame;
	do{
		webcam >> frame;
		imshow("Webcam", frame);
		c = (char)waitKey(27);

		if(c == 32){
			string filename = "dataset/" + arg + "_" + tostring(n) + ".png";
			//stringstream filename = "hc1_" << n << ".png";
			imwrite(filename, frame);

			cout << "Writing to " << filename << endl;
			n++;
		}
	}while(c != 27);
}
