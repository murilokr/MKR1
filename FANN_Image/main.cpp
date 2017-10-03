#include "Kinect.h"
#include <sstream>
#include <floatfann.h>



using namespace std;
using namespace cv;


bool isFileEmpty(fstream& file){
    file.seekg(0, ios::end);
    return file.tellg() == 0;
}

int fileLines(string filename){
    string line;
    int nmbLines = 0;
    fstream file(filename.c_str(), ios::in);
    if(!file.is_open())
        return -1;

    while(getline(file, line))
        nmbLines++;
    return nmbLines;
}

class CSV{
        
    public:
        string tempFilename;
        string csvFilename;
        int maxDimensions;
        int numOutputs;

        CSV() : tempFilename(""), csvFilename(""), maxDimensions(40*40), numOutputs(1){}
        CSV(string tmpFilename, string csvFname, int dimensions, int outputs) : tempFilename(tmpFilename), csvFilename(csvFname), maxDimensions(dimensions), numOutputs(outputs){}



        /**
         * appendTemp
         * Function: Adiciona ao arquivo temporario as informações da mão
         * 
         * In: cv::Mat handSgm (Uma imagem da mão segmentada, ou seja, valores 0 ou 1)
         * In: vector<float> output (Qual a configuração dessa imagem <ex: 0.0 1.0 0.0>)
         * 
         * Out: bool success (Retorna true se tudo ocorreu certo, e false se algo de errado ocorreu)
        */
        bool appendTemp(cv::Mat handSgm, vector<float>* output){
            if(tempFilename == ""){
                cerr << "Filename Temp is null." << endl;
                return false;
            }

            fstream file(tempFilename.c_str(), ios::out | ios::app);
            if(!file.is_open()){
                cerr << "Can't open temp file." << endl;
                return false;
            }

            if(handSgm.rows * handSgm.cols > maxDimensions){

                cerr << "Input image dimensions are higher than maximum dimension." << endl;
                return false;
            }

            //Output the actual image in binary
            for(int r = 0; r < handSgm.rows; r++)
                for(int c = 0; c < handSgm.cols; c++){
                    int bin = handSgm.at<int>(r,c);
                    file << bin << "\t";
                }
            
            //Output the output for this specific image
            for(vector<float>::iterator it = output->begin(); it != output->end(); ++it){
                int output = (*it);
                file << output << "\t";
            }
            file << endl;


            //Update the number of outputs
            if(numOutputs == 1)
                numOutputs = output->size();
            
            file.close();
            return true;
        }

        bool saveAll(){
            if(tempFilename == "" || csvFilename == ""){
                cerr << "Filename (Temp or CSV) is null." << endl;
                return false;
            }

            fstream tmpFile(tempFilename.c_str(), ios::in);
            fstream csvFile(csvFilename.c_str(), ios::out | ios::app);
            if(!tmpFile.is_open()){
                cerr << "Can't open temp file." << endl;
                if(csvFile.is_open())
                    csvFile.close();
                return false;
            }
            if(!csvFile.is_open()){
                cerr << "Can't open csv file." << endl;
                if(tmpFile.is_open())
                    tmpFile.close();
                return false;
            }

            if(!isFileEmpty(csvFile)){
                cerr << "CSV File already exists." << endl;
                return false;
            }
            
            int linesTmp = fileLines(tempFilename);
            if(linesTmp == -1){
                cerr << "Error reading temp file lines." << endl;
            }

            int pairs = floor(linesTmp / 2);
            int inputs = maxDimensions;
            int outputs = numOutputs;

            csvFile << pairs << "\t" << inputs << "\t" << outputs << endl;

            string tmpLine;
            while(getline(tmpFile, tmpLine)){
                csvFile << tmpLine << endl;
            }


            tmpFile.close();
            csvFile.close();
            return true;
        }
};

void writeInfo(char *argv[]){
    cerr << "Usage: " << argv[0] << " <1=Record Data | 2=Write Data>" << endl;
    cerr << "Record data writes all pixels in a image to a temp file." << endl;
    cerr << "Write data writes all of the data in the temp file to a final csv file. (Use only at the end of data gathering)" << endl;
}


int main(int argc, char* argv[]){

    if(argc < 2){
        writeInfo(argv);
        return -1;
    }

    string tmpFilename = "./Data/temp.txt";
    string csvFilename = "./Data/train.csv";
    int defaultDimensions = 70*70;
    int outputs = 4;

    istringstream ss(argv[1]);
    int arg;
    if(!(ss >> arg)){
        cerr << "ERROR: Error converting second argument (Must be a number)." << endl << endl;
        writeInfo(argv);
        return -1;
    }

    if(arg < 1 || arg > 2){
        cerr << "ERROR: Invalid argument." << endl << endl;
        writeInfo(argv);
        return -1;
    }

    if(arg == 2){
        CSV csv(tmpFilename, csvFilename, defaultDimensions, outputs);
        if(!csv.saveAll()){
            cerr << "ERROR: Error in saveAll()." << endl;
            return -1;
        }
        return 0;
    }

    if(!createKinect())
        return -1;
    if(!createOpenCV())
        return -1;



    return 0;
}