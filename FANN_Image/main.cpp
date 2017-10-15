#include "Kinect.h"
#include <sstream>
#include "NeuralNetwork.hpp"



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
            for(int r = 0; r < handSgm.rows; r++){
                for(int c = 0; c < handSgm.cols; c++){
                    int bin = (int)handSgm.at<uchar>(r,c);
                    if(bin > 128)
                        bin = 1;
                    else 
                        bin = 0;
                    file << bin << "\t";
                }
                //file << "\n"; //*********************Debug mode
            }
            file << endl;
            //Output the output for this specific image
            for(vector<float>::iterator it = output->begin(); it != output->end(); ++it){
                float output = (*it);
                file << output << "\t";
            }
            file << endl;
            //file << endl << endl << endl << endl; //*************************Debug Mode

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
            int inputs = fileColumns(tempFilename, 1);//maxDimensions;
            if(inputs == -1){
                cerr << "ERROR: fileColumns(). Returning to normal dimensions" << endl;
                inputs = maxDimensions;
            }
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
    cout << "Usage: " << argv[0] << " <1=Record Data | 2=Write Data | 3=Train Data>" << endl;
    cout << "Record data writes all pixels in a image to a temp file." << endl;
    cout << "Write data writes all of the data in the temp file to a final csv file. (Use only at the end of data gathering)" << endl;
    cout << "Train data uses the csv generated file from 'Write Data' to train a Neural Network." << endl;
}


void getHands(Mat& leftHand, Mat& rightHand, Mat frame, int windowSize){
    XnUserID aUsers[MAX_NUM_USERS];
    XnUInt16 nUsers;
    XnSkeletonJointTransformation jointPosition;

    g_Context.WaitOneUpdateAll(g_UserGenerator);
    nUsers=MAX_NUM_USERS;
    g_UserGenerator.GetUsers(aUsers, nUsers);
    for(XnUInt16 i=0; i<nUsers; i++)
    {
        if(g_UserGenerator.GetSkeletonCap().IsTracking(aUsers[i])==FALSE)
            continue;
                                                                   //Inverter as mãos, pois o kinect "detecta" errado.
        g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i], XN_SKEL_LEFT_HAND, jointPosition);
        XnPoint3D aProjective;        
        g_DepthGenerator.ConvertRealWorldToProjective(1,&jointPosition.position.position, &aProjective);
        int rHandX = aProjective.X;
        int rHandY = aProjective.Y;
        int rHandZ = aProjective.Z;

                                                                   //Inverter as mãos, pois o kinect "detecta" errado.
        g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i], XN_SKEL_RIGHT_HAND, jointPosition);
        g_DepthGenerator.ConvertRealWorldToProjective(1,&jointPosition.position.position, &aProjective);
        int lHandX = aProjective.X;
        int lHandY = aProjective.Y;
        int lHandZ = aProjective.Z;

        cv::Rect roi(rHandX - (windowSize/2), rHandY - (windowSize/2), windowSize, windowSize);
        cv::Rect roiImg(0, 0, frame.cols, frame.rows);
        if( (roi.area() > 0) && ((roiImg & roi).area() == roi.area()) )
            rightHand = frame(roi);    
        threshold(rightHand, rightHand, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        cout << (int)rightHand.at<uchar>(28,28) << endl;

        roi = Rect(lHandX - (windowSize/2), lHandY - (windowSize/2), windowSize, windowSize);
        if( (roi.area() > 0) && ((roiImg & roi).area() == roi.area()) )
            leftHand = frame(roi);
        threshold(leftHand, leftHand, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    }
}


int main(int argc, char* argv[]){

    if(argc < 2){
        writeInfo(argv);
        return -1;
    }

    string tmpRightHandFilename = "./Data/rh_temp.txt";
    string tmpLeftHandFilename = "./Data/lh_temp.txt";
    string csvRightHandFilename = "./Data/rh_train.dat";
    string csvLeftHandFilename = "./Data/lh_train.dat";
    int defaultDimensions = (75*0.75)*(75*0.75);
    int outputs = 4;

    istringstream ss(argv[1]);
    int arg;
    if(!(ss >> arg)){
        cerr << "ERROR: Error converting second argument (Must be a number)." << endl << endl;
        writeInfo(argv);
        return -1;
    }

    if(arg < 1 || arg > 3){
        cerr << "ERROR: Invalid argument." << endl << endl;
        writeInfo(argv);
        return -1;
    }

    if(arg == 2){
        cout << "Appending the hand data..." << endl;
        CSV leftCSV(tmpLeftHandFilename, csvLeftHandFilename, defaultDimensions, outputs);
        CSV rightCSV(tmpRightHandFilename, csvRightHandFilename, defaultDimensions, outputs);
        if(!leftCSV.saveAll()){
            cerr << "ERROR: Error in saveAll()." << endl;
            return -1;
        }
        if(!rightCSV.saveAll()){
            cerr << "ERROR: Error in saveAll()." << endl;
            return -1;
        }
        cout << "Completed." << endl;
        return 0;
    }

    if(arg == 3){
        HandConfiguration hcRightHand("./Data/rightHand.net");
        if(!hcRightHand.train(csvRightHandFilename)){
            cerr << "Error in training neural network" << endl;
            return -1;
        }
        return 0;
    }

    int op = -1;
    do{
        cout << "What gesture are you recording (0-Advance, 1-Return, 2-Zoom In, 3-Zoom-Out)?" << endl << "-> ";
        cin >> op;
    }while(op<0 || op>=4);
    vector<float>* output = new vector<float>();
    for(int i = 0; i < 4; i++){
        if(i == op)
            output->push_back(1.0);
        else
            output->push_back(0.0);
    }
    
    cout << "output = <";
    for(vector<float>::iterator it = output->begin(); it != output->end(); ++it)
        cout << " " << (*it); 
    cout << " >" << endl;

    if(!createKinect())
        return -1;
    if(!createOpenCV())
        return -1;

    bool endit = false;
    bool isRecording = false;

    bool recordR = false;
    bool recordL = false;

    int windowSize = 75;
    double scaleDownFactor = 0.75;

    CSV leftHandCSV(tmpLeftHandFilename, csvLeftHandFilename, floor(pow(windowSize * scaleDownFactor,2)), outputs);        
    CSV rightHandCSV(tmpRightHandFilename, csvRightHandFilename, floor(pow(windowSize * scaleDownFactor,2)), outputs);
    if(op == 1 || (op > 1 && op < 4))       
        recordL = true;
    if(op == 0 || (op > 1 && op < 4))
        recordR = true;

    Mat frame, leftHand, rightHand;

    namedWindow("Depth Image", 1);
    namedWindow("L", 1);
    namedWindow("R", 1);

    while(!endit){
        while (xnOSWasKeyboardHit()){

            char c = xnOSReadCharFromInput();
            if(c == 27) endit = !endit;
            else if (c == 32) isRecording = !isRecording;
        }

        cap.grab();
        cap.retrieve(frame, CV_CAP_OPENNI_DISPARITY_MAP);
        getHands(leftHand, rightHand, frame, windowSize);
    

        if(!leftHand.empty()){
            resize(leftHand, leftHand, Size(), 0.75, 0.75);
            cout << "Left Hand: " << leftHand.cols << "x" << leftHand.rows << endl;
        }
        if(!rightHand.empty()){
            resize(rightHand, rightHand, Size(), 0.75, 0.75);
            cout << "Right Hand: " << rightHand.cols << "x" << rightHand.rows << endl;
        }
        
        if(isRecording){
            if(!leftHand.empty())
                if(recordL)
                    if(!leftHandCSV.appendTemp(leftHand, output))
                        cerr << "Error appending to Left Hand" << endl;
            
            if(!rightHand.empty())
                if(recordR)
                    if(!rightHandCSV.appendTemp(rightHand, output))
                        cerr << "Error appending to Right Hand" << endl;
            
        }

        cv::imshow("Depth Image", frame);
        if(!leftHand.empty())
            cv::imshow("L", leftHand);
        if(!rightHand.empty())
            cv::imshow("R", rightHand);
        char esc = cv::waitKey(33);
        if (esc == 27) break;
        else if (esc == 32) isRecording = !isRecording;
    }

    g_ScriptNode.Release();
    g_UserGenerator.Release();
    g_DepthGenerator.Release();
    g_Context.Release();
    return 0;
}