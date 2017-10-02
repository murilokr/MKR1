#include "HMM.h"


//Return a matrix with NxM dimensions, where:
//N = the number of times a gesture was made / sequence size (unknown)
//M = the size of the gesture (known)
void getGestureObservationsFromTrainingData(KMeans *codebook, string filename, int gestureSize, cv::Mat &observationsMat){
    fstream file(filename.c_str(), ios::in);
    if(!file.is_open())
        return;

    vector<Centroids>* coordinates = new vector<Centroids>();
    
    float rVx, rVy, rVz, rHc;
    float lVx, lVy, lVz, lHc;

    while(!file.eof()){
        file >> rVx >> rVy >> rVz >> rHc;
        file >> lVx >> lVy >> lVz >> lHc;

        Centroids c = {rVx, rVy, rVz, rHc, lVx, lVy, lVz, lHc};
        coordinates->push_back(c);
    }
    file.close();

    vector<int> *observationsVector = codebook->returnObservations(coordinates);
    int nmbSeq = (int)(observationsVector->size()/gestureSize);
    observationsMat = cv::Mat(nmbSeq, gestureSize, CV_32SC1);
    
    vector<int>::iterator it = observationsVector->begin();
    for(int r = 0; r < observationsMat.rows; r++){
        for(int c = 0; c < observationsMat.cols; c++){
            observationsMat.at<int>(r,c) = (*it);
            ++it;
        }
    }
}


void TrainModels(KMeans *codebook, HMM *advanceHMM, HMM *returnHMM, HMM *zoomInHMM, HMM *zoomOutHMM){
    Mat seq;
    getGestureObservationsFromTrainingData(codebook, "./Dataset/advanceData.txt", 40, seq);
    advanceHMM->train(seq, 500);

    getGestureObservationsFromTrainingData(codebook, "./Dataset/returnData.txt", 40, seq);
    returnHMM->train(seq, 500);

    getGestureObservationsFromTrainingData(codebook, "./Dataset/zoomInData.txt", 40, seq);
    zoomInHMM->train(seq, 500);

    getGestureObservationsFromTrainingData(codebook, "./Dataset/zoomOutData.txt", 40, seq);
    zoomOutHMM->train(seq, 500);


    advanceHMM->save();
    returnHMM->save();
    zoomInHMM->save();
    zoomOutHMM->save();
}

void printMat(Mat& data){
    for (int i=0;i<data.rows;i++)
    {
        std::cout << i << ": ";
        for (int j=0;j<data.cols;j++)
            std::cout << data.at<int>(i,j) << "|";
        std::cout << "\n";
    }
}


int main(int argc, char* argv[]){
    string filename = "./Dataset/codebook.txt";
    fstream data(filename.c_str(), ios::in);

    KMeans *Codebook = new KMeans(data);
    data.close();    

    HMM *advanceModel, *returnModel, *zoomInModel, *zoomOutModel;
    advanceModel = new HMM("advance.hmm");
    returnModel = new HMM("return.hmm");
    zoomInModel = new HMM("zoomIn.hmm");
    zoomOutModel = new HMM("zoomOut.hmm");

    if(!advanceModel->isAlreadyModeled() || !returnModel->isAlreadyModeled() || !zoomInModel->isAlreadyModeled() || !zoomOutModel->isAlreadyModeled())
        TrainModels(Codebook, advanceModel, returnModel, zoomInModel, zoomOutModel);


    
    if(!createKinect())
        return -1;
    if(!createOpenCV())
        return -1;
    
    
    Mat observations;
    getGestureObservationsFromTrainingData(Codebook, argv[1], 40, observations);
    printMat(observations);

    double validation;
    for(int i = 0; i < observations.rows; i++){
        cout << "Validation " << i << endl;

        validation = advanceModel->validate(observations.row(0));
        cout << "Advance HMM: " << validation << endl;


        validation = returnModel->validate(observations.row(0));
        cout << "Return HMM: " << validation << endl;

        validation = zoomInModel->validate(observations.row(0));
        cout << "Zoom In HMM: " << validation << endl;

        validation = zoomOutModel->validate(observations.row(0));
        cout << "Zoom Out HMM: " << validation << endl;

        cout << endl << endl;
    }

    return 0;
}