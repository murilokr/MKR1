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

    /*
    vector<Centroids> *coordinates = new vector<Centroids>();
    filename = "./Dataset/data.arff";
    fstream dataset(filename.c_str(), ios::in);
    if(!dataset.is_open())
        return -1;
    
    string ignore;
    for(int i = 0; i < 10; ++i)
        getline(dataset, ignore);

    float rVx, rVy, rVz, rHc;
    float lVx, lVy, lVz, lHc;

    while(!dataset.eof()){
        dataset >> rVx >> rVy >> rVz >> rHc;
        dataset >> lVx >> lVy >> lVz >> lHc;

        Centroids c = {rVx, rVy, rVz, rHc, lVx, lVy, lVz, lHc};
        coordinates->push_back(c);
    }
    dataset.close();

    vector<int> *observations = Codebook->returnObservations(coordinates);
    for(vector<int>::iterator it = observations->begin(); it != observations->end(); ++it)
        cout << (*it) << endl;
    */

    double TRANSdata[] = {0.5 , 0.5 , 0.0,
                          0.0 , 0.7 , 0.3,
                          0.0 , 0.0 , 1.0};
    cv::Mat TRANS = cv::Mat(3,3,CV_64F,TRANSdata).clone();
    double EMISdata[] = {2.0/4.0 , 2.0/4.0 , 0.0/4.0 , 0.0/4.0 ,
                         0.0/4.0 , 2.0/4.0 , 2.0/4.0 , 0.0/4.0 ,
                         0.0/4.0 , 0.0/4.0 , 2.0/4.0 , 2.0/4.0 };
    cv::Mat EMIS = cv::Mat(3,4,CV_64F,EMISdata).clone();
    double INITdata[] = {1.0  , 0.0 , 0.0};
    cv::Mat INIT = cv::Mat(1,3,CV_64F,INITdata).clone();

    HMM *advanceModel, *returnModel, *zoomInModel, *zoomOutModel;
    
    advanceModel = new HMM(TRANS, EMIS, INIT, "advance.hmm");
    returnModel = new HMM(TRANS, EMIS, INIT, "return.hmm");
    zoomInModel = new HMM(TRANS, EMIS, INIT, "zoomIn.hmm");
    zoomOutModel = new HMM(TRANS, EMIS, INIT, "zoomOut.hmm");

    Mat seq;
    getGestureObservationsFromTrainingData(Codebook, "./Dataset/advanceData.txt", 40, seq);
    cout << "Untrained Model:" << endl;
    advanceModel->print();
    
    cout << endl << endl << "Training Model..." << endl << endl;
    advanceModel->train(seq, 500);

    cout << "Trained Model:" << endl;
    advanceModel->print();
    //printMat(seq);
    
    return 0;
}