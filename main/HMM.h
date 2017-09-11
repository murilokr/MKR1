#include "kmeans.h"
#include "CvHMM.h"
#include <sstream>


class HMM{
private:
    Mat TRANS, EMIS, INIT; //Model
    string modelType;

public:

    HMM(Mat &T, Mat &E, Mat &I, string type){
        TRANS = T;
        EMIS = E;
        INIT = I;
        modelType = type;
    }

    HMM(string type){
        modelType = type;
        if(!load())
            cerr << "Error loading " << "./Data/" << modelType;
    }

    void getTransitionMatrix(Mat& data){data = TRANS;}
    void getEmissionMatrix(Mat& data){data = EMIS;}
    void getInitialMatrix(Mat& data){data = INIT;}

    bool load(){
        string filename = "./Data/" + modelType;
        fstream file(filename.c_str(), ios::in);
        if(!file.is_open())
            return false;

        int N, M, value;
        file >> N >> M;
        TRANS = cv::Mat(N,M,CV_64F); //talvez tenha q alterar o tipo
        for(int r = 0; r < TRANS.rows; r++){
            for(int c = 0; c < TRANS.cols; c++){
                file >> value;
                TRANS.at<double>(r,c) = value;
            }
        }

        file >> N >> M;
        EMIS = cv::Mat(N,M,CV_64F); //talvez tenha q alterar o tipo
        for(int r = 0; r < EMIS.rows; r++){
            for(int c = 0; c < EMIS.cols; c++){
                file >> value;
                EMIS.at<double>(r,c) = value;
            }
        }

        file >> N >> M;
        INIT = cv::Mat(N,M,CV_64F); //talvez tenha q alterar o tipo
        for(int r = 0; r < INIT.rows; r++){
            for(int c = 0; c < INIT.cols; c++){
                file >> value;
                INIT.at<double>(r,c) = value;
            }
        }

        return true;
    }

    bool save(){
        string filename = "./Data/" + modelType;
        fstream file(filename.c_str(), ios::out | ios::trunc);
        if(!file.is_open())
            return false;
        
        int N, M;
        N = TRANS.rows;
        M = TRANS.cols;

        file << N << "\t" << M;
        file << endl;
        for(int r=0; r<TRANS.rows; r++)
            for(int c=0; c<TRANS.cols; c++)
                file << TRANS.at<double>(r,c) << "\t";
        file << endl;

        N = EMIS.rows;
        M = EMIS.cols;
        file << N << "\t" << M;
        file << endl;
        for(int r=0; r<EMIS.rows; r++)
            for(int c=0; c<EMIS.cols; c++)
                file << EMIS.at<double>(r,c) << "\t";
        file << endl;

        N = INIT.rows;
        M = INIT.cols;
        file << N << "\t" << M;
        file << endl;
        for(int r=0; r<INIT.rows; r++)
            for(int c=0; c<INIT.cols; c++)
                file << INIT.at<double>(r,c) << "\t";
        file << endl;

        return true;
    }

    void train(Mat &seq, int max_iter){
        CvHMM cvhmm;
        cvhmm.train(seq, max_iter, TRANS, EMIS, INIT);
    }

    double testProbability(Mat &seq){
        CvHMM cvhmm;
        cv::Mat STATES,FORWARD,BACKWARD;
        double logpseq;
        cvhmm.decode(seq, TRANS, EMIS, INIT, logpseq, STATES, FORWARD, BACKWARD);

        return logpseq;
    }

    void print(){
        CvHMM cvhmm;
        cvhmm.printModel(TRANS,EMIS,INIT);
    }
};