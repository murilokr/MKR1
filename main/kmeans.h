#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <math.h>

using namespace std;

struct Centroids{
    float rightVectorX, rightVectorY, rightVectorZ, rightHandConfiguration;
    float leftVectorX, leftVectorY, leftVectorZ, leftHandConfiguration;
}typedef Centroids;


class KMeans{

    private:
        vector<Centroids> *codebook;

        void ReadFromFile(fstream& file){
            if(!file.is_open())
                return;
            

            float rVx, rVy, rVz, rHc;
            float lVx, lVy, lVz, lHc;

            while(!file.eof()){
                file >> rVx >> rVy >> rVz >> rHc;
                file >> lVx >> lVy >> lVz >> lHc;

                Centroids c = {rVx, rVy, rVz, rHc, lVx, lVy, lVz, lHc};
                codebook->push_back(c);
            }
        }

    public:
        KMeans(){
            codebook = new vector<Centroids>();
        };
        
        KMeans(vector<Centroids>* c){
            codebook = c;
        }

        KMeans(fstream& file){
            codebook = new vector<Centroids>();
            ReadFromFile(file);
        }



        int getClusterNumber(){
            return codebook->size();
        }

        vector<Centroids>* returnCentroids(){
            return codebook;
        }


        bool isEmpty(){
            return codebook->empty();
        }


        int GetNearestCluster(Centroids coordinates){
            if(isEmpty())
                return -1;

            //Distancia para cada cluster
            int clstNumb = 0, currClst = 0;
            float minDst = 10000;

            for(vector<Centroids>::iterator cluster = codebook->begin(); cluster != codebook->end(); ++cluster){
                float dXr = (*cluster).rightVectorX - coordinates.rightVectorX;                        
                dXr *= dXr;
                float dYr = (*cluster).rightVectorY - coordinates.rightVectorY;
                dYr *= dYr;
                float dZr = (*cluster).rightVectorZ - coordinates.rightVectorZ;
                dZr *= dZr;
                float dHCr = (*cluster).rightHandConfiguration - coordinates.rightHandConfiguration;
                dHCr *= dHCr;

                float dXl = (*cluster).leftVectorX - coordinates.leftVectorX;
                dXl *= dXl;
                float dYl = (*cluster).leftVectorY - coordinates.leftVectorY;
                dYl *= dYl;
                float dZl = (*cluster).leftVectorZ - coordinates.leftVectorZ;
                dZl *= dZl;
                float dHCl = (*cluster).leftHandConfiguration - coordinates.leftHandConfiguration;
                dHCl *= dHCl;

                float d = sqrt(dXr + dYr + dZr + dHCr + dXl + dYl + dZl + dHCl);
                if(d < minDst){
                    minDst = d;
                    clstNumb = currClst;
                }
                currClst++;
            }

            return currClst;
        }

        vector<int>* returnObservations(vector<Centroids>* clusters){

            vector<int>* observations = new vector<int>();

            for(vector<Centroids>::iterator coord = clusters->begin(); coord != clusters->end(); ++coord){
                int code = GetNearestCluster(*coord);
                observations->push_back(code);
            }

            return observations;
        }
};