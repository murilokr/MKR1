#include "kmeans.h"

int main(int argc, char* argv[]){

    string filename;
    if(argc == 2){
        filename = argv[1];
    }
    else filename = "./Dataset/codebook.txt";
    fstream data(filename.c_str(), ios::in);
    
    KMeans *Codebook = new KMeans(data);


}