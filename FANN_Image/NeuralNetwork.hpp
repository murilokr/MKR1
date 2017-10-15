#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <floatfann.h>
#include <string>
#include <fstream>

int fileColumns(string filename, int line){
    int max = -1;

    fstream file(filename.c_str(), ios::in);
    if(!file.is_open()){
        cerr << filename << " could not be opened" << endl;
        return max;
    }

    int count = 0;
    string check;
    for(int i = 0; i < line; ++i)
        getline(file, check);

    file.close();
    return check.size()/2;
}


enum HC{
    HC_notDefined = -1,
    HC_advance = 0,
    HC_return = 1,
    HC_zoomIn = 2,
    HC_zoomOut = 3
};

string HC_ToString(HC handConfig){
    string retVal = "";
    switch(handConfig){
        case 0:
            retVal = "Advance HandConfiguration";
            break;
        case 1:
            retVal = "Return HandConfiguration";
            break;
        case 2:
            retVal = "Zoom-In HandConfiguration";
            break;
        case 3:
            retVal = "Zoom-Out HandConfiguration";
            break;
        default:
            retVal = "Undefined HandConfiguration";
            break;
    }
    return retVal;
}

class HandConfiguration{

    private:
        string str_netPath;
        struct fann *ann;

        HC returnOutput(fann_type *calc_out, int size){
            fann_type max = -999;
            int enumIt = 0;

            for(int i = 0; i < size; i++)
                if(calc_out[i] > max){
                    max = calc_out[i];
                    enumIt = i;
                }
            

            HC retVal;
            switch(enumIt){
                case 0:
                    retVal = HC_advance;
                    break;
                case 1:
                    retVal = HC_return;
                    break;
                case 2:
                    retVal = HC_zoomIn;
                    break;
                case 3:
                    retVal = HC_zoomOut;
                    break;
                default:
                    retVal = HC_notDefined;
                    break;
            }
            return retVal;
        }
        
    public:
        HandConfiguration(string netPath) : str_netPath(netPath){}
        ~HandConfiguration(){
            fann_destroy(ann);
        }

        void loadNet(){
            ann = fann_create_from_file(str_netPath.c_str());
        }

        HC evaluate(Mat inpMat){
            fann_type *calc_out;
            fann_type input[inpMat.cols * inpMat.rows];

            int k = 0;
            for(int r = 0; r < inpMat.rows; r++){
                for(int c = 0; c < inpMat.cols; c++){
                    input[k] = inpMat.at<int>(r,c);
                    k++;
                }
                k++;
            }

            calc_out = fann_run(ann, input);
            return returnOutput(calc_out, 4);
        }

        bool train(string filename){
            const unsigned int num_input = fileColumns(filename, 2);
            if(num_input == -1)
                cout << "aff" << endl;
            const unsigned int num_output = 4;
            const unsigned int num_layers = 3;
            const unsigned int num_neurons_hidden = (num_input+num_output)/2;
            const float desired_error = (const float) 0.001;
            const unsigned int max_epochs = 1000;
            const unsigned int epochs_between_reports = 15;

            struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

            fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
            fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

            fann_train_on_file(ann, filename.c_str(), max_epochs, epochs_between_reports, desired_error);

            fann_save(ann, str_netPath.c_str());

            fann_destroy(ann);
            return true;
        }

};

#endif //NEURALNETWORK_HPP