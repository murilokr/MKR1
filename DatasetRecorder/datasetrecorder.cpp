/*****************************************************************************
*                                                                            *
*  OpenNI 1.x Alpha                                                          *
*  Copyright (C) 2012 PrimeSense Ltd.                                        *
*                                                                            *
*  This file is part of OpenNI.                                              *
*                                                                            *
*  Licensed under the Apache License, Version 2.0 (the "License");           *
*  you may not use this file except in compliance with the License.          *
*  You may obtain a copy of the License at                                   *
*                                                                            *
*      http://www.apache.org/licenses/LICENSE-2.0                            *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*****************************************************************************/
//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------
#include <XnCppWrapper.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <fstream>
#include <iostream>

//---------------------------------------------------------------------------
// Defines
//---------------------------------------------------------------------------
#define SAMPLE_XML_PATH "../../../../Data/SamplesConfig.xml"
#define SAMPLE_XML_PATH_LOCAL "SamplesConfig.xml"

//---------------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------------
xn::Context g_Context;
xn::ScriptNode g_scriptNode;
xn::UserGenerator g_UserGenerator;
xn::DepthGenerator g_DepthGenerator;

XnBool g_bNeedPose = FALSE;
XnChar g_strPose[20] = "";

std::string relation_name = "gesture";

#define MAX_NUM_USERS 15

//---------------------------------------------------------------------------
// Namespaces
//---------------------------------------------------------------------------
using namespace xn;
using namespace cv;
using namespace std;

//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------

bool isFileEmpty(fstream& file){
    file.seekg(0, ios::end);
    return file.tellg() == 0;
}


typedef struct Frame{
    public:
        int rightHandX, rightHandY, rightHandZ, handConfigurationRight;
        int leftHandX, leftHandY, leftHandZ, handConfigurationLeft;
        int torsoX, torsoY, torsoZ;
        int headX, headY;
        int htRatioX, htRatioY;
        float rightVectorX, rightVectorY, rightVectorZ;
        float leftVectorX, leftVectorY, leftVectorZ;

    
    Frame(int rhX, int rhY, int rhZ, int hc1, int lhX, int lhY, int lhZ, int hc2, int tX, int tY, int tZ, int hX, int hY){
        rightHandX = rhX;
        rightHandY = rhY;
        rightHandZ = rhZ;
        handConfigurationRight = hc1;

        leftHandX = lhX;
        leftHandY = lhY;
        leftHandZ = lhZ;
        handConfigurationLeft = hc2;


        torsoX = tX;
        torsoY = tY;
        torsoZ = tZ;

        headX = hX;
        headY = hY;

        htRatioX = sqrt(pow(torsoX - headX, 2));
        htRatioY = sqrt(pow(torsoY - headY, 2));
        
        //Descomentar para usar valores não negativos
        /*rightVectorX = sqrt(pow(rightHandX - torsoX, 2)); //Para não deixar o valor negativo
        rightVectorY = sqrt(pow(rightHandY - torsoY, 2)); //Para não deixar o valor negativo
        rightVectorZ = sqrt(pow(rightHandZ - torsoZ, 2)); //Para não deixar o valor negativo

        leftVectorX = sqrt(pow(leftHandX - torsoX, 2)); //Para não deixar o valor negativo
        leftVectorY = sqrt(pow(leftHandY - torsoY, 2)); //Para não deixar o valor negativo
        leftVectorZ = sqrt(pow(leftHandZ - torsoZ, 2)); //Para não deixar o valor negativo*/

        rightVectorX = rightHandX - torsoX;
        rightVectorY = rightHandY - torsoY;
        rightVectorZ = rightHandZ - torsoZ;

        leftVectorX = leftHandX - torsoX;
        leftVectorY = leftHandY - torsoY;
        leftVectorZ = leftHandZ - torsoZ;

        int htRatio = sqrt(pow(htRatioX, 2) + pow(htRatioY, 2)); //Magnitude do Ratio da Cabeça e Tronco

        rightVectorX /= htRatio;
        rightVectorY /= htRatio;


        leftVectorX /= htRatio;
        leftVectorY /= htRatio;
    }

    void WriteToFile(fstream& file){
        if(isFileEmpty(file))
            WriteArff(file);
        file << rightVectorX << "\t" << rightVectorY << "\t" << rightVectorZ << "\t" << handConfigurationRight << "\t"; 
        file << leftVectorX << "\t" << leftVectorY << "\t" << leftVectorZ << "\t" << handConfigurationLeft << endl;
    }

    void WriteArff(fstream & file){
        file << "@RELATION " << relation_name << endl;
        file << "@ATTRIBUTE rightVectorX numeric" << endl << "@ATTRIBUTE rightVectorY numeric" << endl << "@ATTRIBUTE rightVectorZ numeric" << endl << "@ATTRIBUTE handConfigurationRight numeric" << endl; 
        file << "@ATTRIBUTE leftVectorX numeric" << endl << "@ATTRIBUTE leftVectorY numeric" << endl << "@ATTRIBUTE leftVectorZ numeric" << endl <<"@ATTRIBUTE handConfigurationLeft numeric" << endl;
        file << "@DATA" << endl;
    }

}Frame;











XnBool fileExists(const char *fn)
{
    XnBool exists;
    xnOSDoesFileExist(fn, &exists);
    return exists;
}

// Callback: New user was detected
void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& /*generator*/, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d New User %d\n", epochTime, nId);
    // New user found
    if (g_bNeedPose)
    {
        g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
    }
    else
    {
        g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
    }
}
// Callback: An existing user was lost
void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& /*generator*/, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Lost user %d\n", epochTime, nId);
}
// Callback: Detected a pose
void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& /*capability*/, const XnChar* strPose, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Pose %s detected for user %d\n", epochTime, strPose, nId);
    g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
    g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}
// Callback: Started calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& /*capability*/, XnUserID nId, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    printf("%d Calibration started for user %d\n", epochTime, nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationComplete(xn::SkeletonCapability& /*capability*/, XnUserID nId, XnCalibrationStatus eStatus, void* /*pCookie*/)
{
    XnUInt32 epochTime = 0;
    xnOSGetEpochTime(&epochTime);
    if (eStatus == XN_CALIBRATION_STATUS_OK)
    {
        // Calibration succeeded
        printf("%d Calibration complete, start tracking user %d\n", epochTime, nId);        
        g_UserGenerator.GetSkeletonCap().StartTracking(nId);
    }
    else
    {
        // Calibration failed
        printf("%d Calibration failed for user %d\n", epochTime, nId);
        if(eStatus==XN_CALIBRATION_STATUS_MANUAL_ABORT)
        {
            printf("Manual abort occured, stop attempting to calibrate!");
            return;
        }
        if (g_bNeedPose)
        {
            g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
        }
        else
        {
            g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
        }
    }
}


#define CHECK_RC(nRetVal, what)                     \
    if (nRetVal != XN_STATUS_OK)                    \
{                                   \
    printf("%s failed: %s\n", what, xnGetStatusString(nRetVal));    \
    return nRetVal;                         \
}


XnStatus InicializarKinect(int *argc, char * argv[]){

    XnStatus nRetVal = XN_STATUS_OK;
    xn::EnumerationErrors errors;

    const char *fn = NULL;
    if    (fileExists(SAMPLE_XML_PATH)) fn = SAMPLE_XML_PATH;
    else if (fileExists(SAMPLE_XML_PATH_LOCAL)) fn = SAMPLE_XML_PATH_LOCAL;
    else {
        printf("Could not find '%s' nor '%s'. Aborting.\n" , SAMPLE_XML_PATH, SAMPLE_XML_PATH_LOCAL);
        return XN_STATUS_ERROR;
    }
    printf("Reading config from: '%s'\n", fn);

    nRetVal = g_Context.InitFromXmlFile(fn, g_scriptNode, &errors);
    if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
    {
        XnChar strError[1024];
        errors.ToString(strError, 1024);
        printf("%s\n", strError);
        return (nRetVal);
    }
    else if (nRetVal != XN_STATUS_OK)
    {
        printf("Open failed: %s\n", xnGetStatusString(nRetVal));
        return (nRetVal);
    }

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        nRetVal = g_UserGenerator.Create(g_Context);
        CHECK_RC(nRetVal, "Find user generator");
    }


    ////////////////////////////////////Acha o depth generator ou cria se não existir
    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        nRetVal = g_DepthGenerator.Create(g_Context);
        CHECK_RC(nRetVal, "Find depth generator");
    }
    ////////////////////////////////////

    XnCallbackHandle hUserCallbacks, hCalibrationStart, hCalibrationComplete, hPoseDetected;
    if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON))
    {
        printf("Supplied user generator doesn't support skeleton\n");
        return XN_STATUS_ERROR;
    }
    nRetVal = g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);
    CHECK_RC(nRetVal, "Register to user callbacks");
    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationStart(UserCalibration_CalibrationStart, NULL, hCalibrationStart);
    CHECK_RC(nRetVal, "Register to calibration start");
    nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationComplete(UserCalibration_CalibrationComplete, NULL, hCalibrationComplete);
    CHECK_RC(nRetVal, "Register to calibration complete");

    if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration())
    {
        g_bNeedPose = TRUE;
        if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION))
        {
            printf("Pose required, but not supported\n");
            return XN_STATUS_ERROR;
        }
        nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseDetected(UserPose_PoseDetected, NULL, hPoseDetected);
        CHECK_RC(nRetVal, "Register to Pose Detected");
        g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
    }

    g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

    nRetVal = g_Context.StartGeneratingAll();
    CHECK_RC(nRetVal, "StartGenerating");

    return nRetVal;
}




cv::Mat adjustDepth(const cv::Mat& inImage)
{
    // from https://orbbec3d.com/product-astra/
    // Astra S has a depth in the range 0.35m to 2.5m
    int maxDepth = 2500;
    int minDepth = 350; // in mm

    cv::Mat retImage = inImage;

    for(int j = 0; j < retImage.rows; j++)
        for(int i = 0; i < retImage.cols; i++)
        {
            if(retImage.at<ushort>(j, i))
                retImage.at<ushort>(j, i) = maxDepth - (retImage.at<ushort>(j, i) - minDepth);
        }

        return retImage;
}


void Histogram(cv::Mat src){
  if( !src.data )
   return;

  /// Separate the image in 3 places ( B, G and R )
  std::vector<cv::Mat> bgr_planes;
  cv::split( src, bgr_planes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  cv::Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = round( (double) hist_w/histSize );

  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  cv::normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  cv::normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  cv::normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - roundf(b_hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - roundf(b_hist.at<float>(i)) ),
                       cv::Scalar( 255, 0, 0), 2, 8, 0  );
      cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - roundf(g_hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - roundf(g_hist.at<float>(i)) ),
                       cv::Scalar( 0, 255, 0), 2, 8, 0  );
      cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - roundf(r_hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - roundf(r_hist.at<float>(i)) ),
                       cv::Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  cv::imshow("calcHist Demo", histImage );
}


int main(int argc, char * argv[]){

    XnStatus err = InicializarKinect(&argc, argv);
    if(err != XN_STATUS_OK)
        return -1;


    string filename = "./Dataset/data.arff";
    std::fstream dataset(filename.c_str(), std::fstream::out | std::fstream::app);
    if(!dataset.is_open()){
        cerr << "Error opening " << filename << endl;
        return -2;
    }


    cv::VideoCapture capture(CV_CAP_OPENNI);
    capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
    if(!capture.isOpened()){
        printf("Couldn't load Kinect into OPENCV\n");
        return -1;
    }


    cout << endl << endl;
    cout << "+-----------Depth DataSet Generator-----------+" << endl; 
	cout << "|     Copyright Murilo Kinape Rivabem 2017    |" << endl;
	cout << "|                                             |" << endl;
    cout << "|     Press Space to Start/Stop Recording     |" << endl;
	cout << "|       Press Esc to exit the program         |" << endl;
	cout << "+---------------------------------------------+" << endl;
    cout << endl << endl;


    XnUserID aUsers[MAX_NUM_USERS];
    XnUInt16 nUsers;
    
    ///Joint Positions
    //XnSkeletonJointPosition jointPosition;
    XnSkeletonJointTransformation jointPosition;

    cv::Mat imageMap;
    cv::Mat depthMap;
    cv::Mat maos;
    float roiSize = 75; //Tamanho da area de interesse das mãos
    int frame = -50;

    bool endit = false;
    bool isRecording = false;


    cv::namedWindow("RGB", 1);
    cv::namedWindow("Depth", 1);
    cv::namedWindow("ROI", 1);
    

    if(g_bNeedPose)
        printf("Assume calibration pose\n");
    

    while(!endit){
        while (xnOSWasKeyboardHit()){

            char c = xnOSReadCharFromInput();
            if(c == 27) endit = !endit;
            else if (c == 32) isRecording = !isRecording;
        }

        capture.grab();

        capture.retrieve(imageMap, CV_CAP_OPENNI_BGR_IMAGE);
        capture.retrieve(depthMap, CV_CAP_OPENNI_DISPARITY_MAP); //É o que mais se encaixa 


        //depthMap = adjustDepth(depthMap);
        g_Context.WaitOneUpdateAll(g_UserGenerator);
        // print the torso information for the first user already tracking
        nUsers=MAX_NUM_USERS;
        g_UserGenerator.GetUsers(aUsers, nUsers);
        for(XnUInt16 i=0; i<nUsers; i++)
        {
            if(g_UserGenerator.GetSkeletonCap().IsTracking(aUsers[i])==FALSE)
                continue;

            //g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i],XN_SKEL_LEFT_HAND,torsoJoint);
            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i], XN_SKEL_TORSO, jointPosition);
            XnPoint3D aProjective;
            g_DepthGenerator.ConvertRealWorldToProjective(1,&jointPosition.position.position, &aProjective);
            int torsoX = aProjective.X;
            int torsoY = aProjective.Y;
            int torsoZ = aProjective.Z;



            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i], XN_SKEL_HEAD, jointPosition);
            g_DepthGenerator.ConvertRealWorldToProjective(1,&jointPosition.position.position, &aProjective);
            int headX = aProjective.X;
            int headY = aProjective.Y;


            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i], XN_SKEL_RIGHT_HAND, jointPosition);
            g_DepthGenerator.ConvertRealWorldToProjective(1,&jointPosition.position.position, &aProjective);
            int rHandX = aProjective.X;
            int rHandY = aProjective.Y;
            int rHandZ = aProjective.Z;


            g_UserGenerator.GetSkeletonCap().GetSkeletonJoint(aUsers[i], XN_SKEL_LEFT_HAND, jointPosition);
            g_DepthGenerator.ConvertRealWorldToProjective(1,&jointPosition.position.position, &aProjective);
            int lHandX = aProjective.X;
            int lHandY = aProjective.Y;
            int lHandZ = aProjective.Z;


            //Frame(int rhX, int rhY, int rhZ, int hc1, int lhX, int lhY, int lhZ, int hc2, int tX, int tY, int tZ, int hX, int hY)
            Frame cFrame(rHandX, rHandY, rHandZ, 1, lHandX, lHandY, lHandZ, 1, torsoX, torsoY, torsoZ, headX, headY);
            cout << frame << ": " << cFrame.rightVectorX << "\t" << cFrame.rightVectorY << "\t" << cFrame.rightVectorZ << "\t" << cFrame.handConfigurationRight << "\t" << cFrame.leftVectorX << "\t" << cFrame.leftVectorY << "\t" << cFrame.leftVectorZ << "\t" << cFrame.handConfigurationLeft << endl;
            if(isRecording){
                if(frame >= 0){// && frame < 100){
                    cFrame.WriteToFile(dataset);    
                }
                frame++;
            }

            /////Pega a posicao da mão em coordenadas da tela
            /*XnPoint3D pt;
            if(jointPosition.fConfidence > 0.5)
                pt = jointPosition.position;
            else{
                pt.X = 0.0;
                pt.Y = 0.0;
                pt.Z = 0.0;
            }
            g_DepthGenerator.ConvertRealWorldToProjective(1, &pt, &pt);*/
            /////
            
            ///////Cria uma janela em volta das mãos
            cv::Rect roi(rHandX - (roiSize/2), rHandY - (roiSize/2), roiSize, roiSize);
            cv::Rect roiImg(0, 0, imageMap.cols, imageMap.rows);

            
            //Checa para ver se esta "dentro" da imagem
            if( (roi.area() > 0) && ((roiImg & roi).area() == roi.area()) )
                maos = depthMap(roi);    
            ///////


            //Segmenta a mao
            threshold(maos, maos, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


            //Draw skeleton
            line(imageMap, Point(headX, headY), Point(torsoX, torsoY), Scalar(255,255,255),2);
            line(imageMap, Point(torsoX, torsoY), Point(lHandX, lHandY), Scalar(0,255,0), 2);
            line(imageMap, Point(torsoX, torsoY), Point(rHandX, rHandY), Scalar(0,255,0), 2);
        }

        cv::imshow("RGB", imageMap);
        cv::imshow("Depth", depthMap);
        if(!maos.empty())
            cv::imshow("ROI", maos);
        char esc = cv::waitKey(33);
		if (esc == 27) break;
		else if (esc == 32) isRecording = !isRecording;
    }
    
    dataset.close();
    g_scriptNode.Release();
    g_UserGenerator.Release();
    g_DepthGenerator.Release();
    g_Context.Release();

}
